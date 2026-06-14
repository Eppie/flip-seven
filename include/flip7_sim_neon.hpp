// flip7_sim_neon.hpp — NEON-vectorized numbers-only Monte-Carlo rollouts.
//
// The scalar simulator runs one rollout at a time, drawing from a physical
// shuffled deck. That deck makes the loop sequential and resists SIMD (each lane
// would need an independent random gather, which NEON has no instruction for). The
// fix is to drop the deck: drawing WITHOUT replacement from a fresh 79-card number
// deck, given the held set S (popcount p), the next card is value v with
// probability (count(v) - [v in S]) / (79 - p) -- exactly the conditional the
// exact DP uses. With no deck, the rollout state is just the 13-bit mask, so we
// run EIGHT independent rollouts in lockstep (one per 16-bit lane, since the mask
// and the <=78 scores both fit in 16 bits) with per-lane alive predicates; when a
// lane ends (Stay / bust / Flip 7) we bank its score and reset that lane to a
// fresh rollout, so all eight lanes stay busy.
//
// The per-draw cost is a 13-step categorical decode (vs. the scalar's O(1) deck
// pop), so the win comes from width: 8 lanes amortize that decode, plus deferred
// horizontal reductions (accumulate in vector lanes, reduce every 8192 steps). The
// result is statistically identical to the scalar simulator and is cross-checked
// against it and the exact DP. Non-ARM builds fall back to the scalar simulator.
#pragma once
#include "flip7_core.hpp"
#include "flip7_dp.hpp"
#include "flip7_sim.hpp"   // MCResult + scalar fallback

#include <cmath>
#include <cstdint>

#if defined(__ARM_NEON)
#include <arm_neon.h>

namespace flip7 {

// Four independent xoshiro128++ streams (state transposed: register k holds word k
// of all four lanes). Two of these feed the eight rollout lanes.
struct Xoshiro128ppx4 {
    uint32x4_t s0, s1, s2, s3;
    void seed(uint64_t seed) {
        uint64_t x = seed;
        auto sm = [&] {
            x += 0x9E3779B97F4A7C15ULL;
            uint64_t z = x;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            return z ^ (z >> 31);
        };
        uint32_t w[4][4];
        for (int lane = 0; lane < 4; ++lane)
            for (int k = 0; k < 4; ++k) w[k][lane] = (uint32_t)sm();
        s0 = vld1q_u32(w[0]); s1 = vld1q_u32(w[1]);
        s2 = vld1q_u32(w[2]); s3 = vld1q_u32(w[3]);
    }
    template <int K> static inline uint32x4_t rotl(uint32x4_t v) {
        return vorrq_u32(vshlq_n_u32(v, K), vshrq_n_u32(v, 32 - K));
    }
    inline uint32x4_t next() {
        const uint32x4_t res = vaddq_u32(rotl<7>(vaddq_u32(s0, s3)), s0);
        const uint32x4_t t   = vshlq_n_u32(s1, 9);
        s2 = veorq_u32(s2, s0); s3 = veorq_u32(s3, s1);
        s1 = veorq_u32(s1, s2); s0 = veorq_u32(s0, s3);
        s2 = veorq_u32(s2, t);  s3 = rotl<11>(s3);
        return res;
    }
};

// NEON numbers-only MC. Same semantics as monte_carlo_solitaire; uses dp.hit[] as
// the policy (pass an all-true table for the Flip-7-maximizing run). If hist != null
// (length kRoundScoreMax+1), it receives the realized score histogram (counts) for a
// full-distribution cross-check against the exact pmf.
inline MCResult monte_carlo_solitaire_neon(const SolitaireTurnDP& dp, uint64_t n, uint64_t seed,
                                           double* hist = nullptr) {
    Xoshiro128ppx4 rngA, rngB;
    rngA.seed(seed);
    rngB.seed(seed ^ 0xD1B54A32D192ED03ULL);

    uint16_t basev[kNumValues];
    for (int v = 0; v < kNumValues; ++v) basev[v] = (uint16_t)numberCount(v);

    const uint16x8_t one = vdupq_n_u16(1);
    const uint16x8_t v79 = vdupq_n_u16(kNumberDeckSize);
    const uint16x8_t v7  = vdupq_n_u16(kFlip7Target);
    const uint16x8_t v15 = vdupq_n_u16(kFlip7Bonus);

    uint16x8_t mask  = vdupq_n_u16(0);   // held set per lane
    uint16x8_t score = vdupq_n_u16(0);   // running sum of held values per lane
    uint16x8_t pop   = vdupq_n_u16(0);   // popcount(mask), tracked incrementally

    // Deferred reductions: accumulate in 32-bit lanes, fold to scalars every 8192
    // steps (a lane gains <= 78 per step for sum, so 8192 steps stays well < 2^32).
    uint32x4_t accSum = vdupq_n_u32(0), accSq = vdupq_n_u32(0);
    uint16x8_t cStay = vdupq_n_u16(0), cBust = vdupq_n_u16(0), cF7 = vdupq_n_u16(0);
    double sum = 0.0, sumsq = 0.0;
    uint64_t completed = 0;
    long busts = 0, flip7s = 0, stays = 0;

    auto flush = [&] {                       // fold lane accumulators to scalars (every 8192 steps)
        sum   += (double)(vaddvq_u32(accSum));
        sumsq += (double)(vaddvq_u32(accSq));
        stays  += vaddvq_u32(vpaddlq_u16(cStay));
        busts  += vaddvq_u32(vpaddlq_u16(cBust));
        flip7s += vaddvq_u32(vpaddlq_u16(cF7));
        completed = (uint64_t)(stays + busts + flip7s);
        accSum = accSq = vdupq_n_u32(0);
        cStay = cBust = cF7 = vdupq_n_u16(0);
    };

    for (int step = 0;; ++step) {
        if ((step & 8191) == 0) { flush(); if (completed >= n) break; }

        // Remaining-deck size and the Hit/Stay decision, from the pre-draw mask.
        const uint16x8_t T = vsubq_u16(v79, pop);            // cards left = 79 - popcount

        uint16_t hm[8]; vst1q_u16(hm, mask);                 // policy gather (8 scalar loads)
        const uint16_t hv[8] = {
            (uint16_t)(dp.hit[hm[0]] ? 0xFFFFu : 0), (uint16_t)(dp.hit[hm[1]] ? 0xFFFFu : 0),
            (uint16_t)(dp.hit[hm[2]] ? 0xFFFFu : 0), (uint16_t)(dp.hit[hm[3]] ? 0xFFFFu : 0),
            (uint16_t)(dp.hit[hm[4]] ? 0xFFFFu : 0), (uint16_t)(dp.hit[hm[5]] ? 0xFFFFu : 0),
            (uint16_t)(dp.hit[hm[6]] ? 0xFFFFu : 0), (uint16_t)(dp.hit[hm[7]] ? 0xFFFFu : 0) };
        const uint16x8_t hit = vld1q_u16(hv);

        // Unbiased bounded draw r in [0,T): (rand32 * T) >> 32 per lane (bias < 2^-25).
        const uint32x4_t rA = rngA.next(), rB = rngB.next();
        const uint32x4_t Tlo = vmovl_u16(vget_low_u16(T)), Thi = vmovl_u16(vget_high_u16(T));
        const uint32x4_t rlo = vcombine_u32(
            vshrn_n_u64(vmull_u32(vget_low_u32(rA),  vget_low_u32(Tlo)),  32),
            vshrn_n_u64(vmull_u32(vget_high_u32(rA), vget_high_u32(Tlo)), 32));
        const uint32x4_t rhi = vcombine_u32(
            vshrn_n_u64(vmull_u32(vget_low_u32(rB),  vget_low_u32(Thi)),  32),
            vshrn_n_u64(vmull_u32(vget_high_u32(rB), vget_high_u32(Thi)), 32));
        const uint16x8_t r = vcombine_u16(vmovn_u32(rlo), vmovn_u32(rhi));

        // Categorical decode: pick value v whose cumulative remaining range holds r.
        // Running-remainder trick: keep rr = r - cum_excl[v] (and msh = mask>>v), so
        // "cum<=r<cum+rem" collapses to one unsigned compare rr<rem (rr underflows to
        // a huge value past the match, killing later compares; rem==0 self-skips).
        uint16x8_t rr = r, msh = mask, drawn = vdupq_n_u16(0);
        for (int v = 0; v < kNumValues; ++v) {
            const uint16x8_t rem = vsubq_u16(vdupq_n_u16(basev[v]), vandq_u16(msh, one));
            const uint16x8_t inr = vcltq_u16(rr, rem);
            drawn = vaddq_u16(drawn, vandq_u16(inr, vdupq_n_u16((uint16_t)v)));
            rr  = vsubq_u16(rr, rem);
            msh = vshrq_n_u16(msh, 1);
        }

        const uint16x8_t bit   = vshlq_u16(one, vreinterpretq_s16_u16(drawn));   // 1<<v
        const uint16x8_t dup   = vtstq_u16(mask, bit);                           // duplicate => bust
        const uint16x8_t add   = vbicq_u16(hit, dup);                            // Hit and not a dup
        score = vaddq_u16(score, vandq_u16(add, drawn));
        mask  = vorrq_u16(mask, vandq_u16(add, bit));
        pop   = vsubq_u16(pop, add);                                             // popcount + (added?1:0)
        const uint16x8_t flip7 = vceqq_u16(pop, v7);

        const uint16x8_t stay   = vmvnq_u16(hit);
        const uint16x8_t busted = vandq_u16(hit, dup);
        const uint16x8_t f7end  = vandq_u16(add, flip7);
        const uint16x8_t ended  = vorrq_u16(stay, vorrq_u16(busted, f7end));

        uint16x8_t fin = vaddq_u16(score, vandq_u16(f7end, v15));   // +15 on Flip 7
        fin = vbicq_u16(fin, busted);                               // bust -> 0
        const uint16x8_t finE = vandq_u16(ended, fin);

        accSum = vaddq_u32(accSum, vpaddlq_u16(finE));
        accSq  = vaddq_u32(accSq,  vpaddlq_u16(vandq_u16(ended, vmulq_u16(fin, fin))));
        cStay  = vsubq_u16(cStay,  stay);    // predicate is 0xFFFF (= -1) => add 1
        cBust  = vsubq_u16(cBust,  busted);
        cF7    = vsubq_u16(cF7,    f7end);

        if (hist) {                                                 // full-distribution check (off the hot path)
            uint16_t fv[8], ev[8];
            vst1q_u16(fv, fin); vst1q_u16(ev, ended);
            for (int i = 0; i < 8; ++i) if (ev[i]) hist[fv[i]] += 1.0;
        }

        mask  = vbicq_u16(mask,  ended);                            // reset ended lanes
        score = vbicq_u16(score, ended);
        pop   = vbicq_u16(pop,   ended);
    }

    MCResult r;
    r.n = completed;
    r.mean = sum / (double)completed;
    double var = sumsq / (double)completed - r.mean * r.mean;
    if (var < 0) var = 0;
    r.stddev = std::sqrt(var);
    r.stderr_ = std::sqrt(var / (double)completed);
    r.p_bust = (double)busts / (double)completed;
    r.p_flip7 = (double)flip7s / (double)completed;
    r.p_stay = (double)stays / (double)completed;
    r.busts = busts; r.flip7s = flip7s; r.stays = stays;
    return r;
}

}  // namespace flip7

#else  // ---- non-ARM: portable fallback to the scalar simulator ----
namespace flip7 {
inline MCResult monte_carlo_solitaire_neon(const SolitaireTurnDP& dp, uint64_t n, uint64_t seed,
                                           double* hist = nullptr) {
    (void)hist;
    return monte_carlo_solitaire(dp, n, seed);
}
}  // namespace flip7
#endif
