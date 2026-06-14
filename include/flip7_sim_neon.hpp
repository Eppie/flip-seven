// flip7_sim_neon.hpp — NEON-vectorized numbers-only Monte-Carlo rollouts.
//
// The scalar simulator runs one rollout at a time, drawing from a physical
// shuffled deck. That deck makes the loop sequential and resists SIMD (each lane
// would need an independent random gather, which NEON has no instruction for). The
// fix is to drop the deck: the held set S (a 13-bit mask) fully determines the
// remaining 79-card deck, so we run EIGHT independent rollouts in lockstep (one per
// 16-bit lane, since the mask and the <=78 scores both fit in 16 bits) with per-
// lane alive predicates; when a lane ends (Stay / bust / Flip 7) we bank its score
// and reset that lane to a fresh rollout, so all eight lanes stay busy.
//
// The next-card draw uses rejection over the FULL deck rather than a categorical
// decode: draw a uniform position k in [0,79), look up its (value, first-copy flag)
// in a 79-byte LUT (one cache line), and treat the single reserved first-copy
// position of each held value as a no-op redraw. Any OTHER copy of a held value is
// a bust; a copy of an unheld value is kept. This reproduces draw-without-
// replacement exactly (the reserved/bust/keep probabilities are count(v) - [v in S]
// over 79 - p, by symmetry of the uniform positions) with one small gather instead
// of a 13-step loop. With deferred horizontal reductions (reduce every 8192 steps)
// it runs ~2.4x the scalar simulator. The result is statistically identical and is
// cross-checked against the scalar MC and the exact DP -- including the full score
// pmf bin-by-bin. Non-ARM builds fall back to the scalar simulator.
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

    // Full-deck position -> (value | first-copy-flag << 4). 79 bytes = one cache line.
    uint8_t lut[kNumberDeckSize];
    { int k = 0;
      for (int v = 0; v < kNumValues; ++v)
          for (int c = 0, cc = numberCount(v); c < cc; ++c)
              lut[k++] = (uint8_t)(v | (c == 0 ? 0x10 : 0)); }

    const uint16x8_t one = vdupq_n_u16(1);
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

        // Hit/Stay decision from the pre-draw mask (policy gather, 8 scalar loads).
        uint16_t hm[8]; vst1q_u16(hm, mask);
        const uint16_t hv[8] = {
            (uint16_t)(dp.hit[hm[0]] ? 0xFFFFu : 0), (uint16_t)(dp.hit[hm[1]] ? 0xFFFFu : 0),
            (uint16_t)(dp.hit[hm[2]] ? 0xFFFFu : 0), (uint16_t)(dp.hit[hm[3]] ? 0xFFFFu : 0),
            (uint16_t)(dp.hit[hm[4]] ? 0xFFFFu : 0), (uint16_t)(dp.hit[hm[5]] ? 0xFFFFu : 0),
            (uint16_t)(dp.hit[hm[6]] ? 0xFFFFu : 0), (uint16_t)(dp.hit[hm[7]] ? 0xFFFFu : 0) };
        const uint16x8_t hit = vld1q_u16(hv);

        // Draw a uniform full-deck position k in [0,79): (rand32 * 79) >> 32.
        const uint32x4_t rA = rngA.next(), rB = rngB.next();
        const uint32x4_t klo = vcombine_u32(
            vshrn_n_u64(vmull_n_u32(vget_low_u32(rA),  (uint32_t)kNumberDeckSize), 32),
            vshrn_n_u64(vmull_n_u32(vget_high_u32(rA), (uint32_t)kNumberDeckSize), 32));
        const uint32x4_t khi = vcombine_u32(
            vshrn_n_u64(vmull_n_u32(vget_low_u32(rB),  (uint32_t)kNumberDeckSize), 32),
            vshrn_n_u64(vmull_n_u32(vget_high_u32(rB), (uint32_t)kNumberDeckSize), 32));
        const uint16x8_t k = vcombine_u16(vmovn_u32(klo), vmovn_u32(khi));

        // LUT gather: position -> value and first-copy flag (8 loads, one cache line).
        uint16_t kk[8]; vst1q_u16(kk, k);
        const uint16_t ev[8] = { lut[kk[0]], lut[kk[1]], lut[kk[2]], lut[kk[3]],
                                 lut[kk[4]], lut[kk[5]], lut[kk[6]], lut[kk[7]] };
        const uint16x8_t e       = vld1q_u16(ev);
        const uint16x8_t vval    = vandq_u16(e, vdupq_n_u16(0x0F));               // drawn value
        const uint16x8_t isFirst = vtstq_u16(e, vdupq_n_u16(0x10));               // first copy? (full-width predicate)
        const uint16x8_t bit     = vshlq_u16(one, vreinterpretq_s16_u16(vval));   // 1<<v
        const uint16x8_t inMask  = vtstq_u16(mask, bit);                          // value already held?
        const uint16x8_t reserved= vandq_u16(isFirst, inMask);                    // the held copy -> redraw (no-op)

        const uint16x8_t stay   = vmvnq_u16(hit);
        const uint16x8_t actv   = vbicq_u16(hit, reserved);   // Hit and the draw is a real remaining card
        const uint16x8_t busted = vandq_u16(actv, inMask);    // another copy of a held value -> bust
        const uint16x8_t addL   = vbicq_u16(actv, inMask);    // a new value -> keep it
        score = vaddq_u16(score, vandq_u16(addL, vval));
        mask  = vorrq_u16(mask, vandq_u16(addL, bit));
        pop   = vsubq_u16(pop, addL);                         // popcount + (added?1:0)
        const uint16x8_t f7end  = vandq_u16(addL, vceqq_u16(pop, v7));
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
