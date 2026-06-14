// flip7_sim.hpp — independent Monte-Carlo simulator (cross-checks the DP).
//
// Chapter 1: solitaire single turn, draw WITHOUT replacement from a fresh
// 79-card number deck via partial Fisher-Yates. A persistent deck array is
// restored after each turn by undoing the (<=7) swaps, so there is no
// per-rollout O(deck) shuffle cost.
#pragma once
#include "flip7_core.hpp"
#include "flip7_dp.hpp"
#include "flip7_rng.hpp"
#include <cmath>
#include <cstdint>
#include <cstring>

namespace flip7 {

struct MCResult {
    double   mean, stddev, stderr_;
    double   p_bust, p_flip7, p_stay;
    uint64_t n;
    long     busts, flip7s, stays;
};

inline MCResult monte_carlo_solitaire(const SolitaireTurnDP& dp, uint64_t n, uint64_t seed) {
    Xoshiro256pp rng;
    rng.seed(seed);

    uint8_t deck[kNumberDeckSize];
    int idx = 0;
    for (int v = 0; v < kNumValues; ++v)
        for (int k = 0, c = numberCount(v); k < c; ++k) deck[idx++] = (uint8_t)v;
    // idx == 79

    double sum = 0.0, sumsq = 0.0;
    long busts = 0, flip7s = 0, stays = 0;
    int sp[8], sj[8];  // recorded swaps for O(draws) deck restore

    for (uint64_t i = 0; i < n; ++i) {
        uint16_t mask = 0;
        int pos = 0, ns = 0;
        double score = 0.0;
        for (;;) {
            if (!dp.hit[mask]) { score = (double)maskSum(mask); ++stays; break; }
            // Draw uniformly from the remaining cards [pos, 78].
            const uint64_t j = pos + rng.bounded((uint64_t)(kNumberDeckSize - pos));
            const uint8_t card = deck[pos];
            deck[pos] = deck[j];
            deck[j] = card;
            sp[ns] = pos;
            sj[ns] = (int)j;
            ++ns;
            const int v = deck[pos];
            ++pos;
            const uint16_t bit = (uint16_t)(1u << v);
            if (mask & bit) { score = 0.0; ++busts; break; }      // duplicate -> bust
            mask |= bit;
            if (maskPop(mask) == kFlip7Target) {                  // 7 uniques -> Flip 7
                score = (double)(maskSum(mask) + kFlip7Bonus);
                ++flip7s;
                break;
            }
        }
        for (int s = ns - 1; s >= 0; --s) {  // restore fresh deck
            const uint8_t t = deck[sp[s]];
            deck[sp[s]] = deck[sj[s]];
            deck[sj[s]] = t;
        }
        sum += score;
        sumsq += score * score;
    }

    MCResult r;
    r.n = n;
    r.busts = busts;
    r.flip7s = flip7s;
    r.stays = stays;
    r.mean = sum / (double)n;
    double var = sumsq / (double)n - r.mean * r.mean;
    if (var < 0) var = 0;
    r.stddev = std::sqrt(var);
    r.stderr_ = std::sqrt(var / (double)n);
    r.p_bust = (double)busts / (double)n;
    r.p_flip7 = (double)flip7s / (double)n;
    r.p_stay = (double)stays / (double)n;
    return r;
}

struct MCFullResult {
    double   mean, stddev, stderr_;
    double   p_bust, p_flip7, p_stay, p_saved;
    uint64_t n;
};

// --- Stage b.1: numbers + modifiers (deck 85, no Second Chance) --------------
// Card codes: 0..12 numbers, 13..18 modifiers (index = code-13).
inline MCFullResult monte_carlo_mod(const SolitaireModDP& dp, uint64_t n, uint64_t seed) {
    Xoshiro256pp rng;
    rng.seed(seed);
    const int total = SolitaireModDP::kDeckTotal;  // 85

    uint8_t deck[SolitaireModDP::kDeckTotal];
    int idx = 0;
    for (int v = 0; v < kNumValues; ++v)
        for (int k = 0, c = numberCount(v); k < c; ++k) deck[idx++] = (uint8_t)v;
    for (int i = 0; i < kNumModifiers; ++i) deck[idx++] = (uint8_t)(13 + i);

    double sum = 0.0, sumsq = 0.0;
    long busts = 0, flip7s = 0, stays = 0;
    int sp[128], sj[128];

    for (uint64_t t = 0; t < n; ++t) {
        uint16_t nm = 0, mm = 0;
        int pos = 0, ns = 0;
        double score = 0.0;
        for (;;) {
            if (!dp.hit[SolitaireModDP::enc(nm, mm)]) { score = (double)fullScore(nm, mm); ++stays; break; }
            const uint64_t j = pos + rng.bounded((uint64_t)(total - pos));
            { const uint8_t tmp = deck[pos]; deck[pos] = deck[j]; deck[j] = tmp; }
            sp[ns] = pos; sj[ns] = (int)j; ++ns;
            const uint8_t card = deck[pos];
            ++pos;
            if (card < 13) {
                const uint16_t bit = (uint16_t)(1u << card);
                if (nm & bit) { score = 0.0; ++busts; break; }   // duplicate -> bust
                nm |= bit;
                if (maskPop(nm) == kFlip7Target) { score = (double)fullScore(nm, mm); ++flip7s; break; }
            } else {
                mm |= (uint16_t)(1u << (card - 13));
            }
        }
        for (int s = ns - 1; s >= 0; --s) {
            const uint8_t tmp = deck[sp[s]]; deck[sp[s]] = deck[sj[s]]; deck[sj[s]] = tmp;
        }
        sum += score;
        sumsq += score * score;
    }

    MCFullResult r;
    r.n = n;
    r.mean = sum / (double)n;
    double var = sumsq / (double)n - r.mean * r.mean;
    if (var < 0) var = 0;
    r.stddev = std::sqrt(var);
    r.stderr_ = std::sqrt(var / (double)n);
    r.p_bust = (double)busts / (double)n;
    r.p_flip7 = (double)flip7s / (double)n;
    r.p_stay = (double)stays / (double)n;
    r.p_saved = 0.0;
    return r;
}

// --- Stage b.2: numbers + modifiers + Second Chance (deck 88, real rules) ----
// Card codes: 0..12 numbers, 13..18 modifiers, 19 = Second Chance. A save
// removes the duplicate (deck depletes) and increments the per-value extra count
// so the exact policy can be looked up by the full state.
inline MCFullResult monte_carlo_full(const SolitaireFullDP& dp, uint64_t n, uint64_t seed) {
    Xoshiro256pp rng;
    rng.seed(seed);
    const int total = SolitaireFullDP::kDeckTotal;  // 88

    uint8_t deck[SolitaireFullDP::kDeckTotal];
    int idx = 0;
    for (int v = 0; v < kNumValues; ++v)
        for (int k = 0, c = numberCount(v); k < c; ++k) deck[idx++] = (uint8_t)v;
    for (int i = 0; i < kNumModifiers; ++i)     deck[idx++] = (uint8_t)(13 + i);
    for (int i = 0; i < kNumSecondChance; ++i)  deck[idx++] = (uint8_t)19;

    double sum = 0.0, sumsq = 0.0;
    long busts = 0, flip7s = 0, stays = 0, saved = 0;
    int sp[256], sj[256];

    for (uint64_t t = 0; t < n; ++t) {
        uint16_t nm = 0, mm = 0;
        int sch = 0, scn = 0, pos = 0, ns = 0;
        uint32_t extra = 0;
        double score = 0.0;
        bool didsave = false;
        for (;;) {
            if (!dp.policy(nm, mm, sch, scn, extra)) { score = (double)fullScore(nm, mm); ++stays; break; }
            const uint64_t j = pos + rng.bounded((uint64_t)(total - pos));
            { const uint8_t tmp = deck[pos]; deck[pos] = deck[j]; deck[j] = tmp; }
            sp[ns] = pos; sj[ns] = (int)j; ++ns;
            const uint8_t card = deck[pos];
            ++pos;
            if (card < 13) {
                const int v = card;
                const uint16_t bit = (uint16_t)(1u << v);
                if (nm & bit) {                              // duplicate
                    if (sch) {                               // Second Chance saves (dup removed)
                        sch = 0;
                        didsave = true;
                        extra = SolitaireFullDP::setExtra(extra, v, SolitaireFullDP::extraOf(extra, v) + 1);
                    } else { score = 0.0; ++busts; break; }  // bust
                } else {
                    nm |= bit;
                    if (maskPop(nm) == kFlip7Target) { score = (double)fullScore(nm, mm); ++flip7s; break; }
                }
            } else if (card < 19) {
                mm |= (uint16_t)(1u << (card - 13));
            } else {
                ++scn;
                if (!sch) sch = 1;   // else excess discarded
            }
        }
        for (int s = ns - 1; s >= 0; --s) {
            const uint8_t tmp = deck[sp[s]]; deck[sp[s]] = deck[sj[s]]; deck[sj[s]] = tmp;
        }
        if (didsave) ++saved;
        sum += score;
        sumsq += score * score;
    }

    MCFullResult r;
    r.n = n;
    r.mean = sum / (double)n;
    double var = sumsq / (double)n - r.mean * r.mean;
    if (var < 0) var = 0;
    r.stddev = std::sqrt(var);
    r.stderr_ = std::sqrt(var / (double)n);
    r.p_bust = (double)busts / (double)n;
    r.p_flip7 = (double)flip7s / (double)n;
    r.p_stay = (double)stays / (double)n;
    r.p_saved = (double)saved / (double)n;
    return r;
}

}  // namespace flip7
