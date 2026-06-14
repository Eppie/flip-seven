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

}  // namespace flip7
