// flip7_compete.hpp — across-rounds (first-to-target) machinery for Chapter 4.
//
// Numbers-only model: each round a player banks a score drawn from a round-score
// distribution; totals accumulate; the first to reach >= target at a round's end
// wins (higher total; tie split). Action-card targeting is deferred (Ch. 5), so
// rounds are independent.
#pragma once
#include "flip7_core.hpp"
#include "flip7_dp.hpp"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace flip7 {

inline constexpr int kRoundScoreMax = 78;  // numbers-only ceiling: {6..12}=63 + 15

// Round-score pmf D[0..78] under the expected-score-optimal numbers-only policy.
inline void round_pmf(double D[kRoundScoreMax + 1]) {
    SolitaireTurnDP dp;
    dp.optimal();
    for (int s = 0; s <= kRoundScoreMax; ++s) D[s] = 0.0;
    std::vector<double> reach(1 << kNumValues, 0.0);
    reach[0] = 1.0;
    for (int pc = 0; pc < kFlip7Target; ++pc)
        for (uint32_t S = 0; S < (1u << kNumValues); ++S) {
            if (maskPop((uint16_t)S) != pc) continue;
            const double pr = reach[S];
            if (pr == 0.0) continue;
            if (!dp.hit[S]) { D[maskSum((uint16_t)S)] += pr; continue; }
            const int T = kNumberDeckSize - pc;
            double bust = 0;
            for (int v = 0; v < kNumValues; ++v) if (S & (1u << v)) bust += numberCount(v) - 1;
            D[0] += pr * bust / (double)T;
            for (int v = 0; v < kNumValues; ++v) {
                const uint16_t bit = (uint16_t)(1u << v);
                if (S & bit) continue;
                const double pv = pr * (double)numberCount(v) / (double)T;
                const uint16_t Sn = (uint16_t)(S | bit);
                if (maskPop(Sn) == kFlip7Target) D[maskSum(Sn) + kFlip7Bonus] += pv;
                else                             reach[Sn] += pv;
            }
        }
}

// Support: scores with positive probability.
inline std::vector<int> pmf_support(const double D[kRoundScoreMax + 1]) {
    std::vector<int> s;
    for (int i = 0; i <= kRoundScoreMax; ++i) if (D[i] > 0) s.push_back(i);
    return s;
}

// Expected number of greedy rounds to reach >= target from 0.
inline double expected_rounds_to_target(const double D[kRoundScoreMax + 1], int target) {
    const auto sup = pmf_support(D);
    std::vector<double> V(target + 1, 0.0);
    for (int c = target - 1; c >= 0; --c) {
        double acc = 1.0;
        for (int s : sup) if (s) acc += D[s] * V[std::min(c + s, target)];
        V[c] = acc / (1.0 - D[0]);
    }
    return V[0];
}

// 2-player win-probability grid, both playing greedy. W[a*target+b] = P(the
// reasoning player wins) at round start with totals (a,b), both < target.
inline std::vector<double> win_prob_greedy(const double D[kRoundScoreMax + 1], int target) {
    const auto sup = pmf_support(D);
    std::vector<double> W((size_t)target * target, 0.0);
    const double self = D[0] * D[0];
    for (int sum = 2 * (target - 1); sum >= 0; --sum) {
        const int alo = std::max(0, sum - (target - 1));
        const int ahi = std::min(target - 1, sum);
        for (int a = alo; a <= ahi; ++a) {
            const int b = sum - a;
            double num = 0.0;
            for (int x : sup) for (int y : sup) {
                if (x == 0 && y == 0) continue;
                const double p = D[x] * D[y];
                const int A = a + x, B = b + y;
                if (A >= target || B >= target) num += p * (A > B ? 1.0 : (A == B ? 0.5 : 0.0));
                else                            num += p * W[(size_t)A * target + B];
            }
            W[(size_t)a * target + b] = num / (1.0 - self);
        }
    }
    return W;
}

}  // namespace flip7
