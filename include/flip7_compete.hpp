// flip7_compete.hpp — across-rounds (first-to-target) machinery for Chapter 4.
//
// Numbers-only model: each round a player banks a score drawn from a round-score
// distribution; totals accumulate; the first to reach >= target at a round's end
// wins (higher total; tie split). Action-card targeting is deferred (Ch. 5), so
// rounds are independent. The round-score pmf is supplied as a vector indexed by
// score, so the same code serves the numbers-only game and the full 94-card game.
#pragma once
#include "flip7_core.hpp"
#include "flip7_dp.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace flip7 {

inline constexpr int kRoundScoreMax = 78;  // numbers-only ceiling: {6..12}=63 + 15

// Round-score pmf D[0..78] under the expected-score-optimal numbers-only policy.
inline std::vector<double> round_pmf_numbers() {
    SolitaireTurnDP dp;
    dp.optimal();
    std::vector<double> D(kRoundScoreMax + 1, 0.0);
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
    return D;
}

// Load a "score probability" pmf file (e.g. the all-94 distribution). Returns
// empty if the file is missing.
inline std::vector<double> load_pmf(const char* path) {
    std::vector<double> D;
    FILE* f = std::fopen(path, "r");
    if (!f) return D;
    char line[256];
    while (std::fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        int s; double p;
        if (std::sscanf(line, "%d %lf", &s, &p) == 2 && s >= 0) {
            if ((int)D.size() <= s) D.resize(s + 1, 0.0);
            D[s] = p;
        }
    }
    std::fclose(f);
    return D;
}

inline std::vector<int> pmf_support(const std::vector<double>& D) {
    std::vector<int> s;
    for (int i = 0; i < (int)D.size(); ++i) if (D[i] > 0) s.push_back(i);
    return s;
}

// Expected number of greedy rounds to reach >= target from 0.
inline double expected_rounds_to_target(const std::vector<double>& D, int target) {
    const auto sup = pmf_support(D);
    std::vector<double> V(target + 1, 0.0);
    for (int c = target - 1; c >= 0; --c) {
        double acc = 1.0;
        for (int s : sup) if (s) acc += D[s] * V[std::min(c + s, target)];
        V[c] = acc / (1.0 - D[0]);
    }
    return V[0];
}

// --- within-round solver with an arbitrary terminal reward g[round_score] -----
// Numbers-only game; reused by the win-probability best response (reward = win
// value of the new total) and profiled directly.
inline int    g_sum[1 << kNumValues], g_pc[1 << kNumValues];
inline double g_bustnum[1 << kNumValues];
inline std::vector<int> g_order;  // states by popcount descending

inline void init_round_tables() {
    for (uint32_t S = 0; S < (1u << kNumValues); ++S) {
        g_sum[S] = maskSum((uint16_t)S);
        g_pc[S]  = maskPop((uint16_t)S);
        double b = 0;
        for (int v = 0; v < kNumValues; ++v) if (S & (1u << v)) b += numberCount(v) - 1;
        g_bustnum[S] = b;
    }
    g_order.resize(1 << kNumValues);
    for (uint32_t S = 0; S < (1u << kNumValues); ++S) g_order[S] = (int)S;
    std::sort(g_order.begin(), g_order.end(), [](int a, int b) { return g_pc[a] > g_pc[b]; });
}

// max E[g(round score)] over within-round Hit/Stay policies; fills hit[] if set.
inline double round_solve(const double* g, double* U, uint8_t* hit) {
    const double g0 = g[0];
    for (int S : g_order) {
        const int pc = g_pc[S];
        if (pc == kFlip7Target) { U[S] = g[g_sum[S] + kFlip7Bonus]; continue; }
        const double stay = g[g_sum[S]];
        double acc = g_bustnum[S] * g0;
        for (unsigned nm = (~(unsigned)S) & 0x1FFFu; nm; nm &= nm - 1) {
            const int v = __builtin_ctz(nm);
            acc += (double)numberCount(v) * U[S | (1u << v)];
        }
        const double hv = acc / (double)(kNumberDeckSize - pc);
        if (hv > stay) { U[S] = hv;  if (hit) hit[S] = 1; }
        else           { U[S] = stay; if (hit) hit[S] = 0; }
    }
    return U[0];
}

// Round-score pmf realized by a within-round policy hit[] (numbers-only).
inline void round_dist(const uint8_t* hit, double* outD /*[kRoundScoreMax+1]*/) {
    for (int s = 0; s <= kRoundScoreMax; ++s) outD[s] = 0.0;
    std::vector<double> reach(1 << kNumValues, 0.0);
    reach[0] = 1.0;
    for (int pc = 0; pc < kFlip7Target; ++pc)
        for (uint32_t S = 0; S < (1u << kNumValues); ++S) {
            if (g_pc[S] != pc) continue;
            const double pr = reach[S];
            if (pr == 0.0) continue;
            if (!hit[S]) { outD[g_sum[S]] += pr; continue; }
            const int T = kNumberDeckSize - pc;
            outD[0] += pr * g_bustnum[S] / (double)T;
            for (unsigned nm = (~(unsigned)S) & 0x1FFFu; nm; nm &= nm - 1) {
                const int v = __builtin_ctz(nm);
                const double pv = pr * (double)numberCount(v) / (double)T;
                const uint16_t Sn = (uint16_t)(S | (1u << v));
                if (g_pc[Sn] == kFlip7Target) outD[g_sum[Sn] + kFlip7Bonus] += pv;
                else                          reach[Sn] += pv;
            }
        }
}

// 2-player win-probability grid, both playing greedy. W[a*target+b] = P(the
// reasoning player wins) at round start with totals (a,b), both < target.
inline std::vector<double> win_prob_greedy(const std::vector<double>& D, int target) {
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
