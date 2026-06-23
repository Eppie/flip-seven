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
#include <thread>
#include <vector>

namespace flip7 {

inline constexpr int kRoundScoreMax = 78;  // numbers-only ceiling: {6..12}=63 + 15

// Split the inclusive integer range [lo,hi] across hardware threads and call
// body(s,e) once per contiguous chunk. Used to parallelize one coordinate-sum
// layer of the N=3 win grids -- every cell in a layer is independent (it reads
// only strictly-higher-sum cells, already finalized), so this is exact and
// order-independent: the per-cell floating-point work is identical to the serial
// path regardless of how chunks are assigned. Each thread allocates its own
// scratch inside body, so there is no sharing. Serial fallback for small ranges.
template <class F>
inline void parallel_chunks(int lo, int hi, F&& body) {
    const int total = hi - lo + 1;
    if (total <= 0) return;
    unsigned T = std::thread::hardware_concurrency();
    if (!T) T = 1;
    if (T == 1 || total < 32) { body(lo, hi); return; }
    if ((unsigned)total < T) T = (unsigned)total;
    std::vector<std::thread> th;
    th.reserve(T);
    const int chunk = (total + (int)T - 1) / (int)T;
    for (unsigned t = 0; t < T; ++t) {
        const int s = lo + (int)t * chunk, e = std::min(hi, s + chunk - 1);
        if (s > e) break;
        th.emplace_back([s, e, &body] { body(s, e); });
    }
    for (auto& x : th) x.join();
}

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

// max E[g(round score)] over within-round Hit/Stay policies. Optional outputs:
//   hit[S]    the chosen policy (1 = Hit), if hit != nullptr
//   B[S],b0   B[S] = dU[S]/dg[0]; b0 = B[0] = P(bust) under the optimal policy.
// B/b0 let the win-probability self-loop (g[0] = base0 + D0*w, w = U[0]) be
// solved in closed form: round_solve is linear in g[0] within a fixed policy
// region, so w* = (U0 - B0*D0*w)/(1 - B0*D0) jumps to the exact fixed point and
// the loop converges in ~1-2 solves instead of ~6-20 plain iterations.
inline double round_solve(const double* g, double* U, uint8_t* hit = nullptr,
                          double* B = nullptr, double* b0 = nullptr) {
    const double g0 = g[0];
    for (int S : g_order) {
        const int pc = g_pc[S];
        if (pc == kFlip7Target) { U[S] = g[g_sum[S] + kFlip7Bonus]; if (B) B[S] = 0.0; continue; }
        const double stay = g[g_sum[S]];
        double accU = g_bustnum[S] * g0;            // bust branch: reward g[0]
        double accB = B ? g_bustnum[S] : 0.0;       // d/dg0 of that branch
        for (unsigned nm = (~(unsigned)S) & 0x1FFFu; nm; nm &= nm - 1) {
            const int v = __builtin_ctz(nm);
            const double c = (double)numberCount(v);
            accU += c * U[S | (1u << v)];
            if (B) accB += c * B[S | (1u << v)];
        }
        const double T = (double)(kNumberDeckSize - pc);
        const double hv = accU / T;
        if (hv > stay) { U[S] = hv;  if (hit) hit[S] = 1; if (B) B[S] = accB / T; }
        else           { U[S] = stay; if (hit) hit[S] = 0; if (B) B[S] = (g_sum[S] == 0 ? 1.0 : 0.0); }
    }
    if (b0) *b0 = B ? B[0] : 0.0;
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

// Exact N-player win-probability grid (N in {2,3}); player 0 is the reasoning
// player. Ds[i] is player i's round-score pmf (allowing a heterogeneous field).
// W[index(t)] = P(player 0 wins) at round-start total vector t, each t_i in
// [0,target), row-major:
//   N=2: index = a*target + b           (same layout as win_prob_greedy)
//   N=3: index = (a*target + b)*target + c
// A round is terminal once any total reaches target; the win goes to the max
// total with ties split 1/k among the k players sharing it. The all-bust
// self-loop (every player scores 0, probability prod_i Ds[i][0]) is folded by
// the closing divide, exactly as the 2-player win_prob_greedy. This is the
// generic generalization of win_prob_greedy; the dedicated 2-player function is
// kept verbatim as the validated path and this is asserted to match it at N=2
// (tests). Intended for N<=3 (target^N states); returns empty for larger N
// (use the Monte-Carlo tournament instead).
inline std::vector<double> win_prob_greedy_n(const std::vector<std::vector<double>>& Ds,
                                             int target) {
    const int N = (int)Ds.size();
    std::vector<std::vector<int>> sup(N);
    double selfprob = 1.0;
    for (int i = 0; i < N; ++i) { sup[i] = pmf_support(Ds[i]); selfprob *= Ds[i][0]; }
    const double inv_self = 1.0 / (1.0 - selfprob);

    if (N == 2) {
        const auto& D0 = Ds[0]; const auto& D1 = Ds[1];
        std::vector<double> W((size_t)target * target, 0.0);
        for (int S = 2 * (target - 1); S >= 0; --S) {
            const int alo = std::max(0, S - (target - 1));
            const int ahi = std::min(target - 1, S);
            for (int a = alo; a <= ahi; ++a) {
                const int b = S - a;
                double num = 0.0;
                for (int x : sup[0]) for (int y : sup[1]) {
                    if (x == 0 && y == 0) continue;
                    const double p = D0[x] * D1[y];
                    const int A = a + x, B = b + y;
                    if (A >= target || B >= target) num += p * (A > B ? 1.0 : (A == B ? 0.5 : 0.0));
                    else                            num += p * W[(size_t)A * target + B];
                }
                W[(size_t)a * target + b] = num * inv_self;
            }
        }
        return W;
    }

    if (N == 3) {
        const auto& D0 = Ds[0]; const auto& D1 = Ds[1]; const auto& D2 = Ds[2];
        const size_t T = (size_t)target;
        std::vector<double> W(T * T * T, 0.0);
        auto idx = [T](int a, int b, int c) -> size_t { return ((size_t)a * T + b) * T + c; };
        for (int S = 3 * (target - 1); S >= 0; --S) {
            const int alo = std::max(0, S - 2 * (target - 1));
            const int ahi = std::min(target - 1, S);
            parallel_chunks(alo, ahi, [&](int as, int ae) {        // layer S: cells independent
              for (int a = as; a <= ae; ++a) {
                const int rem = S - a;                          // b + c
                const int blo = std::max(0, rem - (target - 1));
                const int bhi = std::min(target - 1, rem);
                for (int b = blo; b <= bhi; ++b) {
                    const int c = rem - b;                       // in [0,target)
                    double num = 0.0;
                    for (int x : sup[0]) { const int A = a + x; const double p0 = D0[x];
                      for (int y : sup[1]) { const int B = b + y; const double p01 = p0 * D1[y];
                        for (int z : sup[2]) {
                            if (x == 0 && y == 0 && z == 0) continue;   // all-bust self-loop
                            const int C = c + z;
                            const double p = p01 * D2[z];
                            if (A >= target || B >= target || C >= target) {
                                const int M = std::max(A, std::max(B, C));
                                if (A == M) num += p / ((A == M) + (B == M) + (C == M));
                            } else {
                                num += p * W[idx(A, B, C)];
                            }
                        }
                      }
                    }
                    W[idx(a, b, c)] = num * inv_self;
                }
              }
            });
        }
        return W;
    }

    fprintf(stderr, "win_prob_greedy_n: exact grid only supports N<=3 (got N=%d); use MC.\n", N);
    return {};
}

// Best-response win grid for N players (N in {2,3}): player 0 re-optimizes the
// within-round Hit/Stay policy every round to maximize WIN PROBABILITY, while the
// other n-1 players draw from the fixed field pmf D (greedy). Returns W over the
// n-D grid (player 0 = axis 0; same indexing as win_prob_greedy_n). The within-
// round solver round_solve is reused UNCHANGED -- only the terminal reward
// g[round_score] = E[win value of total a+x | opponents draw from D] changes, and
// the all-bust self-loop folds in closed form with selfprob = D[0]^(n-1):
//   g[0] = base0 + selfprob*w,  w* = (U0 - B0*selfprob*w)/(1 - B0*selfprob).
// init_round_tables() must have been called. Cost is ~target^n * |sup|^(n-1) plus
// a round_solve per cell -- fast for n=2, a multi-minute one-off for n=3.
// One best-response cell: solve the closed-form all-bust self-loop in place. g[]
// is the terminal reward with g[0] holding base0 (the self term excluded); U/Bc
// are caller-owned scratch (one set per thread). Returns the win value w*.
inline double br_solve_cell(double* g, double* U, double* Bc, double selfprob, double warm) {
    const double base0 = g[0];
    double w = warm;
    for (int it = 0; it < 12; ++it) {
        g[0] = base0 + selfprob * w;
        double B0;
        const double U0 = round_solve(g, U, nullptr, Bc, &B0);
        const double denom = 1.0 - B0 * selfprob;
        const double wn = (denom > 1e-12) ? (U0 - B0 * selfprob * w) / denom : U0;
        if (std::fabs(wn - w) < 1e-12) { w = wn; break; }
        w = wn;
    }
    return w;
}

inline std::vector<double> best_response_grid_n(const std::vector<double>& D, int target, int n) {
    const auto sup = pmf_support(D);
    double selfprob = 1.0; for (int i = 1; i < n; ++i) selfprob *= D[0];

    if (n == 2) {                                              // fast; kept serial (validated path)
        std::vector<double> U(1 << kNumValues), Bc(1 << kNumValues), g(kRoundScoreMax + 1);
        std::vector<double> W((size_t)target * target, 0.0);
        for (int S = 2 * (target - 1); S >= 0; --S) {
            const int alo = std::max(0, S - (target - 1)), ahi = std::min(target - 1, S);
            for (int a = alo; a <= ahi; ++a) {
                const int b = S - a;
                for (int x = 0; x <= kRoundScoreMax; ++x) {
                    const int A = a + x; double gx = 0.0;
                    for (int y : sup) {
                        const int B = b + y;
                        gx += D[y] * ((A >= target || B >= target)
                                          ? (A > B ? 1.0 : (A == B ? 0.5 : 0.0))
                                          : W[(size_t)A * target + B]);
                    }
                    g[x] = gx;
                }
                W[(size_t)a * target + b] = br_solve_cell(g.data(), U.data(), Bc.data(), selfprob, 0.5);
            }
        }
        return W;
    }

    if (n == 3) {
        const size_t T = (size_t)target;
        std::vector<double> W(T * T * T, 0.0);
        auto idx = [T](int a, int b, int c) { return ((size_t)a * T + b) * T + c; };
        for (int S = 3 * (target - 1); S >= 0; --S) {
            const int alo = std::max(0, S - 2 * (target - 1)), ahi = std::min(target - 1, S);
            parallel_chunks(alo, ahi, [&](int as, int ae) {    // per-thread scratch below
              std::vector<double> U(1 << kNumValues), Bc(1 << kNumValues), g(kRoundScoreMax + 1);
              for (int a = as; a <= ae; ++a) {
                const int rem = S - a;
                const int blo = std::max(0, rem - (target - 1)), bhi = std::min(target - 1, rem);
                for (int b = blo; b <= bhi; ++b) {
                    const int c = rem - b;
                    for (int x = 0; x <= kRoundScoreMax; ++x) {
                        const int A = a + x; double gx = 0.0;
                        for (int y : sup) { const int B = b + y; const double dy = D[y];
                            for (int z : sup) {
                                const int C = c + z; const double p = dy * D[z];
                                if (A >= target || B >= target || C >= target) {
                                    const int M = std::max(A, std::max(B, C));
                                    if (A == M) gx += p / ((A == M) + (B == M) + (C == M));
                                } else {
                                    gx += p * W[idx(A, B, C)];
                                }
                            }
                        }
                        g[x] = gx;
                    }
                    W[idx(a, b, c)] = br_solve_cell(g.data(), U.data(), Bc.data(), selfprob, 1.0 / 3.0);
                }
              }
            });
        }
        return W;
    }

    fprintf(stderr, "best_response_grid_n: exact only supports N<=3 (got N=%d); use MC.\n", n);
    return {};
}

}  // namespace flip7
