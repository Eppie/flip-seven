// flip7_actions.hpp — Chapter 5 exact action-card analysis (numbers-only layer).
//
// Adversarial targeting of Freeze and Flip Three. The exact, MC-checkable layer
// models the *effect* of an externally-injected action card on a single
// numbers-only player who otherwise plays the expected-score-optimal policy:
//
//   Freeze(S)      -- the target is forced to Stay at hand S. Round score = sum(S),
//                     deterministic. Its competitive value is the points it DENIES
//                     vs. optimal continuation: damage(S) = EV(S) - sum(S) >= 0.
//   FlipThree(S)   -- the target is forced to draw 3 cards from S (ending early on
//                     bust / Flip 7), then resumes optimal play. This RAISES the
//                     target's score when S is shallow (free cards, low dup risk)
//                     and LOWERS it when S is deep (high dup risk) -- a gift that
//                     becomes an attack past a crossover popcount k*.
//
// Every quantity here comes from an exact forward DP and is cross-checked against
// an independent numbers-only Monte-Carlo (mc_numbers_forced) before it is
// reported, per the project's discipline. Requires init_round_tables() first
// (g_sum / g_pc / g_bustnum live in flip7_compete.hpp).
#pragma once
#include "flip7_compete.hpp"
#include "flip7_core.hpp"
#include "flip7_rng.hpp"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace flip7 {

// numbers-only expected-score-optimal policy: fills U[S]=EV(S) and hit[S].
inline void numbers_opt_policy(std::vector<double>& U, std::vector<uint8_t>& hit) {
    std::vector<double> g(kRoundScoreMax + 1);
    for (int s = 0; s <= kRoundScoreMax; ++s) g[s] = (double)s;
    U.assign(1u << kNumValues, 0.0);
    hit.assign(1u << kNumValues, 0);
    round_solve(g.data(), U.data(), hit.data());
}

// Round-score pmf when the active-hand distribution starts at init[S] (each with
// `forced0` forced draws), then plays optimally per hit[]. `out` length
// kRoundScoreMax+1 is OVERWRITTEN; returns the total Flip-7 probability mass.
// Forward enumeration over (forced_remaining, hand); acyclic because every draw
// adds a card. init need not be normalized (out integrates to sum(init)).
inline double pmf_from_dist(const double* init, int forced0, const uint8_t* hit, double* out) {
    constexpr int N = 1 << kNumValues;
    for (int s = 0; s <= kRoundScoreMax; ++s) out[s] = 0.0;
    std::vector<double> reach((size_t)4 * N, 0.0);  // reach[f*N + S], f in 0..3
    const int f0 = forced0 < 0 ? 0 : (forced0 > 3 ? 3 : forced0);
    for (int S = 0; S < N; ++S)
        if (init[S] != 0.0) reach[(size_t)f0 * N + S] = init[S];

    double pf7 = 0.0;
    for (int pc = 0; pc < kFlip7Target; ++pc) {
        for (int S = 0; S < N; ++S) {
            if (g_pc[S] != pc) continue;
            for (int f = 0; f <= 3; ++f) {
                const double pr = reach[(size_t)f * N + S];
                if (pr == 0.0) continue;
                const bool draw = (f > 0) || hit[S];
                if (!draw) { out[g_sum[S]] += pr; continue; }       // voluntary Stay
                const int newf = (f > 0) ? f - 1 : 0;
                const int T = kNumberDeckSize - pc;
                out[0] += pr * g_bustnum[S] / (double)T;            // bust
                for (unsigned nm = (~(unsigned)S) & 0x1FFFu; nm; nm &= nm - 1) {
                    const int v = __builtin_ctz(nm);
                    const double pv = pr * (double)numberCount(v) / (double)T;
                    const int Sn = S | (1 << v);
                    if (g_pc[Sn] == kFlip7Target) { out[g_sum[Sn] + kFlip7Bonus] += pv; pf7 += pv; }
                    else                            reach[(size_t)newf * N + Sn] += pv;
                }
            }
        }
    }
    return pf7;
}

// Convenience: pmf starting from a single hand `start` with `forced0` forced draws.
inline double pmf_forced(uint16_t start, int forced0, const uint8_t* hit, double* out) {
    std::vector<double> init(1u << kNumValues, 0.0);
    init[start] = 1.0;
    return pmf_from_dist(init.data(), forced0, hit, out);
}

// Round-score pmf when exactly one Flip Three (3 forced draws) is injected at the
// target's FIRST natural Stay point, then optimal play resumes. Models the
// adversary flipping the opponent at the moment they would bank -- the strongest
// single-card timing against an optimal player. `out` overwritten; returns P(Flip 7).
inline double pmf_flip3_at_stop(const uint8_t* hit, double* out) {
    constexpr int N = 1 << kNumValues;
    for (int s = 0; s <= kRoundScoreMax; ++s) out[s] = 0.0;
    // reach[(used*4 + f)*N + S]; used = whether the one Flip Three already fired.
    std::vector<double> reach((size_t)2 * 4 * N, 0.0);
    reach[0] = 1.0;  // (used=0, f=0, S=0)
    auto at = [&](int u, int f, int S) -> double& { return reach[(((size_t)u * 4 + f) * N) + S]; };

    double pf7 = 0.0;
    for (int pc = 0; pc < kFlip7Target; ++pc) {
        for (int S = 0; S < N; ++S) {
            if (g_pc[S] != pc) continue;
            for (int u = 0; u < 2; ++u)
                for (int f = 0; f <= 3; ++f) {
                    const double pr = at(u, f, S);
                    if (pr == 0.0) continue;
                    int uu = u, newf;
                    if (f > 0)            { newf = f - 1; }          // mid forced sequence
                    else if (hit[S])      { newf = 0; }              // optimal Hit
                    else if (u == 0)      { newf = 2; uu = 1; }      // inject Flip Three (this draw is #1 of 3)
                    else                  { out[g_sum[S]] += pr; continue; }  // real Stay
                    const int T = kNumberDeckSize - pc;
                    out[0] += pr * g_bustnum[S] / (double)T;
                    for (unsigned nm = (~(unsigned)S) & 0x1FFFu; nm; nm &= nm - 1) {
                        const int v = __builtin_ctz(nm);
                        const double pv = pr * (double)numberCount(v) / (double)T;
                        const int Sn = S | (1 << v);
                        if (g_pc[Sn] == kFlip7Target) { out[g_sum[Sn] + kFlip7Bonus] += pv; pf7 += pv; }
                        else                            at(uu, newf, Sn) += pv;
                    }
                }
        }
    }
    return pf7;
}

// Probability of arriving at each hand S as an active decision hand under optimal
// play (i.e. the player Hit at every proper prefix and has not busted). reach[S]
// covers both Hit and Stay states; summed over a popcount level it is P(the player
// ever holds that many unique cards). Used to weight the state-resolved analysis.
inline void optimal_reach(const uint8_t* hit, std::vector<double>& reach) {
    constexpr int N = 1 << kNumValues;
    reach.assign(N, 0.0);
    reach[0] = 1.0;
    for (int pc = 0; pc < kFlip7Target; ++pc)
        for (int S = 0; S < N; ++S) {
            if (g_pc[S] != pc) continue;
            const double pr = reach[S];
            if (pr == 0.0 || !hit[S]) continue;           // Stay states are leaves
            const int T = kNumberDeckSize - pc;
            for (unsigned nm = (~(unsigned)S) & 0x1FFFu; nm; nm &= nm - 1) {
                const int v = __builtin_ctz(nm);
                const int Sn = S | (1 << v);
                if (g_pc[Sn] == kFlip7Target) continue;   // Flip 7 is terminal, not a decision hand
                reach[Sn] += pr * (double)numberCount(v) / (double)T;
            }
        }
}

// Mean / P(bust) / P(Flip 7) of a pmf (out[0]=P(bust)).
struct PmfStats { double mean, p_bust, p_flip7; };
inline PmfStats pmf_stats(const double* p, double pf7) {
    double m = 0, tot = 0;
    for (int s = 0; s <= kRoundScoreMax; ++s) { m += s * p[s]; tot += p[s]; }
    return {tot > 0 ? m / tot : 0.0, tot > 0 ? p[0] / tot : 0.0, tot > 0 ? pf7 / tot : 0.0};
}

// Win-probability grid for first-to-`target` when the agent may aim one Flip
// Three every round: none (both play D), self (agent plays the gift pmf De), or
// opp (opponent plays the attacked pmf Dl). Returns the grid W[a*target+b] =
// P(agent wins) at round-start totals (a,b); if polOut != nullptr it receives the
// chosen target per state (0 none / 1 self / 2 opp). Topological by (a+b) with the
// both-bust self-loop folded into each option's closed form (linear in W[a][b]).
inline std::vector<double> win_prob_flip3_target(
        const std::vector<double>& D, const std::vector<double>& De,
        const std::vector<double>& Dl, int target, std::vector<uint8_t>* polOut = nullptr) {
    auto support = [](const std::vector<double>& P) {
        std::vector<int> s;
        for (int i = 0; i < (int)P.size(); ++i) if (P[i] > 0) s.push_back(i);
        return s;
    };
    const auto sD = support(D), sDe = support(De), sDl = support(Dl);
    std::vector<double>  W((size_t)target * target, 0.0);
    std::vector<uint8_t> pol((size_t)target * target, 0);
    auto optval = [&](int a, int b, const std::vector<double>& pa, const std::vector<int>& sa,
                      const std::vector<double>& po, const std::vector<int>& so) {
        double num = 0.0;
        const double self = pa[0] * po[0];
        for (int x : sa) {
            const double pax = pa[x];
            const int A = a + x;
            for (int y : so) {
                if (x == 0 && y == 0) continue;
                const double p = pax * po[y];
                const int B = b + y;
                if (A >= target || B >= target) num += p * (A > B ? 1.0 : (A == B ? 0.5 : 0.0));
                else                            num += p * W[(size_t)A * target + B];
            }
        }
        return num / (1.0 - self);
    };
    for (int sumab = 2 * (target - 1); sumab >= 0; --sumab) {
        const int alo = std::max(0, sumab - (target - 1));
        const int ahi = std::min(target - 1, sumab);
        for (int a = alo; a <= ahi; ++a) {
            const int b = sumab - a;
            const double vn = optval(a, b, D, sD, D, sD);
            const double vs = optval(a, b, De, sDe, D, sD);
            const double vo = optval(a, b, D, sD, Dl, sDl);
            double best = vn; uint8_t bp = 0;
            if (vs > best) { best = vs; bp = 1; }
            if (vo > best) { best = vo; bp = 2; }
            W[(size_t)a * target + b] = best;
            pol[(size_t)a * target + b] = bp;
        }
    }
    if (polOut) *polOut = std::move(pol);
    return W;
}

// --- independent numbers-only MC validator (force-injection) ------------------
// mode: 0 = optimal play (no injection); 1 = forced 3 draws at the START; 2 =
// forced 3 draws at the FIRST natural Stay point. Draws WITHOUT replacement from a
// fresh 79-card number deck (partial Fisher-Yates with O(draws) restore).
struct ForcedMC { double mean, p_bust, p_flip7; uint64_t n; };
inline ForcedMC mc_numbers_forced(int mode, const uint8_t* hit, uint64_t n, uint64_t seed) {
    Xoshiro256pp rng;
    rng.seed(seed);
    uint8_t deck[kNumberDeckSize];
    int idx = 0;
    for (int v = 0; v < kNumValues; ++v)
        for (int k = 0, c = numberCount(v); k < c; ++k) deck[idx++] = (uint8_t)v;

    double sum = 0.0;
    long busts = 0, flip7s = 0;
    int sp[16], sj[16];
    for (uint64_t i = 0; i < n; ++i) {
        uint16_t mask = 0;
        int pos = 0, ns = 0;
        int forced = (mode == 1) ? 3 : 0;
        bool used = (mode != 2);                 // mode 2 injects at first stop; others never
        double score = 0.0;
        for (;;) {
            if (forced == 0) {
                if (!hit[mask]) {
                    if (!used) { forced = 3; used = true; }            // inject Flip Three here
                    else       { score = (double)maskSum(mask); break; }
                }
            }
            const uint64_t j = pos + rng.bounded((uint64_t)(kNumberDeckSize - pos));
            const uint8_t card = deck[pos];
            deck[pos] = deck[j]; deck[j] = card;
            sp[ns] = pos; sj[ns] = (int)j; ++ns;
            const int v = deck[pos];
            ++pos;
            if (forced > 0) --forced;
            const uint16_t bit = (uint16_t)(1u << v);
            if (mask & bit) { score = 0.0; ++busts; break; }
            mask |= bit;
            if (maskPop(mask) == kFlip7Target) { score = (double)(maskSum(mask) + kFlip7Bonus); ++flip7s; break; }
        }
        for (int s = ns - 1; s >= 0; --s) {
            const uint8_t t = deck[sp[s]]; deck[sp[s]] = deck[sj[s]]; deck[sj[s]] = t;
        }
        sum += score;
    }
    return {sum / (double)n, (double)busts / (double)n, (double)flip7s / (double)n, n};
}

}  // namespace flip7
