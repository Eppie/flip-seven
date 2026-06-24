// flip7_oracle.hpp -- the decision oracle behind the `decide` CLI.
//
// Given a live situation (number of players, your hand, your match total, the
// opponents' totals/hands, the cards you have already seen, and whether you hold a
// Freeze / Flip Three), recommend the best action. A single exact table over the
// full game state is impossible, but the optimal decision FACTORS: your Hit/Stay
// choice needs only (your hand, your total, opponents' totals, n); only targeting
// needs opponents' hands. The oracle composes:
//
//   1. a WIN-VALUE backbone winval(total) = P(you win the match | you bank to this
//      match total now) -- read from the exact best-response win grid for n<=3
//      (best_response_grid_n), Monte-Carlo for n>=4;
//   2. a COUNT-AWARE within-round solve from your current hand using the ACTUAL
//      remaining deck (seen cards + all visible hands removed) -- giving the
//      win-optimal AND expected-score-optimal Hit/Stay and the exact count-aware
//      P(bust) of hitting now;
//   3. a TARGETING evaluation (Freeze / Flip Three) scored through the win grid.
//
// Honest scope: this is a one-step best response against a greedy field, not the
// exact Nash of the full multiplayer game. The count-aware layer corrects the
// CURRENT round's draw odds exactly; the multi-round continuation still assumes a
// fresh deck (the project's measured-small shoe effect). Card-counting is
// numbers-only (which number cards are gone -> bust risk); held modifiers ride the
// score, opponents are the numbers-only field.
#pragma once
#include "flip7_actions.hpp"
#include "flip7_compete.hpp"
#include "flip7_core.hpp"
#include "flip7_dp.hpp"
#include "flip7_duel.hpp"
#include "flip7_rng.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <unordered_map>
#include <vector>

namespace flip7 {

// Largest possible single-round banked score (numbers x2 + all additive mods + 15).
inline constexpr int kMaxBank = 63 * 2 + (2 + 4 + 6 + 8 + 10) + kFlip7Bonus;  // 171

struct OracleDecision {
    bool   hit_winopt;        // win-probability-optimal Hit (else Stay)
    bool   hit_evopt;         // expected-score-optimal Hit (else Stay)
    double p_bust;            // count-aware P(bust) if you Hit now
    double w_hit, w_stay;     // win probability of Hit vs Stay (win-optimal backbone)
    double ev_hit, ev_stay;   // expected round score of Hit vs Stay (count-aware)
    // targeting (only filled if you hold an action card)
    bool   has_action = false;
    int    target = -1;       // -2 none, -1 self, >=0 opponent index
    double w_action = 0.0;    // your win prob after the recommended action
};

struct Oracle {
    int n, target;
    std::vector<double> D;          // numbers-only field round-score pmf
    std::vector<int>    supD;
    std::vector<double> W;          // best-response grid over standings (n<=3); empty n>=4
    bool exact;                     // true if W is an exact grid (n<=3)

    Oracle(int players, int tgt) : n(players), target(tgt) {
        init_round_tables();
        D = round_pmf_numbers();
        supD = pmf_support(D);
        exact = (n <= 3);
        if (exact) W = load_or_build_grid();
    }

    // ---- win grid (n<=3): build best_response_grid_n once, cache to disk ----
    std::vector<double> load_or_build_grid() {
        char path[256];
        snprintf(path, sizeof path, "data/winbr_n%d_t%d.bin", n, target);
        size_t cells = 1; for (int i = 0; i < n; ++i) cells *= (size_t)target;
        if (FILE* f = std::fopen(path, "rb")) {
            std::vector<double> g(cells);
            const size_t got = std::fread(g.data(), sizeof(double), cells, f);
            std::fclose(f);
            if (got == cells) return g;
        }
        std::vector<double> g = best_response_grid_n(D, target, n);
        if (FILE* f = std::fopen(path, "wb")) {
            std::fwrite(g.data(), sizeof(double), g.size(), f);
            std::fclose(f);
        }
        return g;
    }

    // ---- terminal outcome for player 0 given final totals (idx 0 = you) ----
    static double outcome0(const std::vector<long>& tot) {
        long m = -1; for (long t : tot) m = std::max(m, t);
        int k = 0; for (long t : tot) k += (t == m);
        return (tot[0] == m) ? 1.0 / k : 0.0;
    }

    size_t gidx(const std::vector<int>& t) const {
        size_t i = 0; for (int v : t) i = i * (size_t)target + (size_t)v; return i;
    }

    // Value at start-of-round standings `t` (idx0 = you), exact grid or terminal.
    double grid_value(std::vector<int> t) const {
        bool term = false; for (int v : t) term |= (v >= target);
        if (term) { std::vector<long> l(t.begin(), t.end()); return outcome0(l); }
        return W[gidx(t)];
    }

    // ---- winval(mt): P(you win) if you bank to match total mt at this round's end,
    //      the still-active opponents each completing one field round from their
    //      totals. opp = current opponent totals; act[j] = opponent j still active. ----
    std::vector<double> build_winval(const std::vector<int>& opp, const std::vector<char>& act) {
        std::vector<double> wv(target + kMaxBank + 1, 0.0);
        if (exact) {
            for (int mt = 0; mt < (int)wv.size(); ++mt) {
                const int myt = std::min(mt, target);  // grid axis is < target; >= handled as terminal
                double acc = 0.0;
                // enumerate opponents' field draws (active ones), product pmf
                std::function<void(int, double, std::vector<int>&)> rec =
                    [&](int j, double p, std::vector<int>& cur) {
                        if (j == n - 1) {
                            std::vector<int> t(n); t[0] = myt;
                            for (int o = 0; o < n - 1; ++o) t[o + 1] = std::min(cur[o], target);
                            acc += p * grid_value(t);
                            return;
                        }
                        if (!act[j]) { cur[j] = opp[j]; rec(j + 1, p, cur); return; }
                        for (int y : supD) { cur[j] = opp[j] + y; rec(j + 1, p * D[y], cur); }
                    };
                std::vector<int> cur(n - 1, 0);
                rec(0, 1.0, cur);
                wv[mt] = acc;
            }
        } else {
            wv = build_winval_mc(opp, act);
        }
        return wv;
    }

    // n>=4: Monte-Carlo winval on a coarse grid of banked totals, interpolated.
    std::vector<double> build_winval_mc(const std::vector<int>& opp, const std::vector<char>& act) {
        std::vector<double> wv(target + kMaxBank + 1, 0.0);
        std::vector<double> cdf(supD.size());
        { double c = 0; for (size_t i = 0; i < supD.size(); ++i) { c += D[supD[i]]; cdf[i] = c; } }
        const int step = 4;
        const uint64_t G = 4000;
        std::vector<int> knots;
        for (int mt = 0; mt < (int)wv.size(); mt += step) knots.push_back(mt);
        if (knots.back() != (int)wv.size() - 1) knots.push_back((int)wv.size() - 1);
        std::vector<double> kv(knots.size());
        Xoshiro256pp rng; rng.seed(0xC0DEC0DEULL);
        auto draw = [&] {
            const double u = (rng.next() >> 11) * 0x1.0p-53;
            return supD[std::min((size_t)(std::lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin()), supD.size() - 1)];
        };
        for (size_t ki = 0; ki < knots.size(); ++ki) {
            const int mt = knots[ki];
            long wins = 0;
            for (uint64_t g = 0; g < G; ++g) {
                std::vector<long> t(n);
                t[0] = mt;
                for (int o = 0; o < n - 1; ++o) t[o + 1] = act[o] ? opp[o] + draw() : opp[o];
                // play to target, all greedy
                for (int round = 0; round < 100000; ++round) {
                    long mx = -1; for (long v : t) mx = std::max(mx, v);
                    if (mx >= target) break;
                    for (int i = 0; i < n; ++i) t[i] += draw();
                }
                if (outcome0(t) >= 1.0) ++wins;  // count only clean wins (ties rare)
                else if (outcome0(t) > 0) { /* tie share */ }
            }
            kv[ki] = (double)wins / (double)G;
        }
        for (size_t ki = 0; ki + 1 < knots.size(); ++ki) {
            const int a = knots[ki], b = knots[ki + 1];
            for (int mt = a; mt <= b; ++mt) {
                const double f = (b > a) ? (double)(mt - a) / (b - a) : 0.0;
                wv[mt] = kv[ki] * (1 - f) + kv[ki + 1] * f;
            }
        }
        return wv;
    }

    // ---- count-aware within-round solve from the current hand ----
    // r0[v] = copies of value v left in the unseen draw pile; S0 = your current
    // number-hand; mods/sc as held. rew(total) is the leaf reward (winval for the
    // win-optimal pass, or score for the EV-optimal pass). Returns {value, hit?}.
    struct Solve { double value; bool hit; double p_bust; double v_hit, v_stay; };
    Solve count_aware(uint16_t S0, uint16_t mods, bool sc, long my_total,
                      const int r0[kNumValues], const std::function<double(long)>& rew) {
        long T0 = 0; for (int v = 0; v < kNumValues; ++v) T0 += r0[v];
        std::unordered_map<int, double> memo;  // key (S<<1|sc)
        std::function<double(uint16_t, bool)> val = [&](uint16_t S, bool sch) -> double {
            const int key = (S << 1) | (sch ? 1 : 0);
            auto it = memo.find(key); if (it != memo.end()) return it->second;
            const int pc = maskPop(S);
            const double stay = rew(my_total + fullScore(S, mods));
            double v;
            if (pc == kFlip7Target) { v = stay; memo[key] = v; return v; }
            long Tsz = T0 - (long)maskPop((uint16_t)(S & ~S0));   // cards already drawn in lookahead
            if (Tsz <= 0) { memo[key] = stay; return stay; }
            // bust mass = remaining copies of values already held
            long bust = 0;
            for (uint16_t m = S; m; m &= (uint16_t)(m - 1)) {
                const int vv = __builtin_ctz(m);
                bust += r0[vv] - ((S0 >> vv & 1) ? 0 : 1);       // -1 if drawn during lookahead
            }
            double hit = 0.0;
            if (bust > 0) {
                const double pb = (double)bust / (double)Tsz;
                hit += pb * (sch ? val(S, false) : rew(my_total + 0));  // SC saves the first bust
            }
            for (int vv = 0; vv < kNumValues; ++vv) {
                if (S >> vv & 1) continue;
                const long rem = r0[vv];                          // vv not in S => not yet drawn
                if (rem <= 0) continue;
                const double p = (double)rem / (double)Tsz;
                const uint16_t Sn = (uint16_t)(S | (1u << vv));
                if (maskPop(Sn) == kFlip7Target) hit += p * rew(my_total + fullScore(Sn, mods));
                else                             hit += p * val(Sn, sch);
            }
            v = std::max(stay, hit);
            memo[key] = v;
            return v;
        };
        // top-level: compute hit/stay/p_bust at S0 explicitly
        const double stay = rew(my_total + fullScore(S0, mods));
        long bust = 0; for (uint16_t m = S0; m; m &= (uint16_t)(m - 1)) bust += r0[__builtin_ctz(m)];
        const double pbust = (T0 > 0) ? (double)bust / (double)T0 : 0.0;
        double hit = 0.0;
        if (maskPop(S0) < kFlip7Target && T0 > 0) {
            if (bust > 0) hit += (double)bust / T0 * (sc ? val(S0, false) : rew(my_total + 0));
            for (int vv = 0; vv < kNumValues; ++vv) {
                if (S0 >> vv & 1) continue;
                if (r0[vv] <= 0) continue;
                const double p = (double)r0[vv] / T0;
                const uint16_t Sn = (uint16_t)(S0 | (1u << vv));
                if (maskPop(Sn) == kFlip7Target) hit += p * rew(my_total + fullScore(Sn, mods));
                else                             hit += p * val(Sn, sc);
            }
        } else {
            hit = stay;  // already at Flip 7 or empty deck -> no real hit
        }
        return {std::max(stay, hit), hit > stay, pbust, hit, stay};
    }

    // ---- the public decision ----
    // S0/mods/sc = your hand; my_total = your match total; opp = opponents' totals;
    // oppHands[j] = opponent j's number-hand (0 if unknown); oppActive[j];
    // r0 = remaining number-deck counts. action: 0 none, 1 Freeze, 2 Flip Three.
    OracleDecision decide(uint16_t S0, uint16_t mods, bool sc, long my_total,
                          const std::vector<int>& opp, const std::vector<char>& oppActive,
                          const std::vector<uint16_t>& oppHands, const int r0[kNumValues],
                          int action) {
        OracleDecision d{};
        const std::vector<double> wv = build_winval(opp, oppActive);
        auto winval = [&](long total) { return wv[std::max(0L, std::min(total, (long)wv.size() - 1))]; };
        auto score  = [&](long total) { return (double)(total - my_total); };

        Solve win = count_aware(S0, mods, sc, my_total, r0, winval);
        Solve ev  = count_aware(S0, mods, sc, my_total, r0, score);
        d.hit_winopt = win.hit; d.w_hit = win.v_hit; d.w_stay = win.v_stay; d.p_bust = win.p_bust;
        d.hit_evopt = ev.hit;  d.ev_hit = ev.v_hit;  d.ev_stay = ev.v_stay;

        if (action != 0) d = best_target(d, S0, mods, my_total, opp, oppActive, oppHands, action);
        return d;
    }

    // ---- targeting: pick none / self / which opponent maximizes your win prob ----
    OracleDecision best_target(OracleDecision d, uint16_t /*S0*/, uint16_t /*mods*/, long my_total,
                               const std::vector<int>& opp, const std::vector<char>& oppActive,
                               const std::vector<uint16_t>& oppHands, int action) {
        d.has_action = true;
        // baseline: aim at no one -> your win prob is just winval at your current stay.
        // We compare the effect of the action on the FIELD'S round completion through
        // the grid: build winval under each option and read your value at staying now.
        std::vector<double> Uo; std::vector<uint8_t> hit;
        numbers_opt_policy(Uo, hit);

        auto eval_option = [&](int tgt) -> double {
            // modified opponents' pmfs: a frozen opp banks sum(hand) deterministically;
            // a flip-three'd opp draws the forced-3-from-hand pmf; others field D.
            // We approximate "your win value" as winval(my_total) under the modified
            // field (you stay on your current total this round).
            std::vector<std::vector<double>> oppP(n - 1, D);
            for (int j = 0; j < n - 1; ++j) {
                if (!oppActive[j]) continue;
                if (tgt == j) {
                    if (action == 1) {                          // Freeze: cap at sum(hand)
                        std::vector<double> p(kRoundScoreMax + 1, 0.0);
                        const int s = std::min((int)maskSum(oppHands[j]), kRoundScoreMax);
                        p[s] = 1.0; oppP[j] = p;
                    } else {                                    // Flip Three: forced 3 from hand
                        std::vector<double> init(1 << kNumValues, 0.0), out(kRoundScoreMax + 1, 0.0);
                        init[oppHands[j]] = 1.0;
                        pmf_from_dist(init.data(), 3, hit.data(), out.data());
                        oppP[j] = out;
                    }
                }
            }
            // expected value of you staying at my_total now, opponents draw oppP.
            std::function<double(int, double, std::vector<int>&)> rec =
                [&](int j, double p, std::vector<int>& cur) -> double {
                    if (j == n - 1) {
                        std::vector<int> t(n); t[0] = std::min((int)my_total, target);
                        for (int o = 0; o < n - 1; ++o) t[o + 1] = std::min(cur[o], target);
                        return p * grid_value(t);
                    }
                    if (!oppActive[j]) { cur[j] = opp[j]; return rec(j + 1, p, cur); }
                    double acc = 0.0;
                    for (int s = 0; s <= kRoundScoreMax; ++s) {
                        if (oppP[j][s] <= 0) continue;
                        cur[j] = opp[j] + s; acc += rec(j + 1, p * oppP[j][s], cur);
                    }
                    return acc;
                };
            std::vector<int> cur(n - 1, 0);
            return exact ? rec(0, 1.0, cur) : winvalapprox(my_total, opp, oppActive, oppP);
        };

        double best = eval_option(-1);  // none
        int bestT = -2;
        for (int j = 0; j < n - 1; ++j) {
            if (!oppActive[j] || oppHands[j] == 0) continue;
            const double v = eval_option(j);
            if (v > best) { best = v; bestT = j; }
        }
        d.target = bestT; d.w_action = best;
        return d;
    }

    // crude n>=4 fallback for targeting value (one-round MC with modified field).
    double winvalapprox(long my_total, const std::vector<int>& opp, const std::vector<char>& act,
                        const std::vector<std::vector<double>>& oppP) {
        std::vector<double> cdfD(supD.size());
        { double c = 0; for (size_t i = 0; i < supD.size(); ++i) { c += D[supD[i]]; cdfD[i] = c; } }
        Xoshiro256pp rng; rng.seed(0xA11CE5ULL);
        auto draw = [&](const std::vector<double>& P) {
            const double u = (rng.next() >> 11) * 0x1.0p-53; double c = 0;
            for (int s = 0; s < (int)P.size(); ++s) { c += P[s]; if (u < c) return s; } return 0;
        };
        const uint64_t G = 20000; long wins = 0;
        for (uint64_t g = 0; g < G; ++g) {
            std::vector<long> t(n); t[0] = my_total;
            for (int o = 0; o < n - 1; ++o) t[o + 1] = act[o] ? opp[o] + draw(oppP[o]) : opp[o];
            for (int round = 0; round < 100000; ++round) {
                long mx = -1; for (long v : t) mx = std::max(mx, v);
                if (mx >= target) break;
                for (int i = 0; i < n; ++i) t[i] += draw(D);
            }
            if (outcome0(t) >= 1.0) ++wins;
        }
        return (double)wins / G;
    }
};

}  // namespace flip7
