// Chapter 4: competitive first-to-200 (win probability), numbers-only model.
//
//   A. round-score distribution under expected-score-optimal play
//   B. solitaire-to-target: expected rounds to reach 200
//   C. 2-player win probability with both playing the greedy (E[score]) policy
//   D. best response to a greedy field: each round, re-optimize the within-round
//      Hit/Stay policy to maximize WIN PROBABILITY rather than expected score
//      (push when behind, play safe when ahead).
//
// The within-round solver maximizes E[reward]; for the win game the terminal
// reward is g(round_score) = win value of the resulting total, so we reuse it.
#include "flip7_compete.hpp"
#include "flip7_core.hpp"
#include "flip7_dp.hpp"
#include "flip7_duel.hpp"
#include "flip7_rng.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace flip7;

static constexpr int kT = 200;  // target

// ===================================================================== N players
// Competitive first-to-target for n >= 3: exact win-probability DP for n == 3
// (Monte-Carlo beyond), each result cross-checked by an independent numbers-only
// MC tournament. The 2-player chapter below is unchanged (it is the validated
// headline path); this generalizes it in place via win_prob_greedy_n /
// best_response_grid_n in flip7_compete.hpp. A symmetric field has value 1/n.
static int competitive_nplayer(int n, int target) {
    printf("=== Flip 7 - Chapter 4: competitive first-to-%d, %d players ===\n\n", target, n);
    init_round_tables();
    const std::vector<double> D = round_pmf_numbers();
    const auto sup = pmf_support(D);
    double mean = 0; for (int s = 0; s < (int)D.size(); ++s) mean += s * D[s];
    printf("numbers-only round: mean=%.4f  P(bust)=%.5f\n\n", mean, D[0]);

    // sampler for the numbers-only MC tournaments
    std::vector<double> cdfD(sup.size());
    { double c = 0; for (size_t i = 0; i < sup.size(); ++i) { c += D[sup[i]]; cdfD[i] = c; } }
    Xoshiro256pp rng; rng.seed(0xC0FFEE5ULL);
    auto u01 = [&] { return (rng.next() >> 11) * 0x1.0p-53; };
    auto draw_greedy = [&] {
        const double u = u01();
        size_t i = (size_t)(std::lower_bound(cdfD.begin(), cdfD.end(), u) - cdfD.begin());
        return sup[std::min(i, sup.size() - 1)];
    };
    // play one numbers-only game; players[0..n) draw from per-player samplers.
    auto play_numbers_game = [&](auto&& draw_for) -> int {
        std::vector<long> tot(n, 0);
        for (int round = 0; round < 100000; ++round) {
            for (int i = 0; i < n; ++i) tot[i] += draw_for(i, tot);
            long mx = -1; for (int i = 0; i < n; ++i) mx = std::max(mx, tot[i]);
            if (mx >= target) {
                int winner = -1, ties = 0;
                for (int i = 0; i < n; ++i) if (tot[i] == mx) { ties++; winner = i; }
                if (ties == 1) return winner;
            }
        }
        return -1;
    };

    // ---- greedy field: exact grid (n<=3) + MC ----
    printf("--- greedy field (everyone maximizes E[score]) ---\n");
    if (n <= 3) {
        std::vector<std::vector<double>> Ds(n, D);
        const auto tg = std::chrono::steady_clock::now();
        std::vector<double> Wg = win_prob_greedy_n(Ds, target);
        const double gs = std::chrono::duration<double>(std::chrono::steady_clock::now() - tg).count();
        printf("   exact W(0,..,0) = %.6f  (symmetry => 1/%d = %.6f)   [%.1f s]\n",
               Wg[0], n, 1.0 / n, gs);
    } else {
        printf("   exact grid intractable at n=%d (target^%d states); Monte-Carlo only.\n", n, n);
    }
    {
        const uint64_t G = 2'000'000; long w0 = 0; double nd = 0;
        for (uint64_t g = 0; g < G; ++g) {
            const int r = play_numbers_game([&](int, const std::vector<long>&) { return draw_greedy(); });
            if (r == 0) ++w0; else if (r < 0) nd += 1.0 / n;
        }
        printf("   [MC] player-0 win rate (all greedy) = %.5f  (expect 1/%d = %.5f)\n\n",
               (w0 + nd) / G, n, 1.0 / n);
    }

    // ---- best response: exact (n==3) + on-the-fly MC; MC-only beyond ----
    printf("--- player 0 best-responds to a greedy field ---\n");
    if (n == 3) {
        const auto tb = std::chrono::steady_clock::now();
        std::vector<double> Wbr = best_response_grid_n(D, target, n);
        const double bs = std::chrono::duration<double>(std::chrono::steady_clock::now() - tb).count();
        const size_t T = (size_t)target;
        auto idx = [T](int a, int b, int c) { return ((size_t)a * T + b) * T + c; };
        printf("   exact W_br(0,0,0) = %.6f  => adapting is worth +%.4f vs greedy (1/3)  [%.0f s]\n",
               Wbr[0], Wbr[0] - 1.0 / 3.0, bs);

        // on-the-fly MC cross-check: p0 plays the win-optimal round each round
        // (its policy is recomputed from Wbr at the live standings). Parallel over
        // games, per-thread scratch + per-game-seeded RNG (reproducible).
        const double selfprob = D[0] * D[0];
        const uint64_t G = 200'000;
        unsigned NT = std::thread::hardware_concurrency(); if (!NT) NT = 1;
        std::vector<long> wins(NT, 0);
        std::vector<std::thread> mcth;
        const uint64_t mcchunk = (G + NT - 1) / NT;
        for (unsigned t = 0; t < NT; ++t) {
            const uint64_t g0 = (uint64_t)t * mcchunk, g1 = std::min(G, g0 + mcchunk);
            if (g0 >= g1) break;
            mcth.emplace_back([&, g0, g1, t] {
                std::vector<double> U(1 << kNumValues), gvec(kRoundScoreMax + 1), rd(kRoundScoreMax + 1);
                std::vector<uint8_t> hit(1 << kNumValues);
                Xoshiro256pp r;
                auto u01t = [&] { return (r.next() >> 11) * 0x1.0p-53; };
                auto draw_sup = [&](const std::vector<double>& P) {
                    const double u = u01t(); double cc = 0;
                    for (int s = 0; s < (int)P.size(); ++s) { cc += P[s]; if (u < cc) return s; } return 0;
                };
                auto draw_greedy_t = [&] {
                    const double u = u01t();
                    size_t i = (size_t)(std::lower_bound(cdfD.begin(), cdfD.end(), u) - cdfD.begin());
                    return sup[std::min(i, sup.size() - 1)];
                };
                auto p0_round_pmf = [&](long a, long b, long c) -> const std::vector<double>& {
                    for (int x = 0; x <= kRoundScoreMax; ++x) {
                        const long A = a + x; double gx = 0.0;
                        for (int y : sup) { const long B = b + y; const double dy = D[y];
                            for (int z : sup) {
                                const long C = c + z; const double p = dy * D[z];
                                if (A >= target || B >= target || C >= target) {
                                    const long M = std::max(A, std::max(B, C));
                                    if (A == M) gx += p / ((A == M) + (B == M) + (C == M));
                                } else gx += p * Wbr[idx((int)A, (int)B, (int)C)];
                            }
                        }
                        gvec[x] = gx;
                    }
                    const double base0 = gvec[0];
                    const double w = (a < target && b < target && c < target) ? Wbr[idx((int)a, (int)b, (int)c)] : 0.0;
                    gvec[0] = base0 + selfprob * w;
                    round_solve(gvec.data(), U.data(), hit.data());
                    round_dist(hit.data(), rd.data());
                    return rd;
                };
                long lw = 0;
                for (uint64_t gm = g0; gm < g1; ++gm) {
                    uint64_t sm = 0xBE57ULL + gm; r.seed(splitmix64(sm));
                    long t0 = 0, t1 = 0, t2 = 0;
                    for (int round = 0; round < 100000; ++round) {
                        const std::vector<double>& pp = (t0 < target && t1 < target && t2 < target)
                            ? p0_round_pmf(t0, t1, t2) : D;
                        t0 += draw_sup(pp); t1 += draw_greedy_t(); t2 += draw_greedy_t();
                        const long mx = std::max(t0, std::max(t1, t2));
                        if (mx >= target) {
                            const long tt[3] = {t0, t1, t2}; int win = -1, ties = 0;
                            for (int i = 0; i < 3; ++i) if (tt[i] == mx) { ties++; win = i; }
                            if (ties == 1) { if (win == 0) ++lw; break; }
                        }
                    }
                }
                wins[t] = lw;
            });
        }
        for (auto& x : mcth) x.join();
        long w0 = 0; for (long v : wins) w0 += v;
        printf("   [MC] best-response win rate vs greedy field = %.5f  (DP %.5f)\n\n",
               (double)w0 / G, Wbr[0]);
    } else {
        printf("   exact best-response grid intractable at n=%d; (MC adaptive policy not modeled here).\n\n", n);
    }

    // ---- full real-rules 94-card tournament: adversarial targeting value ----
    printf("--- real 94-card tournament: adversarial targeting vs a field ---\n");
    SolitaireModDP mdp; mdp.optimal();
    const int katt = 3;
    const uint64_t G = 300'000;
    std::vector<int> all_self(n, TP_SELF), all_rnd(n, TP_RANDOM);
    std::vector<int> adv_vs_self = all_self; adv_vs_self[0] = TP_ADVERSARIAL;
    std::vector<int> adv_vs_rnd  = all_rnd;  adv_vs_rnd[0]  = TP_ADVERSARIAL;
    auto rate = [](const DuelStats& s) { return s.p0_score / (double)s.games; };
    DuelStats sR = run_tournament(mdp, all_rnd,      katt, target, G, 0xD1ULL);
    DuelStats sAr = run_tournament(mdp, adv_vs_rnd,  katt, target, G, 0xD2ULL);
    DuelStats sAs = run_tournament(mdp, adv_vs_self, katt, target, G, 0xD3ULL);
    printf("   symmetric random field : %.4f  (sanity ~1/%d=%.4f)\n", rate(sR), n, 1.0 / n);
    printf("   adversarial vs random  : %.4f  (+%.4f vs 1/%d)\n", rate(sAr), rate(sAr) - 1.0 / n, n);
    printf("   adversarial vs self    : %.4f  (+%.4f vs 1/%d)\n", rate(sAs), rate(sAs) - 1.0 / n, n);
    return 0;
}
// The within-round solver (init_round_tables / round_solve / round_dist) lives
// in flip7_compete.hpp so both this program and the profiler can use it.

// A/B/C for a given round-score distribution; prints headline win-prob numbers.
static void competitive_summary(const char* label, const std::vector<double>& D, int kT) {
    double mean = 0;
    for (int s = 0; s < (int)D.size(); ++s) mean += s * D[s];
    const double er = expected_rounds_to_target(D, kT);
    const auto Wg = win_prob_greedy(D, kT);
    auto wg = [&](int a, int b) { return Wg[(size_t)a * kT + b]; };
    printf("[%s] round mean=%.4f  P(bust)=%.5f\n", label, mean, D[0]);
    printf("    B. expected greedy rounds to 200 = %.4f\n", er);
    printf("    C. W(0,0)=%.6f (sym 0.5);  ~1-round lead worth: early(+18)=%.4f mid(118,100)=%.4f late(180,162)=%.4f\n",
           wg(0, 0), wg(std::max(1, (int)mean), 0), wg(118, 100), wg(180, 162));
}

int main(int argc, char** argv) {
    const int n      = (argc > 1) ? atoi(argv[1]) : 2;   // players (default 2)
    const int target = (argc > 2) ? atoi(argv[2]) : kT;  // first-to-target (default 200)
    if (n >= 3) return competitive_nplayer(n, target);
    // n == 2: the original validated headline path (unchanged output).
    printf("=== Flip 7 - Chapter 4: competitive first-to-200 ===\n\n");
    init_round_tables();

    std::vector<double> D = round_pmf_numbers();
    const auto sup = pmf_support(D);

    // ---- A / B / C for both card sets ----
    printf("--- A/B/C: across-rounds win probabilities (both players greedy) ---\n");
    competitive_summary("numbers only ", D, kT);
    std::vector<double> Dall = load_pmf("data/round_pmf_all94.txt");
    if (!Dall.empty()) competitive_summary("ALL 94 cards ", Dall, kT);
    else printf("[ALL 94 cards] data/round_pmf_all94.txt missing -- run 'make all-cards' to generate it.\n");
    printf("\n");
    const auto Wg = win_prob_greedy(D, kT);
    auto wg = [&](int a, int b) { return Wg[(size_t)a * kT + b]; };

    // ---- D. best response to a greedy field (numbers-only model) ----
    printf("--- D. best response vs a greedy field [numbers-only] (re-optimize each round) ---\n");
    // sanity: maximizing E[score] (g = identity) reproduces 18.5652
    {
        std::vector<double> g(kRoundScoreMax + 1), U(1 << kNumValues);
        for (int s = 0; s <= kRoundScoreMax; ++s) g[s] = s;
        printf("   (within-round solver sanity: max E[score] = %.6f)\n", round_solve(g.data(), U.data(), nullptr));
    }

    std::vector<double> Wbr((size_t)kT * kT, 0.0);
    std::vector<float>  pmfA((size_t)kT * kT * (kRoundScoreMax + 1), 0.0f);  // win-optimal A round pmf per state
    std::vector<double> U(1 << kNumValues), Bc(1 << kNumValues), g(kRoundScoreMax + 1);
    std::vector<uint8_t> hit(1 << kNumValues);
    std::vector<double> rd(kRoundScoreMax + 1);
    long solves = 0;  // round_solve calls (closed-form: ~1-2/state vs ~6-20 plain)
    const auto t_br = std::chrono::steady_clock::now();

    for (int sumab = 2 * (kT - 1); sumab >= 0; --sumab) {
        const int alo = std::max(0, sumab - (kT - 1));
        const int ahi = std::min(kT - 1, sumab);
        for (int a = alo; a <= ahi; ++a) {
            const int b = sumab - a;
            for (int x = 0; x <= kRoundScoreMax; ++x) {
                double gx = 0;
                const int A = a + x;
                for (int y : sup) {
                    const int B = b + y;
                    double term;
                    if (A >= kT || B >= kT) term = (A > B ? 1.0 : (A == B ? 0.5 : 0.0));
                    else                    term = Wbr[(size_t)A * kT + B];  // higher state, done (Wbr[a][b]=0 now -> self handled below)
                    gx += D[y] * term;
                }
                g[x] = gx;
            }
            const double base0 = g[0];                 // = sum_{y>0} D[y] outcome(a,b+y)  (self term was 0)
            double w = wg(a, b);                        // warm start
            // Self-loop g[0] = base0 + D0*w, w = U[0]. round_solve is linear in
            // g[0] within a policy region, so jump to the exact fixed point each
            // step: w* = (U0 - B0*D0*w)/(1 - B0*D0). Converges in ~1-2 solves.
            for (int it = 0; it < 12; ++it) {
                g[0] = base0 + D[0] * w;
                double B0;
                const double U0 = round_solve(g.data(), U.data(), nullptr, Bc.data(), &B0);
                ++solves;
                const double denom = 1.0 - B0 * D[0];
                const double w_new = (denom > 1e-12) ? (U0 - B0 * D[0] * w) / denom : U0;
                if (std::fabs(w_new - w) < 1e-12) { w = w_new; break; }
                w = w_new;
            }
            Wbr[(size_t)a * kT + b] = w;
            // store the win-optimal round pmf for the MC cross-check
            g[0] = base0 + D[0] * w;
            round_solve(g.data(), U.data(), hit.data());
            round_dist(hit.data(), rd.data());
            float* dst = &pmfA[((size_t)a * kT + b) * (kRoundScoreMax + 1)];
            for (int s = 0; s <= kRoundScoreMax; ++s) dst[s] = (float)rd[s];
        }
    }
    const double br_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_br).count();
    printf("   W_br(0,0) = %.6f  => adapting is worth +%.4f win prob vs also playing greedy (0.5)\n",
           Wbr[0], Wbr[0] - 0.5);
    printf("   best-response grid solved in %.2f s  (%.2f round_solve calls/state, closed-form self-loop)\n\n",
           br_s, (double)solves / ((double)kT * kT));

    // push/safe behavior: win-optimal round at a mid-game total, behind vs ahead.
    printf("   push-when-behind / safe-when-ahead (your total ~110, vary the gap):\n");
    printf("     your-total opp-total |  P(bust)  E[round]   (win-optimal round)\n");
    auto show = [&](int a, int b) {
        const float* p = &pmfA[((size_t)a * kT + b) * (kRoundScoreMax + 1)];
        double pb = p[0], es = 0; for (int s = 0; s <= kRoundScoreMax; ++s) es += s * p[s];
        printf("       %4d      %4d     |  %.4f   %6.3f\n", a, b, pb, es);
    };
    show(110, 160); show(110, 135); show(110, 110); show(110, 85); show(110, 60);

    // ---- MC: A best-responds (cached pmf), B greedy; check W_br(0,0) ----
    std::vector<double> cdfD(sup.size());
    { double c = 0; for (size_t i = 0; i < sup.size(); ++i) { c += D[sup[i]]; cdfD[i] = c; } }
    Xoshiro256pp rng; rng.seed(0xC0FFEEULL);
    auto u01 = [&]{ return (rng.next() >> 11) * 0x1.0p-53; };
    auto draw_greedy = [&]{ const double u = u01(); size_t i = (size_t)(std::lower_bound(cdfD.begin(), cdfD.end(), u) - cdfD.begin()); return sup[std::min(i, sup.size() - 1)]; };
    auto draw_A = [&](int a, int b){ const float* p = &pmfA[((size_t)a * kT + b) * (kRoundScoreMax + 1)]; double u = u01(), c = 0; for (int s = 0; s <= kRoundScoreMax; ++s) { c += p[s]; if (u < c) return s; } return 0; };
    const uint64_t G = 3'000'000;
    long wins = 0; double half = 0;
    for (uint64_t gm = 0; gm < G; ++gm) {
        int a = 0, b = 0;
        for (;;) { a += draw_A(a, b); b += draw_greedy();
            if (a >= kT || b >= kT) { if (a > b) ++wins; else if (a == b) half += 1; break; } }
    }
    printf("\n   [MC] best-response win rate vs greedy = %.5f  (DP %.5f)\n", (wins + 0.5 * half) / G, Wbr[0]);
    return 0;
}
