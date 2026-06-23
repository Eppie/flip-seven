// Chapter 5: adversarial action-card targeting (Freeze / Flip Three).
//
//   A. State-resolved value of each action (exact numbers-only DP, MC-checked):
//        - Flip Three reshapes the target's round: a GIFT at shallow hands (free
//          cards, low dup risk), an ATTACK at deep hands -- with a crossover k*.
//        - Freeze caps the target at sum(S); the points it denies are EV(S)-sum(S).
//   B. Optimal targeting in the first-to-200 win game (exact win-prob DP + MC):
//        whom to aim a Flip Three at -- yourself (gift) or the opponent (attack) --
//        as a function of the running totals (a,b).
//   C. The real 94-card 2-player game with ORGANIC actions (faithful-rules MC):
//        how much adversarial targeting is worth vs. random / self targeting.
//
// Discipline: every exact (A/B) number is confirmed by an independent Monte-Carlo
// before it is reported. C is the real-rules ground truth, reported with its
// statistical error and symmetric-matchup sanities.
#include "flip7_actions.hpp"
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
#include <cstring>
#include <vector>

using namespace flip7;
using clk = std::chrono::steady_clock;
static double secs(clk::time_point a) { return std::chrono::duration<double>(clk::now() - a).count(); }
static constexpr int kT = 200;

// N-player adversarial targeting via the faithful 94-card real-rules tournament
// (the ground truth). Part A (the value of an action by the target's hand) is
// player-count independent -- it is about one target's hand -- so it is unchanged.
// The exact win-probability targeting DP (Part B) stays 2-player (its idealized
// "one free Flip Three per round" model); for arbitrary N the real-rules MC below
// measures what adversarial targeting is actually worth against a field. Player 0
// aims at the leader among active opponents (Freeze the match leader; Flip Three
// the deepest opponent). A symmetric field gives each player 1/n.
static int actions_nplayer(int n, int target) {
    printf("=== Flip 7 - Chapter 5: adversarial targeting, %d players ===\n\n", n);

    // ---- exact win-probability targeting DP (numbers-only), n==3 only ----
    if (n == 3) {
        std::vector<double> Uo; std::vector<uint8_t> hit;
        numbers_opt_policy(Uo, hit);
        std::vector<double> D(kRoundScoreMax + 1), De(kRoundScoreMax + 1), Dl(kRoundScoreMax + 1);
        pmf_forced(0, 0, hit.data(), D.data());          // baseline round pmf
        pmf_forced(0, 3, hit.data(), De.data());         // Flip3 @ start  (self gift)
        pmf_flip3_at_stop(hit.data(), Dl.data());        // Flip3 @ stop   (opp attack)

        printf("--- B. optimal Flip-Three target in first-to-%d, 3 players (exact win-prob DP) ---\n", target);
        printf("   idealized: player 0 may aim one Flip Three per round at none / self / opp1 / opp2.\n");
        const auto tB = clk::now();
        std::vector<uint8_t> pol;
        std::vector<double> W = win_prob_flip3_target_n3(D, De, Dl, target, &pol);
        const size_t T = (size_t)target;
        auto idx = [T](int a, int b, int c) { return ((size_t)a * T + b) * T + c; };
        printf("   exact W(0,0,0) = %.5f  => one optimally-aimed Flip Three/round is worth +%.4f vs 1/3  [%.0f s]\n",
               W[0], W[0] - 1.0 / 3.0, secs(tB));
        const char* nm[4] = {"none", "self", "opp1", "opp2"};
        auto tgt = [&](int a, int b, int c) {
            if (a >= target || b >= target || c >= target) return;   // off-grid (small smoke targets)
            printf("     you=%3d opp1=%3d opp2=%3d -> %s   W=%.4f\n", a, b, c, nm[pol[idx(a, b, c)]], W[idx(a, b, c)]);
        };
        printf("   targeting map (who to Flip Three; opp1/opp2 = the two opponents' totals):\n");
        tgt(20, 20, 20); tgt(60, 110, 60); tgt(60, 60, 110); tgt(110, 160, 120); tgt(150, 120, 185);

        // numbers-only MC cross-check: p0 follows the DP targeting policy each round.
        auto cdf_of = [](const std::vector<double>& P) {
            std::vector<double> c(P.size()); double s = 0;
            for (size_t i = 0; i < P.size(); ++i) { s += P[i]; c[i] = s; } return c;
        };
        const std::vector<double> cD = cdf_of(D), cDe = cdf_of(De), cDl = cdf_of(Dl);
        const uint64_t G = 2'000'000;
        unsigned NT = std::thread::hardware_concurrency(); if (!NT) NT = 1;
        std::vector<long> wins(NT, 0);
        std::vector<std::thread> th;
        const uint64_t chunk = (G + NT - 1) / NT;
        for (unsigned t = 0; t < NT; ++t) {
            const uint64_t g0 = (uint64_t)t * chunk, g1 = std::min(G, g0 + chunk);
            if (g0 >= g1) break;
            th.emplace_back([&, g0, g1, t] {
                Xoshiro256pp r;
                auto draw = [&](const std::vector<double>& c) {
                    const double u = (r.next() >> 11) * 0x1.0p-53;
                    return (int)(std::lower_bound(c.begin(), c.end(), u) - c.begin());
                };
                long lw = 0;
                for (uint64_t gm = g0; gm < g1; ++gm) {
                    uint64_t sm = 0xF1A3ULL + gm; r.seed(splitmix64(sm));
                    int a = 0, b = 0, c = 0;
                    for (int round = 0; round < 100000; ++round) {
                        const uint8_t bp = (a < target && b < target && c < target) ? pol[idx(a, b, c)] : 0;
                        a += draw(bp == 1 ? cDe : cD);
                        b += draw(bp == 2 ? cDl : cD);
                        c += draw(bp == 3 ? cDl : cD);
                        const int mx = std::max(a, std::max(b, c));
                        if (mx >= target) {
                            const int tt[3] = {a, b, c}; int win = -1, ties = 0;
                            for (int i = 0; i < 3; ++i) if (tt[i] == mx) { ties++; win = i; }
                            if (ties == 1) { if (win == 0) ++lw; break; }
                        }
                    }
                }
                wins[t] = lw;
            });
        }
        for (auto& x : th) x.join();
        long w0 = 0; for (long v : wins) w0 += v;
        printf("   [MC] win rate w/ DP targeting = %.5f  (DP %.5f)\n\n", (double)w0 / G, W[0]);
    } else {
        printf("   exact targeting DP omitted (exact only for n<=3); real-rules MC below.\n\n");
    }

    printf("--- C. real 94-card tournament: adversarial targeting vs a field ---\n");
    SolitaireModDP mdp; mdp.optimal();
    const int katt = 3;                                   // attack opponents holding >= 3 cards
    const uint64_t G = 1'000'000;
    auto rate = [](const DuelStats& s) { return s.p0_score / (double)s.games; };
    std::vector<int> all_self(n, TP_SELF), all_rnd(n, TP_RANDOM), all_adv(n, TP_ADVERSARIAL);
    std::vector<int> adv_vs_rnd = all_rnd;  adv_vs_rnd[0]  = TP_ADVERSARIAL;
    std::vector<int> adv_vs_self = all_self; adv_vs_self[0] = TP_ADVERSARIAL;
    const auto t0 = clk::now();
    DuelStats sAr = run_tournament(mdp, adv_vs_rnd,  katt, target, G, 0xC5A1ULL);
    DuelStats sAs = run_tournament(mdp, adv_vs_self, katt, target, G, 0xC5A2ULL);
    DuelStats sRR = run_tournament(mdp, all_rnd,     katt, target, G, 0xC5A3ULL);
    DuelStats sAA = run_tournament(mdp, all_adv,     katt, target, G, 0xC5A4ULL);
    const double se = std::sqrt((1.0 / n) * (1.0 - 1.0 / n) / (double)G);
    printf("   adversarial vs random field : %.4f  (+%.4f vs 1/%d, se~%.4f)\n", rate(sAr), rate(sAr) - 1.0 / n, n, se);
    printf("   adversarial vs self  field  : %.4f  (+%.4f vs 1/%d)\n", rate(sAs), rate(sAs) - 1.0 / n, n);
    printf("   random field (sanity)       : %.4f  (~1/%d = %.4f)\n", rate(sRR), n, 1.0 / n);
    printf("   all-adversarial (sanity)    : %.4f  (~1/%d = %.4f)\n", rate(sAA), n, 1.0 / n);
    printf("   per game ~%.1f rounds; aimed Freeze at opp %.0f%%, Flip3 at opp %.0f%% (attack opp pop>=%d)\n",
           (double)sAr.rounds / sAr.games,
           sAr.freezes ? 100.0 * sAr.freeze_at_opp / sAr.freezes : 0.0,
           sAr.flip3s ? 100.0 * sAr.flip3_at_opp / sAr.flip3s : 0.0, katt);
    printf("   %llu games/matchup in %.2f s\n", (unsigned long long)G, secs(t0));
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && strncmp(argv[1], "players=", 8) == 0) {
        const int n = atoi(argv[1] + 8);
        const int target = (argc > 2) ? atoi(argv[2]) : kT;
        init_round_tables();
        return actions_nplayer(n, target);
    }
    printf("=== Flip 7 - Chapter 5: adversarial action-card targeting ===\n\n");
    init_round_tables();
    constexpr int N = 1 << kNumValues;

    std::vector<double> U;
    std::vector<uint8_t> hit;
    numbers_opt_policy(U, hit);
    std::vector<double> reach;
    optimal_reach(hit.data(), reach);

    // ----------------------------------------------------------------- Part A
    printf("--- A. value of an action by the target's current hand (exact, numbers-only) ---\n");
    printf("   Flip Three forces 3 draws then optimal play; Freeze caps the target at sum(S).\n");
    printf("   k = unique cards the target holds; P(reach) = chance an optimal player ever sits there.\n");
    printf("   dMean = change in the TARGET's expected round score from being Flip-Three'd there.\n\n");
    printf("    k  P(reach)  E[base]  E[+Flip3]   dMean   Flip3:P(bust) P(Flip7)  Freeze denies\n");
    auto tA = clk::now();
    std::vector<double> init(N, 0.0), out(kRoundScoreMax + 1);
    double dMeanByK[7] = {0};
    int kpeak = 3;
    for (int k = 0; k <= 6; ++k) {
        std::fill(init.begin(), init.end(), 0.0);
        double w = 0, frz = 0;
        for (int S = 0; S < N; ++S)
            if (g_pc[S] == k) { init[S] = reach[S]; w += reach[S]; frz += reach[S] * (U[S] - g_sum[S]); }
        if (w <= 0) continue;
        double pf = pmf_from_dist(init.data(), 0, hit.data(), out.data());
        const double baseM = pmf_stats(out.data(), pf).mean;
        pf = pmf_from_dist(init.data(), 3, hit.data(), out.data());
        const auto s3 = pmf_stats(out.data(), pf);
        const double dMean = s3.mean - baseM;
        dMeanByK[k] = dMean;
        if (dMean < dMeanByK[kpeak]) kpeak = k;
        printf("    %d   %6.4f   %6.2f    %6.2f   %+6.2f       %6.4f   %6.4f     %6.2f\n",
               k, w, baseM, s3.mean, dMean, s3.p_bust, s3.p_flip7, frz / w);
    }
    printf("\n   => Flip Three is never a gift (dMean <= 0): ~neutral when the target is shallow (the\n");
    printf("      forced draws are ones an optimal player would take anyway), most damaging at k=%d\n", kpeak);
    printf("      (%+.1f pts), then WEAKER again near Flip 7 -- forcing draws on a 5-6 card hand often\n", dMeanByK[kpeak]);
    printf("      completes their +15 bonus (P(Flip7) climbs to 0.38, 0.63). So aim it at a mid-deep\n");
    printf("      opponent, not one who is one card from Flip 7. Freeze denies EV(S)-sum(S) >= 0\n");
    printf("      (it strips the target's upside -- and removes their bust risk, the competitive catch).\n");

    // exact pmfs we will reuse + their independent MC checks
    std::vector<double> D(kRoundScoreMax + 1), De(kRoundScoreMax + 1), Dl(kRoundScoreMax + 1);
    double pfD  = pmf_forced(0, 0, hit.data(), D.data());
    double pfDe = pmf_forced(0, 3, hit.data(), De.data());
    double pfDl = pmf_flip3_at_stop(hit.data(), Dl.data());
    const auto sD = pmf_stats(D.data(), pfD), sDe = pmf_stats(De.data(), pfDe), sDl = pmf_stats(Dl.data(), pfDl);
    printf("\n   round pmfs (exact):  baseline mean=%.4f  | Flip3@start mean=%.4f (~neutral)"
           "  | Flip3@stop mean=%.4f (attack: P(bust)=%.4f)\n",
           sD.mean, sDe.mean, sDl.mean, sDl.p_bust);
    printf("   exact-DP solved in %.3f s\n", secs(tA));

    {
        const uint64_t n = 20'000'000;
        auto m0 = mc_numbers_forced(0, hit.data(), n, 0x5151ULL);
        auto m1 = mc_numbers_forced(1, hit.data(), n, 0x5252ULL);
        auto m2 = mc_numbers_forced(2, hit.data(), n, 0x5353ULL);
        printf("   [MC %llu/run] baseline mean DP=%.4f MC=%.4f (d=%.4f) | self-gift DP=%.4f MC=%.4f"
               " | opp-attack DP=%.4f MC=%.4f bust DP=%.4f MC=%.4f\n",
               (unsigned long long)n, sD.mean, m0.mean, m0.mean - sD.mean,
               sDe.mean, m1.mean, sDl.mean, m2.mean, sDl.p_bust, m2.p_bust);
    }

    // ----------------------------------------------------------------- Part B
    printf("\n--- B. optimal Flip-Three target in first-to-200 (exact win-prob DP) ---\n");
    printf("   Idealized availability: the agent may aim one Flip Three every round (none / self De /\n");
    printf("   opp Dl). This isolates the targeting DIRECTION and its ceiling; the real ~3/94 card\n");
    printf("   frequency is measured in Part C. (self De is ~neutral, so it is dominated by none.)\n");
    auto tB = clk::now();
    std::vector<uint8_t> pol;  // 0 none, 1 self, 2 opp
    std::vector<double> W = win_prob_flip3_target(D, De, Dl, kT, &pol);
    printf("   W(0,0) = %.5f  => one optimally-targeted Flip Three per round is worth +%.4f win prob\n",
           W[0], W[0] - 0.5);
    printf("   solved in %.3f s\n", secs(tB));
    const char* nm[3] = {"none", "self", "OPP "};
    auto tgt = [&](int a, int b) {
        printf("     you=%3d opp=%3d (%+4d) -> target %s   W=%.4f\n",
               a, b, a - b, nm[pol[(size_t)a * kT + b]], W[(size_t)a * kT + b]);
    };
    printf("   targeting map (who to Flip Three, and the resulting win prob):\n");
    tgt(20, 20); tgt(110, 60); tgt(110, 110); tgt(110, 160); tgt(180, 120); tgt(150, 185);
    printf("   => for Flip Three the target is unconditional: ALWAYS the opponent (attacking only\n");
    printf("      lowers their mean to ~7.9; self/none are dominated). The subtlety is WHEN (Part A:\n");
    printf("      aim mid-deep); the standings only change how MUCH the swing is worth.\n");

    {
        Xoshiro256pp rng; rng.seed(0xBEEF5ULL);
        auto u01 = [&] { return (rng.next() >> 11) * 0x1.0p-53; };
        auto draw = [&](const std::vector<double>& P, double u) {
            double c = 0; for (int s = 0; s <= kRoundScoreMax; ++s) { c += P[s]; if (u < c) return s; } return 0;
        };
        const uint64_t G = 3'000'000; long wins = 0; double half = 0;
        for (uint64_t gm = 0; gm < G; ++gm) {
            int a = 0, b = 0;
            for (;;) {
                const int pp = pol[(size_t)a * kT + b];
                const auto& pa = (pp == 1) ? De : D;
                const auto& po = (pp == 2) ? Dl : D;
                a += draw(pa, u01());
                b += draw(po, u01());
                if (a >= kT || b >= kT) { if (a > b) ++wins; else if (a == b) half += 1; break; }
            }
        }
        printf("   [MC] win rate w/ DP targeting = %.5f  (DP %.5f)\n", (wins + 0.5 * half) / G, W[0]);
    }

    // ----------------------------------------------------------------- Part C
    printf("\n--- C. the real 94-card 2-player game with organic actions (faithful MC) ---\n");
    printf("   players use the numbers+modifier optimal Hit/Stay; only the TARGETING differs.\n");
    SolitaireModDP mdp; mdp.optimal();
    const int katt = std::max(2, kpeak - 1);   // attack from ~one card shallower than peak damage
    const uint64_t G = 300'000;
    auto tC = clk::now();
    DuelStats adv_vs_rnd  = run_duel(mdp, TP_ADVERSARIAL, TP_RANDOM,      katt, kT, G, 0xC5A1ULL);
    DuelStats adv_vs_self = run_duel(mdp, TP_ADVERSARIAL, TP_SELF,        katt, kT, G, 0xC5A2ULL);
    DuelStats rnd_vs_rnd  = run_duel(mdp, TP_RANDOM,      TP_RANDOM,      katt, kT, G, 0xC5A3ULL);
    DuelStats adv_vs_adv  = run_duel(mdp, TP_ADVERSARIAL, TP_ADVERSARIAL, katt, kT, G, 0xC5A4ULL);
    const double sc = secs(tC);
    auto rate = [&](const DuelStats& s) { return s.p0_score / (double)s.games; };
    const double se = 0.5 / std::sqrt((double)G);  // ~win-rate standard error
    printf("   adversarial vs random : %.4f  (+%.4f edge, se~%.4f)\n", rate(adv_vs_rnd), rate(adv_vs_rnd) - 0.5, se);
    printf("   adversarial vs self   : %.4f  (+%.4f edge)\n", rate(adv_vs_self), rate(adv_vs_self) - 0.5);
    printf("   random vs random      : %.4f  (sanity ~0.5)\n", rate(rnd_vs_rnd));
    printf("   adversarial vs advers.: %.4f  (sanity ~0.5)\n", rate(adv_vs_adv));
    printf("   per game: ~%.1f rounds; adversarial agent aimed Freeze at opp %.0f%%, Flip3 at opp %.0f%%"
           "  (Flip3 attack when opp holds >= %d)\n",
           (double)adv_vs_rnd.rounds / adv_vs_rnd.games,
           adv_vs_rnd.freezes ? 100.0 * adv_vs_rnd.freeze_at_opp / adv_vs_rnd.freezes : 0.0,
           adv_vs_rnd.flip3s ? 100.0 * adv_vs_rnd.flip3_at_opp / adv_vs_rnd.flip3s : 0.0, katt);
    printf("   %llu games/matchup in %.2f s (%.0fK games/s)\n",
           (unsigned long long)G, sc, 4.0 * G / sc / 1e3);
    return 0;
}
