// test_ch4 — assert the across-rounds (first-to-200) results (numbers-only model).
//
// Covers the fast A/B/C layers exactly. The best-response grid (D) is heavy
// (~24 s) and is cross-checked by ch4_competitive's own Monte-Carlo; its headline
// W_br(0,0) = 0.5593 is recorded in the README.
#include "flip7_compete.hpp"
#include "flip7_dp.hpp"
#include "flip7_duel.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace flip7;

static int failures = 0;
static void check(bool ok, const char* what) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}
static bool close(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

int main() {
    printf("Chapter 4 tests\n");
    std::vector<double> D = round_pmf_numbers();
    double tot = 0, mean = 0;
    for (int s = 0; s < (int)D.size(); ++s) { tot += D[s]; mean += s * D[s]; }
    check(close(tot, 1.0, 1e-9), "round pmf sums to 1");
    check(close(mean, 18.5652176, 1e-4), "round pmf mean == 18.5652");

    const double er = expected_rounds_to_target(D, 200);
    printf("  (expected rounds to 200 = %.4f)\n", er);
    check(close(er, 11.5134, 1e-3), "expected rounds to 200 == 11.5134");

    const auto W = win_prob_greedy(D, 200);
    auto wg = [&](int a, int b) { return W[(size_t)a * 200 + b]; };
    check(close(wg(0, 0), 0.5, 1e-9), "W(0,0) == 0.5 (symmetry)");
    check(wg(18, 0) > 0.5 && wg(36, 0) > wg(18, 0), "a lead raises win probability monotonically");
    check(wg(180, 162) > wg(118, 100), "the same 18-pt lead is worth more later in the game");

    // ---- N-player generalization cross-checks ----
    init_round_tables();
    // 1. the generic N-D grid reproduces the frozen 2-player path (full target=200)
    {
        const auto Wn = win_prob_greedy_n({D, D}, 200);
        double maxd = 0; for (size_t i = 0; i < W.size(); ++i) maxd = std::max(maxd, std::fabs(W[i] - Wn[i]));
        check(maxd < 1e-9, "win_prob_greedy_n(N=2) matches the 2-player path (<1e-9)");
    }
    // 2. exact 3-player symmetric value is 1/3; opponents are exchangeable
    {
        constexpr int T = 60;
        const auto W3 = win_prob_greedy_n({D, D, D}, T);
        check(close(W3[0], 1.0 / 3.0, 1e-9), "3-player symmetric W(0,0,0) == 1/3 (exact)");
        auto i3 = [](int a, int b, int c) { return ((size_t)a * T + b) * T + c; };
        double exd = 0;
        for (int a = 0; a < 6; ++a) for (int b = 0; b < 6; ++b) for (int c = 0; c < 6; ++c)
            exd = std::max(exd, std::fabs(W3[i3(a, b, c)] - W3[i3(a, c, b)]));
        check(exd < 1e-12, "3-player grid is opponent-exchangeable W(a,b,c)==W(a,c,b)");
    }
    // 3. best response reproduces the 2-player headline and beats the field at N=3
    {
        const auto Wbr2 = best_response_grid_n(D, 200, 2);
        check(close(Wbr2[0], 0.559345, 1e-4), "2-player best-response W_br(0,0) == 0.5593");
        const int T = 40;
        const auto Wbr3 = best_response_grid_n(D, T, 3);
        const auto Wg3 = win_prob_greedy_n({D, D, D}, T);
        check(Wbr3[0] > Wg3[0] + 1e-6, "3-player best-response beats the greedy 1/3 baseline");
    }
    // 4. symmetric real-rules MC tournament gives each player 1/n
    {
        SolitaireModDP mdp; mdp.optimal();
        const uint64_t G = 300'000;
        for (int n = 3; n <= 4; ++n) {
            DuelStats s = run_tournament(mdp, std::vector<int>(n, TP_RANDOM), 3, 200, G, 0x7E57ULL + n);
            const double rate = s.p0_score / (double)s.games;
            const double se = std::sqrt((1.0 / n) * (1.0 - 1.0 / n) / (double)G);
            char msg[96]; snprintf(msg, sizeof msg, "symmetric %d-player MC win rate == 1/%d (within 5 se)", n, n);
            check(std::fabs(rate - 1.0 / n) < 5 * se, msg);
        }
    }

    printf("%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
