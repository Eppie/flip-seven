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
#include "flip7_rng.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace flip7;

static constexpr int kT = 200;  // target
// The within-round solver (init_round_tables / round_solve / round_dist) lives
// in flip7_compete.hpp so both this program and the profiler can use it.

// A/B/C for a given round-score distribution; prints headline win-prob numbers.
static void competitive_summary(const char* label, const std::vector<double>& D, int kT) {
    double tot = 0, mean = 0;
    for (int s = 0; s < (int)D.size(); ++s) { tot += D[s]; mean += s * D[s]; }
    const double er = expected_rounds_to_target(D, kT);
    const auto Wg = win_prob_greedy(D, kT);
    auto wg = [&](int a, int b) { return Wg[(size_t)a * kT + b]; };
    printf("[%s] round mean=%.4f  P(bust)=%.5f\n", label, mean, D[0]);
    printf("    B. expected greedy rounds to 200 = %.4f\n", er);
    printf("    C. W(0,0)=%.6f (sym 0.5);  ~1-round lead worth: early(+18)=%.4f mid(118,100)=%.4f late(180,162)=%.4f\n",
           wg(0, 0), wg(std::max(1, (int)mean), 0), wg(118, 100), wg(180, 162));
}

int main() {
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
    std::vector<double> U(1 << kNumValues), g(kRoundScoreMax + 1);
    std::vector<uint8_t> hit(1 << kNumValues);
    std::vector<double> rd(kRoundScoreMax + 1);

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
            for (int it = 0; it < 25; ++it) {
                g[0] = base0 + D[0] * w;                // self-loop: both bust -> stay at (a,b)
                const double w2 = round_solve(g.data(), U.data(), nullptr);
                if (std::fabs(w2 - w) < 1e-11) { w = w2; break; }
                w = w2;
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
    printf("   W_br(0,0) = %.6f  => adapting is worth +%.4f win prob vs also playing greedy (0.5)\n\n",
           Wbr[0], Wbr[0] - 0.5);

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
