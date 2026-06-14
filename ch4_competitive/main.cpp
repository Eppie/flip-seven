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

// ---- within-round solver with an arbitrary terminal reward g[round_score] ----
static int    g_sum[1 << kNumValues], g_pc[1 << kNumValues];
static double g_bustnum[1 << kNumValues];  // sum over held v of (count(v)-1)
static std::vector<int> g_order;           // states by popcount descending
static void init_round_tables() {
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
// max E[g(round score)]; fills hit[] policy if non-null. U is a scratch buffer.
static double round_solve(const double* g, double* U, uint8_t* hit) {
    const double g0 = g[0];
    for (int S : g_order) {
        const int pc = g_pc[S];
        if (pc == kFlip7Target) { U[S] = g[g_sum[S] + kFlip7Bonus]; continue; }
        const double stay = g[g_sum[S]];
        double acc = g_bustnum[S] * g0;
        for (unsigned nm = (~(unsigned)S) & 0x1FFFu; nm; nm &= nm - 1) {  // iterate undrawn values only
            const int v = __builtin_ctz(nm);
            acc += (double)numberCount(v) * U[S | (1u << v)];
        }
        const double hv = acc / (double)(kNumberDeckSize - pc);
        if (hv > stay) { U[S] = hv;  if (hit) hit[S] = 1; }
        else           { U[S] = stay; if (hit) hit[S] = 0; }
    }
    return U[0];
}
// Round-score pmf realized by a given within-round policy hit[].
static void round_dist(const uint8_t* hit, double* outD) {
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
            double bust = 0;
            for (int v = 0; v < kNumValues; ++v) if (S & (1u << v)) bust += numberCount(v) - 1;
            outD[0] += pr * bust / (double)T;
            for (int v = 0; v < kNumValues; ++v) {
                const uint16_t bit = (uint16_t)(1u << v);
                if (S & bit) continue;
                const double pv = pr * (double)numberCount(v) / (double)T;
                const uint16_t Sn = (uint16_t)(S | bit);
                if (g_pc[Sn] == kFlip7Target) outD[g_sum[Sn] + kFlip7Bonus] += pv;
                else                          reach[Sn] += pv;
            }
        }
}

int main() {
    printf("=== Flip 7 - Chapter 4: competitive first-to-200 (numbers-only model) ===\n\n");
    init_round_tables();

    double D[kRoundScoreMax + 1];
    round_pmf(D);
    const auto sup = pmf_support(D);

    // ---- A / B / C ----
    double tot = 0, mean = 0;
    for (int s = 0; s <= kRoundScoreMax; ++s) { tot += D[s]; mean += s * D[s]; }
    printf("A. round-score pmf: sum=%.10f mean=%.6f P(bust)=%.5f\n", tot, mean, D[0]);

    const double er = expected_rounds_to_target(D, kT);
    printf("B. expected greedy rounds to reach 200 = %.4f\n", er);

    const auto Wg = win_prob_greedy(D, kT);
    auto wg = [&](int a, int b) { return Wg[(size_t)a * kT + b]; };
    printf("C. 2-player win prob, both greedy: W(0,0)=%.6f (sym check 0.5)\n", wg(0, 0));
    printf("   value of an ~18-pt lead: early (18,0)=%.4f, mid (118,100)=%.4f, late (180,162)=%.4f\n\n",
           wg(18, 0), wg(118, 100), wg(180, 162));

    // ---- D. best response to a greedy field ----
    printf("D. best response vs a greedy field (re-optimize each round for win prob)\n");
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
