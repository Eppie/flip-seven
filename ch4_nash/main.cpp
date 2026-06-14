// Chapter 4 (cont.): symmetric Nash equilibrium of first-to-200 via self-play.
//
// The game is symmetric zero-sum, so its value is 0.5 by symmetry; the object of
// interest is the equilibrium POLICY -- the fixed point where each player's
// within-round play is a best response to the other's. We find it by iterated
// best response with damping (a fictitious-play-style self-play): keep a per-
// state belief about the opponent's round-score distribution, best-respond to
// it, and move the belief toward the best response. At the fixed point the
// best-response value W(0,0) -> 0.5 and the policy stops changing.
//
// Numbers-only model; rounds independent (action-card targeting is Ch. 5).
//   usage: ch4_nash [iterations] [damping]
#include "flip7_compete.hpp"
#include "flip7_core.hpp"
#include "flip7_rng.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace flip7;

static constexpr int kT = 200;
static constexpr int kSn = kRoundScoreMax + 1;  // per-state pmf stride

int main(int argc, char** argv) {
    int iters = (argc > 1) ? atoi(argv[1]) : 40;
    double damp = (argc > 2) ? atof(argv[2]) : 0.0;  // 0 => fictitious play (alpha=1/(k+1))
    printf("=== Flip 7 - Chapter 4: symmetric Nash equilibrium (numbers-only) ===\n");
    printf("Value is 0.5 by symmetry; fictitious self-play converges the equilibrium policy.\n");
    printf("iterations=%d  update=%s\n\n", iters, damp > 0 ? "fixed-damping" : "fictitious-play");

    init_round_tables();
    const std::vector<double> D = round_pmf_numbers();
    const auto sup = pmf_support(D);

    // pmfOpp[(s)*kSn + y] : current belief about the opponent's round-score pmf at state s.
    std::vector<double> pmfOpp((size_t)kT * kT * kSn, 0.0);
    std::vector<double> pmfBR((size_t)kT * kT * kSn, 0.0);
    for (size_t s = 0; s < (size_t)kT * kT; ++s)
        for (int y = 0; y <= kRoundScoreMax; ++y) pmfOpp[s * kSn + y] = D[y];  // start: greedy field

    std::vector<double> W = win_prob_greedy(D, kT);   // warm start for the inner solves
    std::vector<double> U(1 << kNumValues), Bc(1 << kNumValues), g(kSn);
    std::vector<uint8_t> hit(1 << kNumValues);
    std::vector<double> rd(kSn);

    auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < iters; ++k) {
        // best response to the current belief pmfOpp; fill W and pmfBR.
        for (int sumab = 2 * (kT - 1); sumab >= 0; --sumab) {
            const int alo = std::max(0, sumab - (kT - 1));
            const int ahi = std::min(kT - 1, sumab);
            for (int a = alo; a <= ahi; ++a) {
                const int b = sumab - a;
                const double* dopp = &pmfOpp[((size_t)b * kT + a) * kSn];  // opponent sits at (b,a)
                const double popp0 = dopp[0];
                for (int x = 0; x <= kRoundScoreMax; ++x) {
                    double gx = 0;
                    const int A = a + x;
                    for (int y : sup) {
                        const int B = b + y;
                        double term;
                        if (A >= kT || B >= kT) term = (A > B ? 1.0 : (A == B ? 0.5 : 0.0));
                        else                    term = W[(size_t)A * kT + B];
                        gx += dopp[y] * term;
                    }
                    g[x] = gx;
                }
                const double wprev = W[(size_t)a * kT + b];
                // g[0] picked up the y=0 self term popp0*W[a][b] (stale); remove it
                // so the closed-form re-adds the correct self contribution popp0*w.
                const double base0 = g[0] - popp0 * wprev;
                double w = wprev;
                for (int it = 0; it < 12; ++it) {
                    g[0] = base0 + popp0 * w;             // self-loop: both bust -> (a,b)
                    double B0;
                    const double U0 = round_solve(g.data(), U.data(), nullptr, Bc.data(), &B0);
                    const double denom = 1.0 - B0 * popp0;
                    const double wn = (denom > 1e-12) ? (U0 - B0 * popp0 * w) / denom : U0;
                    if (std::fabs(wn - w) < 1e-12) { w = wn; break; }
                    w = wn;
                }
                W[(size_t)a * kT + b] = w;
                g[0] = base0 + popp0 * w;
                round_solve(g.data(), U.data(), hit.data());
                round_dist(hit.data(), rd.data());
                double* dst = &pmfBR[((size_t)a * kT + b) * kSn];
                for (int y = 0; y <= kRoundScoreMax; ++y) dst[y] = rd[y];
            }
        }
        // Fictitious play: belief = running average of best responses (alpha=1/(k+1)),
        // which converges to a Nash strategy in a zero-sum game (Robinson). A fixed
        // damping can be forced with arg 2. (W[0] is the *pure* best-response value
        // vs the mixed belief; it need not reach 0.5 -- mixing closes that gap.)
        const double alpha = (damp > 0.0) ? damp : 1.0 / (k + 1);
        double moved = 0.0;
        for (size_t i = 0; i < pmfOpp.size(); ++i) {
            const double nv = (1.0 - alpha) * pmfOpp[i] + alpha * pmfBR[i];
            moved += std::fabs(nv - pmfOpp[i]);
            pmfOpp[i] = nv;
        }
        printf("  iter %2d: policy L1 move = %.5f   (pure best-response value vs belief = %.4f)\n",
               k + 1, moved / ((double)kT * kT), W[0]);
        fflush(stdout);
    }
    printf("  (%.1f s)\n\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    // Equilibrium policy is pmfOpp. Characterize it vs greedy (P(bust)/E[round]).
    auto desc = [&](int a, int b) {
        const double* p = &pmfOpp[((size_t)a * kT + b) * kSn];
        double pb = p[0], es = 0; for (int y = 0; y <= kRoundScoreMax; ++y) es += y * p[y];
        printf("     %4d   %4d   |  %.4f   %6.3f\n", a, b, pb, es);
    };
    printf("--- equilibrium round play (your total ~110, vary the gap) ---\n");
    printf("     you    opp    |  P(bust)  E[round]\n");
    desc(110, 160); desc(110, 135); desc(110, 110); desc(110, 85); desc(110, 60);
    printf("   (greedy field for reference: P(bust)=%.4f E[round]=%.3f)\n\n", D[0],
           [&]{ double m=0; for (int y=0;y<=kRoundScoreMax;++y) m+=y*D[y]; return m; }());

    // MC tournament: both players use the equilibrium policy -> ~0.5.
    std::vector<std::vector<double>> cdf((size_t)kT * kT);
    Xoshiro256pp rng; rng.seed(0xEEEEULL);
    auto draw = [&](int a, int b) {
        const double* p = &pmfOpp[((size_t)a * kT + b) * kSn];
        const double u = (rng.next() >> 11) * 0x1.0p-53;
        double c = 0; for (int y = 0; y <= kRoundScoreMax; ++y) { c += p[y]; if (u < c) return y; } return 0;
    };
    const uint64_t G = 3'000'000; long wins = 0; double half = 0;
    for (uint64_t gm = 0; gm < G; ++gm) {
        int a = 0, b = 0;
        for (;;) { const int da = draw(a, b), db = draw(b, a);  // simultaneous: both from pre-round state
            a += da; b += db;
            if (a >= kT || b >= kT) { if (a > b) ++wins; else if (a == b) half += 1; break; } }
    }
    printf("[MC] both play the converged equilibrium policy: P1 win rate = %.5f\n",
           (wins + 0.5 * half) / G);
    printf("     => the symmetric Nash value is 0.5 (as it must be by symmetry); the\n");
    printf("        equilibrium *policy* is the push-when-behind / safe-when-ahead schedule above.\n");
    return 0;
}
