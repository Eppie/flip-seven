// test_ch3 — assert the tail-probability findings (numbers only + all cards).
#include "flip7_core.hpp"
#include "flip7_dp.hpp"
#include "flip7_sim.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

using namespace flip7;

static int failures = 0;
static void check(bool ok, const char* what) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}
static bool close(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

// Exact max P(Flip 7) DP (numbers only).
static std::array<double, 1 << kNumValues> Vf;
static std::array<bool, 1 << kNumValues>   Vd;
static double Vflip7(uint16_t S) {
    if (maskPop(S) == kFlip7Target) return 1.0;
    if (Vd[S]) return Vf[S];
    const int T = kNumberDeckSize - maskPop(S);
    double hit = 0.0;
    for (int v = 0; v < kNumValues; ++v) {
        const uint16_t bit = (uint16_t)(1u << v);
        if (S & bit) continue;
        hit += (double)numberCount(v) / (double)T * Vflip7((uint16_t)(S | bit));
    }
    Vf[S] = hit; Vd[S] = true;
    return hit;
}

int main() {
    printf("Chapter 3 tests\n");

    SolitaireTurnDP dp;
    dp.optimal();
    const auto opt = dp.analyze();                 // expected-optimal tails
    const double maxf = Vflip7(0);                 // exact max P(Flip 7)

    printf("  (numbers-only: P(Flip7) optimal=%.5f, max=%.5f)\n", opt.p_flip7, maxf);
    check(close(maxf, 0.0866127, 1e-5), "numbers: exact max P(Flip 7) == 0.08661");
    check(opt.p_flip7 < 0.003, "numbers: expected-optimal P(Flip 7) ~ 0.28%");
    check(maxf > 25 * opt.p_flip7, "numbers: max P(Flip 7) is >25x the score-optimal value");

    // Always-hit MC (numbers only) must match the Flip-7-max DP.
    SolitaireTurnDP allhit;
    allhit.hit.fill(true);
    const MCResult mc = monte_carlo_solitaire(allhit, 30'000'000ULL, 0xABCDEFULL);
    const double z = (mc.p_flip7 - maxf) / std::sqrt(maxf * (1 - maxf) / mc.n);
    printf("  (always-hit MC: P(Flip7)=%.5f vs DP %.5f, z=%+.2f; P(bust)=%.5f)\n",
           mc.p_flip7, maxf, z, mc.p_bust);
    check(std::fabs(z) < 6.0, "numbers: always-hit MC P(Flip 7) matches DP");
    check(close(mc.p_bust, 1.0 - maxf, 2e-4), "numbers: always-hit P(bust) = 1 - P(Flip 7)");

    // All 94 cards: going for it reaches Flip 7 far more often than playing for score.
    const MCFullResult full = monte_carlo_all_alwayshit(20'000'000ULL, 0xABCDEFULL);
    printf("  (all-94 always-hit MC: P(Flip7)=%.4f, P(bust)=%.4f, froze=%.4f)\n",
           full.p_flip7, full.p_bust, full.p_froze);
    check(full.p_flip7 > 0.05, "all-94: Flip-7-max reaches Flip 7 well above the ~1.3% score-optimal rate");

    printf("%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
