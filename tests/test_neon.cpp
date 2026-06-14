// test_neon — the NEON-vectorized MC must agree with the scalar MC and the exact
// DP (it samples the same draw-without-replacement conditional, just 4 lanes at a
// time). Also prints the measured speedup. On non-ARM the NEON entry point is the
// scalar simulator, so this still builds and trivially passes.
#include "flip7_compete.hpp"   // round_pmf_numbers (exact score pmf)
#include "flip7_dp.hpp"
#include "flip7_sim.hpp"
#include "flip7_sim_neon.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace flip7;
using clk = std::chrono::steady_clock;

static int failures = 0;
static void check(bool ok, const char* what) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}
static bool close(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

int main() {
    printf("NEON MC tests\n");
    SolitaireTurnDP dp;
    const double opt = dp.optimal();
    const auto st = dp.analyze();  // exact p_bust / p_flip7 / p_stay

    const uint64_t N = 50'000'000ULL;
    const auto t0 = clk::now();
    const MCResult sc = monte_carlo_solitaire(dp, N, 0xABCDEF01ULL);
    const auto t1 = clk::now();
    const MCResult nv = monte_carlo_solitaire_neon(dp, N, 0x1234ABCDULL);
    const auto t2 = clk::now();

    // NEON vs exact DP.
    check(close(nv.mean, opt, 0.02), "NEON mean == exact optimum (18.5652)");
    check(close(nv.p_bust, st.p_bust, 0.001), "NEON P(bust) == exact DP");
    check(close(nv.p_flip7, st.p_flip7, 0.0005), "NEON P(Flip 7) == exact DP");
    // NEON vs scalar MC (independent generators, same conditional).
    check(close(nv.mean, sc.mean, 0.02), "NEON mean agrees with scalar MC mean");
    check(close(nv.p_bust, sc.p_bust, 0.001), "NEON P(bust) agrees with scalar MC");

    // Flip-7-maximizing policy (always hit): a separate exact target.
    SolitaireTurnDP ah; ah.hit.fill(true);
    const MCResult nv_ah = monte_carlo_solitaire_neon(ah, N, 0x55AA55AAULL);
    const MCResult sc_ah = monte_carlo_solitaire(ah, N, 0x9988ULL);
    check(close(nv_ah.p_flip7, 0.08661, 0.001), "NEON always-hit P(Flip 7) == 0.0866 (exact max)");
    check(close(nv_ah.p_flip7, sc_ah.p_flip7, 0.001), "NEON always-hit agrees with scalar MC");
    check(close(nv_ah.p_bust, 1.0 - nv_ah.p_flip7, 1e-9), "NEON always-hit: P(bust) = 1 - P(Flip 7)");

    // Strongest check: the FULL realized score distribution must match the exact
    // pmf bin-by-bin (not just the mean / bust / flip7 summary stats). This is what
    // pins down "same deck behavior" -- same draw-without-replacement law.
#if defined(__ARM_NEON)
    {
        std::vector<double> hist(kRoundScoreMax + 1, 0.0);
        const MCResult hv = monte_carlo_solitaire_neon(dp, N, 0xFEED1234ULL, hist.data());
        const std::vector<double> D = round_pmf_numbers();   // exact score pmf
        double maxdev = 0.0; int worst = -1;
        for (int s = 0; s <= kRoundScoreMax; ++s) {
            const double f = hist[s] / (double)hv.n;
            const double ex = (s < (int)D.size()) ? D[s] : 0.0;
            if (std::fabs(f - ex) > maxdev) { maxdev = std::fabs(f - ex); worst = s; }
        }
        printf("  full-distribution check: max |NEON freq - exact pmf| = %.5f at score %d\n", maxdev, worst);
        check(maxdev < 0.0015, "NEON score distribution matches the exact pmf across all 79 bins");
    }
#endif

    const double sc_ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / (double)N;
    const double nv_ns = std::chrono::duration<double, std::nano>(t2 - t1).count() / (double)N;
    printf("  [bench %lluM rollouts] scalar %.2f ns/roll  |  NEON %.2f ns/roll  =>  %.2fx\n",
           (unsigned long long)(N / 1'000'000), sc_ns, nv_ns, sc_ns / nv_ns);

    printf("%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
