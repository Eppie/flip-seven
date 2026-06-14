// test_all_cards — assert the complete 94-card solitaire headline (exact + MC).
#include "flip7_core.hpp"
#include "flip7_dp.hpp"
#include "flip7_sim.hpp"

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

// Regression value from the exact DP (1,208,732,216 states; see README).
static constexpr double kOptAll = 20.2980220287;

int main() {
    printf("All-94-cards solitaire tests\n");
    SolitaireAllDPBlocked dp;
    const double opt = dp.optimal();
    printf("  (exact optimum = %.10f, states = %ld, load = %.3f)\n",
           opt, dp.states_evaluated, dp.load_factor());

    check(dp.load_factor() < 0.7, "hash table load factor < 0.7");
    check(close(opt, kOptAll, 1e-6), "exact optimum matches regression");
    {
        const MCFullResult mc = monte_carlo_all(dp, 20'000'000ULL, 0xABCDEFULL);
        const double z = (mc.mean - opt) / mc.stderr_;
        printf("  (MC mean=%.6f z=%+.2f, froze=%.4f flip3=%.4f saved=%.4f)\n",
               mc.mean, z, mc.p_froze, mc.p_flip3, mc.p_saved);
        check(std::fabs(z) < 6.0, "MC agrees with exact DP");
    }

    // Adding Freeze + Flip Three changes E[score] vs the 88-card deck.
    SolitaireFullDP dp_full;
    const double opt_full = dp_full.optimal();
    printf("  (88-card optimum = %.6f, all-94 = %.6f, delta = %+.6f)\n",
           opt_full, opt, opt - opt_full);

    printf("%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED",
           failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
