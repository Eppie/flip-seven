// test_ch1 — assert the Chapter 1 headline numbers.
//
// Checks: (1) DP internal consistency, (2) regression values for the exact
// optimum and outcome probabilities, (3) independent Monte-Carlo agreement,
// (4) the non-separability of the optimal policy.
#include "flip7_core.hpp"
#include "flip7_dp.hpp"
#include "flip7_sim.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <initializer_list>

using namespace flip7;

static int failures = 0;

static void check(bool ok, const char* what) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}
static bool close(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

// Regression values from the exact DP (see README headline table).
static constexpr double kOptExpected    = 18.5652176141;
static constexpr double kPBustExpected   = 0.3134964608;
static constexpr double kPFlip7Expected  = 0.0028113032;
static constexpr double kPStayExpected   = 0.6836922360;

static uint16_t mk(std::initializer_list<int> vals) {
    uint16_t m = 0;
    for (int v : vals) m |= (uint16_t)(1u << v);
    return m;
}

int main() {
    SolitaireTurnDP dp;
    const double opt = dp.optimal();
    const auto st = dp.analyze();

    printf("Chapter 1 tests\n");

    // (1) DP internal consistency.
    check(close(st.e_score, opt, 1e-9), "forward-pass E[score] equals DP optimum");
    check(close(st.p_bust + st.p_flip7 + st.p_stay, 1.0, 1e-9), "outcome probabilities sum to 1");

    // (2) Regression on the exact values.
    check(close(opt, kOptExpected, 1e-6), "exact optimum == 18.5652176141");
    check(close(st.p_bust,  kPBustExpected,  1e-6), "P(bust)   regression");
    check(close(st.p_flip7, kPFlip7Expected, 1e-6), "P(Flip 7) regression");
    check(close(st.p_stay,  kPStayExpected,  1e-6), "P(stay)   regression");

    // (3) Independent Monte-Carlo agreement (fixed seed for determinism).
    const uint64_t N = 30'000'000ULL;
    const MCResult mc = monte_carlo_solitaire(dp, N, 0xABCDEFULL);
    const double z = (mc.mean - opt) / mc.stderr_;
    printf("  (MC: mean=%.6f opt=%.6f z=%+.2f sigma, N=%llu)\n",
           mc.mean, opt, z, (unsigned long long)N);
    check(std::fabs(z) < 6.0, "MC mean within 6 sigma of DP optimum");
    check(close(mc.p_bust,  st.p_bust,  2e-4), "MC P(bust)   matches DP");
    check(close(mc.p_flip7, st.p_flip7, 5e-5), "MC P(Flip 7) matches DP");

    // (4) Non-separability: decisions differ at equal popcount; not a pure
    //     score/count threshold. Low numbers are safe to chase; high are not.
    check(dp.hit[0] == true, "policy hits from an empty hand");
    check(dp.hit[mk({1, 2, 3})] == true,    "HIT {1,2,3}   (low, safe)");
    check(dp.hit[mk({10, 11, 12})] == false, "STAY {10,11,12} (high, same popcount)");
    check(dp.hit[mk({0, 1, 2, 3, 4})] == true,  "HIT {0,1,2,3,4} (low 5-card hand)");
    check(dp.hit[mk({6, 7, 8, 9, 10})] == false, "STAY {6,7,8,9,10} (high 5-card hand)");

    printf("%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED",
           failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
