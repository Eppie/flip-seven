// test_ch4 — assert the across-rounds (first-to-200) results (numbers-only model).
//
// Covers the fast A/B/C layers exactly. The best-response grid (D) is heavy
// (~24 s) and is cross-checked by ch4_competitive's own Monte-Carlo; its headline
// W_br(0,0) = 0.5593 is recorded in the README.
#include "flip7_compete.hpp"

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

    printf("%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
