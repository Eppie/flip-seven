// test_ch1b — assert the Stage b headline numbers (real rules, exact DPs).
//
// Part 1 (numbers + modifiers) and Part 2 (+ Second Chance) are each exact and
// independently cross-checked by Monte-Carlo.
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

// Regression values from the exact DPs (see README).
static constexpr double kOptMod  = 20.2291054307;  // numbers + modifiers
static constexpr double kOptFull = 22.2504034143;  // + Second Chance (real rules)

int main() {
    printf("Stage b tests\n");
    const uint64_t N = 20'000'000ULL;

    // ---- Part 1: numbers + modifiers (exact) ----
    SolitaireModDP dp_mod;
    const double opt_mod = dp_mod.optimal();
    const auto st_mod = dp_mod.analyze();
    check(close(st_mod.e_score, opt_mod, 1e-9), "mod: forward-pass equals DP optimum");
    check(close(st_mod.p_bust + st_mod.p_flip7 + st_mod.p_stay, 1.0, 1e-9), "mod: probabilities sum to 1");
    check(close(opt_mod, kOptMod, 1e-6), "mod: exact optimum == 20.2291054307");
    {
        const MCFullResult mc = monte_carlo_mod(dp_mod, N, 0xABCDEFULL);
        const double z = (mc.mean - opt_mod) / mc.stderr_;
        printf("  (mod MC: mean=%.6f z=%+.2f)\n", mc.mean, z);
        check(std::fabs(z) < 6.0, "mod: MC agrees with DP");
        check(close(mc.p_bust, st_mod.p_bust, 2e-4), "mod: MC P(bust) matches DP");
    }

    // ---- Part 2: + Second Chance (exact, real rules) ----
    SolitaireFullDP dp_full;
    const double opt_full = dp_full.optimal();
    check(close(opt_full, kOptFull, 1e-6), "full: exact optimum == 22.2504034143");
    check(dp_full.load_factor() < 0.7, "full: hash table load factor < 0.7");
    {
        const MCFullResult mc = monte_carlo_full(dp_full, N, 0xABCDEFULL);
        const double z = (mc.mean - opt_full) / mc.stderr_;
        printf("  (full MC: mean=%.6f z=%+.2f, saved=%.4f)\n", mc.mean, z, mc.p_saved);
        check(std::fabs(z) < 6.0, "full: MC agrees with exact DP");
    }

    // ---- Progression ----
    SolitaireTurnDP dp_num;
    const double opt_num = dp_num.optimal();
    check(opt_num < opt_mod && opt_mod < opt_full, "progression: numbers < +modifiers < +Second Chance");

    printf("%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED",
           failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
