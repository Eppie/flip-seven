// test_ch1b — assert the Stage b headline numbers.
//
// Part 1 (modifiers) is fully exact: DP, internal forward-pass, and an
// independent Monte-Carlo all agree. Part 2 (Second Chance) validates the DP
// math against an idealized-rules MC, and checks the (known, detectable)
// idealization gap to the true game.
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

// Regression values from the exact DP (see README).
static constexpr double kOptMod  = 20.2291054307;  // numbers + modifiers (exact)
static constexpr double kOptFull = 22.2317718622;  // + Second Chance (idealized model)

int main() {
    printf("Stage b tests\n");
    const uint64_t N = 20'000'000ULL;

    // ---- Part 1: numbers + modifiers (fully exact) ----
    SolitaireFullDP dp_mod(0);
    const double opt_mod = dp_mod.optimal();
    const auto st_mod = dp_mod.analyze();
    check(close(st_mod.e_score, opt_mod, 1e-9), "mod: forward-pass equals DP optimum");
    check(close(st_mod.p_bust + st_mod.p_flip7 + st_mod.p_stay, 1.0, 1e-9), "mod: probabilities sum to 1");
    check(close(opt_mod, kOptMod, 1e-6), "mod: exact optimum == 20.2291054307");
    {
        const MCFullResult mc = monte_carlo_full(dp_mod, N, 0xABCDEFULL, Rules::TrueRules);
        const double z = (mc.mean - opt_mod) / mc.stderr_;
        printf("  (mod MC: mean=%.6f z=%+.2f)\n", mc.mean, z);
        check(std::fabs(z) < 6.0, "mod: MC agrees with DP (no idealization here)");
        check(close(mc.p_bust, st_mod.p_bust, 2e-4), "mod: MC P(bust) matches DP");
    }

    // ---- Part 2: + Second Chance ----
    SolitaireFullDP dp_full(kNumSecondChance);
    const double opt_full = dp_full.optimal();
    const auto st_full = dp_full.analyze();
    check(close(st_full.e_score, opt_full, 1e-8), "full: forward-pass equals DP optimum");
    check(close(st_full.p_bust + st_full.p_flip7 + st_full.p_stay, 1.0, 1e-9), "full: probabilities sum to 1");
    check(close(opt_full, kOptFull, 1e-6), "full: idealized optimum == 22.2317718622");
    {
        // Idealized-rules MC must agree with the DP (same model) -> validates the math.
        const MCFullResult mci = monte_carlo_full(dp_full, N, 0xABCDEFULL, Rules::Idealized);
        const double zi = (mci.mean - opt_full) / mci.stderr_;
        printf("  (full idealized MC: mean=%.6f z=%+.2f)\n", mci.mean, zi);
        check(std::fabs(zi) < 6.0, "full: idealized MC agrees with DP");

        // True-rules MC is slightly higher (saved duplicate removed -> safer).
        const MCFullResult mct = monte_carlo_full(dp_full, N, 0xABCDEFULL, Rules::TrueRules);
        const double gap = mct.mean - opt_full;
        printf("  (full true MC:      mean=%.6f gap=%+.6f)\n", mct.mean, gap);
        check(gap > 0.008 && gap < 0.040, "full: true-game idealization gap is small & positive (~0.02)");
    }

    // ---- Progression ----
    SolitaireTurnDP dp_num;
    const double opt_num = dp_num.optimal();
    check(opt_num < opt_mod && opt_mod < opt_full, "progression: numbers < +modifiers < +Second Chance");

    printf("%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED",
           failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
