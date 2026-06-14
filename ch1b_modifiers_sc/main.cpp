// Chapter 1, Stage b: solitaire single-turn DP with modifiers and Second Chance.
//
//   Part 1: numbers + modifiers (+N, x2)        -- fully exact, no idealization.
//   Part 2: + Second Chance                     -- exact for the idealized model
//           (saved duplicate returned to deck); the gap to the true game is
//           measured by a true-rules Monte-Carlo.
//
//   usage: ch1b_modifiers_sc [num_rollouts] [seed]
#include "flip7_core.hpp"
#include "flip7_dp.hpp"
#include "flip7_sim.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

using namespace flip7;
using clk = std::chrono::steady_clock;
static double secs(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

static void report_dp(const char* tag, SolitaireFullDP& dp, double solve_ms) {
    const auto st = dp.analyze();
    printf("[DP %s] exact optimal E[score]       = %.10f\n", tag, dp.ev[SolitaireFullDP::enc(0, 0, 0, 0)]);
    printf("[DP %s] forward-pass E[score] (check) = %.10f  (delta %.2e)\n",
           tag, st.e_score, st.e_score - dp.ev[SolitaireFullDP::enc(0, 0, 0, 0)]);
    printf("[DP %s] P(bust)/P(Flip7)/P(stay)      = %.6f / %.6f / %.6f   (sum %.10f)\n",
           tag, st.p_bust, st.p_flip7, st.p_stay, st.p_bust + st.p_flip7 + st.p_stay);
    printf("[DP %s] states evaluated / solve time = %ld / %.2f ms\n",
           tag, dp.states_evaluated, solve_ms);
}

static void report_mc(const char* tag, const MCFullResult& mc, double opt, double sec) {
    const double z = (mc.mean - opt) / mc.stderr_;
    printf("[MC %s] mean=%.6f (se %.6f)  mean-DP=%+.6f (z=%+.2f)  "
           "bust/flip7/stay=%.4f/%.5f/%.4f  saved=%.4f  %.2f M/s\n",
           tag, mc.mean, mc.stderr_, mc.mean - opt, z,
           mc.p_bust, mc.p_flip7, mc.p_stay, mc.p_saved, mc.n / sec / 1e6);
}

int main(int argc, char** argv) {
    uint64_t N = 50'000'000ULL;
    uint64_t seed = 0xF117ULL;
    if (argc > 1) N = strtoull(argv[1], nullptr, 10);
    if (argc > 2) seed = strtoull(argv[2], nullptr, 10);

    printf("=== Flip 7 - Chapter 1 / Stage b: modifiers + Second Chance ===\n");
    printf("Number cards count toward Flip 7; modifiers/Second Chance do not.\n");
    printf("Score = sum(numbers) x (x2?2:1) + (+N modifiers) + (15 if 7 numbers).\n\n");

    // ---- Reference: numbers only (Chapter 1 milestone) ----
    SolitaireTurnDP dp_num;
    const double opt_num = dp_num.optimal();

    // ---- Part 1: numbers + modifiers (deck = 85, no Second Chance) ----
    printf("--- Part 1: numbers + modifiers (deck = 85) ---\n");
    auto t0 = clk::now();
    SolitaireFullDP dp_mod(0);
    const double opt_mod = dp_mod.optimal();
    auto t1 = clk::now();
    report_dp("mod", dp_mod, secs(t0, t1) * 1e3);
    {
        auto m0 = clk::now();
        const MCFullResult mc = monte_carlo_full(dp_mod, N, seed, Rules::TrueRules);
        auto m1 = clk::now();
        report_mc("mod, exact rules", mc, opt_mod, secs(m0, m1));
    }
    printf("\n");

    // ---- Part 2: + Second Chance (deck = 88) ----
    printf("--- Part 2: numbers + modifiers + Second Chance (deck = 88) ---\n");
    auto t2 = clk::now();
    SolitaireFullDP dp_full(kNumSecondChance);
    const double opt_full = dp_full.optimal();
    auto t3 = clk::now();
    report_dp("full", dp_full, secs(t2, t3) * 1e3);
    {
        auto m0 = clk::now();
        const MCFullResult mci = monte_carlo_full(dp_full, N, seed, Rules::Idealized);
        auto m1 = clk::now();
        report_mc("full, idealized (validates DP)", mci, opt_full, secs(m0, m1));

        auto m2 = clk::now();
        const MCFullResult mct = monte_carlo_full(dp_full, N, seed, Rules::TrueRules);
        auto m3 = clk::now();
        report_mc("full, TRUE rules (real game)  ", mct, opt_full, secs(m2, m3));
        printf("[gap] idealized DP optimum - true-rules MC mean = %+.6f  "
               "(idealization cost on E[score])\n", opt_full - mct.mean);
    }
    printf("\n");

    // ---- Progression ----
    printf("--- E[round score] progression (single solitaire turn) ---\n");
    printf("  numbers only            : %.4f\n", opt_num);
    printf("  + modifiers (+N, x2)    : %.4f   (+%.4f)\n", opt_mod, opt_mod - opt_num);
    printf("  + Second Chance         : %.4f   (+%.4f)\n", opt_full, opt_full - opt_mod);
    return 0;
}
