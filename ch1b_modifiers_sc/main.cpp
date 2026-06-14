// Chapter 1, Stage b: solitaire single-turn DP with modifiers and Second Chance.
//
// Both parts model the REAL game rules (draw without replacement) exactly:
//   Part 1: numbers + modifiers (+N, x2)   -- deck 85, dense DP.
//   Part 2: + Second Chance                -- deck 88, exact DP over a hashed
//           state that tracks per-value duplicates removed by saves.
// Each exact DP is cross-checked against an independent Monte-Carlo simulator.
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

static void report_mc(const char* tag, const MCFullResult& mc, double opt, double sec) {
    const double z = (mc.mean - opt) / mc.stderr_;
    printf("[MC %s] mean=%.6f (se %.6f)  mean-DP=%+.6f (z=%+.2f)  "
           "bust/flip7/stay=%.4f/%.5f/%.4f  saved=%.4f  %.1f M/s\n",
           tag, mc.mean, mc.stderr_, mc.mean - opt, z,
           mc.p_bust, mc.p_flip7, mc.p_stay, mc.p_saved, mc.n / sec / 1e6);
}

int main(int argc, char** argv) {
    uint64_t N = 50'000'000ULL;
    uint64_t seed = 0xF117ULL;
    if (argc > 1) N = strtoull(argv[1], nullptr, 10);
    if (argc > 2) seed = strtoull(argv[2], nullptr, 10);

    printf("=== Flip 7 - Chapter 1 / Stage b: modifiers + Second Chance ===\n");
    printf("Real rules, draw without replacement. Number cards count toward Flip 7;\n");
    printf("modifiers/Second Chance do not. Score = sum(numbers) x (x2?2:1)\n");
    printf("                                 + (+N modifiers) + (15 if 7 numbers).\n\n");

    // ---- Reference: numbers only (Chapter 1 milestone) ----
    SolitaireTurnDP dp_num;
    const double opt_num = dp_num.optimal();

    // ---- Part 1: numbers + modifiers (deck = 85) ----
    printf("--- Part 1: numbers + modifiers (deck = 85) ---\n");
    auto t0 = clk::now();
    SolitaireModDP dp_mod;
    const double opt_mod = dp_mod.optimal();
    auto t1 = clk::now();
    const auto st_mod = dp_mod.analyze();
    printf("[DP] exact optimal E[score]        = %.10f\n", opt_mod);
    printf("[DP] forward-pass E[score] (check)  = %.10f  (delta %.2e)\n", st_mod.e_score, st_mod.e_score - opt_mod);
    printf("[DP] P(bust)/P(Flip7)/P(stay)       = %.6f / %.6f / %.6f  (sum %.10f)\n",
           st_mod.p_bust, st_mod.p_flip7, st_mod.p_stay, st_mod.p_bust + st_mod.p_flip7 + st_mod.p_stay);
    printf("[DP] states / solve time            = %ld / %.2f ms\n", dp_mod.states_evaluated, secs(t0, t1) * 1e3);
    {
        auto m0 = clk::now();
        const MCFullResult mc = monte_carlo_mod(dp_mod, N, seed);
        report_mc("exact rules", mc, opt_mod, secs(m0, clk::now()));
    }
    printf("\n");

    // ---- Part 2: + Second Chance (deck = 88) ----
    printf("--- Part 2: numbers + modifiers + Second Chance (deck = 88) ---\n");
    auto t2 = clk::now();
    SolitaireFullDP dp_full;
    const double opt_full = dp_full.optimal();
    auto t3 = clk::now();
    printf("[DP] exact optimal E[score]        = %.10f\n", opt_full);
    printf("[DP] states / solve time           = %ld / %.2f ms   (hash load factor %.3f)\n",
           dp_full.states_evaluated, secs(t2, t3) * 1e3, dp_full.load_factor());
    {
        auto m0 = clk::now();
        const MCFullResult mc = monte_carlo_full(dp_full, N, seed);
        report_mc("exact rules", mc, opt_full, secs(m0, clk::now()));
    }
    printf("\n");

    // ---- Progression ----
    printf("--- E[round score] progression (single solitaire turn, exact) ---\n");
    printf("  numbers only            : %.4f\n", opt_num);
    printf("  + modifiers (+N, x2)    : %.4f   (+%.4f)\n", opt_mod, opt_mod - opt_num);
    printf("  + Second Chance         : %.4f   (+%.4f)\n", opt_full, opt_full - opt_mod);
    return 0;
}
