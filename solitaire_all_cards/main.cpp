// Complete solitaire single turn: ALL 94 cards, exact, real rules.
//
// Numbers + modifiers + Second Chance + Freeze + Flip Three. Action cards are
// self-targeted (solitaire): Freeze = forced Stay, Flip Three = forced 3 draws.
// Exact DP cross-checked against an independent Monte-Carlo simulator.
//
//   usage: solitaire_all_cards [num_rollouts] [seed]
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

int main(int argc, char** argv) {
    uint64_t N = 50'000'000ULL;
    uint64_t seed = 0xF117ULL;
    if (argc > 1) N = strtoull(argv[1], nullptr, 10);
    if (argc > 2) seed = strtoull(argv[2], nullptr, 10);

    printf("=== Flip 7 - complete solitaire single turn (all 94 cards) ===\n");
    printf("Self-targeted action cards: Freeze = forced Stay; Flip Three = forced\n");
    printf("3 draws (bust or 7th unique ends early; stacks; a Freeze drawn during\n");
    printf("it is pending and Stays after the forced draws). Real rules, no replacement.\n\n");

    // Progression references (fast exact solves).
    SolitaireTurnDP dp_num;  const double opt_num = dp_num.optimal();
    SolitaireModDP  dp_mod;  const double opt_mod = dp_mod.optimal();
    auto tf = clk::now();
    SolitaireFullDP dp_full; const double opt_full = dp_full.optimal();
    printf("(solved numbers / +modifiers / +Second Chance in %.2f s)\n\n", secs(tf, clk::now()));

    // The full 94-card exact DP.
    printf("--- all 94 cards (exact DP) ---\n");
    auto t0 = clk::now();
    SolitaireAllDP dp_all;
    const double opt_all = dp_all.optimal();
    auto t1 = clk::now();
    printf("[DP] exact optimal E[score]        = %.10f\n", opt_all);
    printf("[DP] states / solve time           = %ld / %.2f s   (hash load factor %.3f)\n",
           dp_all.states_evaluated, secs(t0, t1), dp_all.load_factor());

    auto m0 = clk::now();
    const MCFullResult mc = monte_carlo_all(dp_all, N, seed);
    const double sec = secs(m0, clk::now());
    const double z = (mc.mean - opt_all) / mc.stderr_;
    printf("[MC] mean=%.6f (se %.6f)  mean-DP=%+.6f (z=%+.2f)\n", mc.mean, mc.stderr_, mc.mean - opt_all, z);
    printf("[MC] bust/flip7/stay = %.4f / %.5f / %.4f   (froze %.4f, flip3 %.4f, saved %.4f)\n",
           mc.p_bust, mc.p_flip7, mc.p_stay, mc.p_froze, mc.p_flip3, mc.p_saved);
    printf("[MC] %.1f M rollouts/s\n\n", N / sec / 1e6);

    printf("--- E[round score] progression (single solitaire turn, exact) ---\n");
    printf("  numbers only            : %.4f\n", opt_num);
    printf("  + modifiers (+N, x2)    : %.4f   (%+.4f)\n", opt_mod, opt_mod - opt_num);
    printf("  + Second Chance         : %.4f   (%+.4f)\n", opt_full, opt_full - opt_mod);
    printf("  + Freeze + Flip Three   : %.4f   (%+.4f)   <- all 94 cards\n", opt_all, opt_all - opt_full);
    return 0;
}
