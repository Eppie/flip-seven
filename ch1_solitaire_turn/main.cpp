// Chapter 1: numbers-only solitaire single-turn optimal stopping.
//
// Computes the exact optimal expected round score via DP, then confirms it with
// an independent Monte-Carlo simulation. Reports timing/throughput throughout.
//
//   usage: ch1_solitaire_turn [num_rollouts] [seed]
#include "flip7_core.hpp"
#include "flip7_dp.hpp"
#include "flip7_sim.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

using namespace flip7;
using clk = std::chrono::steady_clock;
static double secs(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

int main(int argc, char** argv) {
    uint64_t N = 100'000'000ULL;
    uint64_t seed = 0xF117ULL;
    if (argc > 1) N = strtoull(argv[1], nullptr, 10);
    if (argc > 2) seed = strtoull(argv[2], nullptr, 10);

    printf("=== Flip 7 - Chapter 1: numbers-only solitaire single turn ===\n");
    printf("Deck : 79 number cards (one 0, one 1, two 2s, ... twelve 12s)\n");
    printf("Model: draw WITHOUT replacement; Stay banks the sum of held numbers;\n");
    printf("       a duplicate busts (score 0); 7 unique numbers => +15 and stop.\n\n");

    // ---- Exact DP ----
    auto t0 = clk::now();
    SolitaireTurnDP dp;
    const double opt = dp.optimal();
    auto t1 = clk::now();
    const auto st = dp.analyze();
    auto t2 = clk::now();

    printf("[DP] exact optimal E[score]        = %.10f\n", opt);
    printf("[DP] forward-pass E[score] (check)  = %.10f   (delta %.2e)\n",
           st.e_score, st.e_score - opt);
    printf("[DP] P(bust)   under optimal policy = %.10f\n", st.p_bust);
    printf("[DP] P(Flip 7) under optimal policy = %.10f\n", st.p_flip7);
    printf("[DP] P(stay)   under optimal policy = %.10f\n", st.p_stay);
    printf("[DP] probability sum (check)        = %.12f\n",
           st.p_bust + st.p_flip7 + st.p_stay);
    printf("[DP] states evaluated               = %ld\n", dp.states_evaluated);
    printf("[DP] solve time                     = %.4f ms\n", secs(t0, t1) * 1e3);
    printf("[DP] analyze time                   = %.4f ms\n\n", secs(t1, t2) * 1e3);

    // ---- Independent Monte-Carlo cross-check ----
    auto m0 = clk::now();
    const MCResult mc = monte_carlo_solitaire(dp, N, seed);
    auto m1 = clk::now();
    const double sec = secs(m0, m1);
    const double z = (mc.mean - opt) / mc.stderr_;

    printf("[MC] rollouts                       = %llu\n", (unsigned long long)N);
    printf("[MC] mean score                     = %.6f   (stderr %.6f, sd %.4f)\n",
           mc.mean, mc.stderr_, mc.stddev);
    printf("[MC] mean - DP                      = %+.6f  (z = %+.2f sigma)\n",
           mc.mean - opt, z);
    printf("[MC] P(bust)                        = %.6f   (DP %.6f, d=%+.2e)\n",
           mc.p_bust, st.p_bust, mc.p_bust - st.p_bust);
    printf("[MC] P(Flip 7)                      = %.6f   (DP %.6f, d=%+.2e)\n",
           mc.p_flip7, st.p_flip7, mc.p_flip7 - st.p_flip7);
    printf("[MC] time                           = %.3f s\n", sec);
    printf("[MC] throughput                     = %.1f M rollouts/s\n\n", N / sec / 1e6);

    // ---- Policy preview (foreshadows Chapter 2 non-separability) ----
    printf("--- policy preview: same #uniques / similar sums, different decisions ---\n");
    auto show = [&](const char* name, std::initializer_list<int> vals) {
        uint16_t m = 0;
        for (int v : vals) m |= (uint16_t)(1u << v);
        dp.solve(m);
        printf("  %-16s pop=%d sum=%2d  EV=%7.4f  ->  %s\n",
               name, maskPop(m), maskSum(m), dp.ev[m], dp.hit[m] ? "HIT" : "STAY");
    };
    show("{1,2,3}", {1, 2, 3});
    show("{10,11,12}", {10, 11, 12});
    show("{0,1,2,3,4}", {0, 1, 2, 3, 4});
    show("{2,4,6,8,9}", {2, 4, 6, 8, 9});
    show("{6,7,8,9,10}", {6, 7, 8, 9, 10});
    return 0;
}
