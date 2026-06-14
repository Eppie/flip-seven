// Chapter 3: tail probabilities -- P(bust) and P(Flip 7), the "perfect game".
//
// Contrasts the expected-score-optimal policy with the policy that MAXIMIZES
// P(Flip 7). Maximizing P(Flip 7) is "always hit" (staying => 0 chance of 7
// uniques, so any hit weakly dominates) -- we prove that with an exact
// Flip-7-maximizing DP, then show how strategy-dependent the tails are.
#include "flip7_core.hpp"
#include "flip7_dp.hpp"
#include "flip7_sim.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace flip7;

struct Tail { double e_score, p_bust, p_flip7, p_stay; };

// Outcome distribution of a Hit/Stay policy in the numbers-only game.
template <class Hit>
static Tail eval_tail(Hit should_hit) {
    std::vector<double> reach(1 << kNumValues, 0.0);
    reach[0] = 1.0;
    double e = 0, pb = 0, pf = 0, ps = 0;
    for (int p = 0; p < kFlip7Target; ++p)
        for (uint32_t S = 0; S < (1u << kNumValues); ++S) {
            if (maskPop((uint16_t)S) != p) continue;
            const double pr = reach[S];
            if (pr == 0.0) continue;
            if (!should_hit((uint16_t)S)) { ps += pr; e += pr * (double)maskSum((uint16_t)S); continue; }
            const int T = kNumberDeckSize - p;
            double bust = 0;
            for (int v = 0; v < kNumValues; ++v) if (S & (1u << v)) bust += numberCount(v) - 1;
            pb += pr * bust / (double)T;
            for (int v = 0; v < kNumValues; ++v) {
                const uint16_t bit = (uint16_t)(1u << v);
                if (S & bit) continue;
                const double pv = pr * (double)numberCount(v) / (double)T;
                const uint16_t Sn = (uint16_t)(S | bit);
                if (maskPop(Sn) == kFlip7Target) { pf += pv; e += pv * (double)(maskSum(Sn) + kFlip7Bonus); }
                else                              reach[Sn] += pv;
            }
        }
    return {e, pb, pf, ps};
}

// Exact Flip-7-maximizing DP (numbers only): V(S) = max P(reach 7 uniques).
static std::array<double, 1 << kNumValues> Vf;
static std::array<bool, 1 << kNumValues>   Vdone;
static double Vflip7(uint16_t S) {
    if (maskPop(S) == kFlip7Target) return 1.0;
    if (Vdone[S]) return Vf[S];
    const int T = kNumberDeckSize - maskPop(S);
    double hit = 0.0;
    for (int v = 0; v < kNumValues; ++v) {
        const uint16_t bit = (uint16_t)(1u << v);
        if (S & bit) continue;
        hit += (double)numberCount(v) / (double)T * Vflip7((uint16_t)(S | bit));
    }
    Vf[S] = hit;  // stay gives 0, so always hit
    Vdone[S] = true;
    return Vf[S];
}

int main() {
    printf("=== Flip 7 - Chapter 3: tail probabilities (the \"perfect game\") ===\n\n");

    // ---- Numbers-only, exact ----
    SolitaireTurnDP dp;
    dp.optimal();
    const Tail ev_opt = eval_tail([&](uint16_t S){ return dp.hit[S]; });
    const Tail allhit = eval_tail([](uint16_t){ return true; });
    const double maxflip7 = Vflip7(0);

    printf("--- numbers only (exact) ---\n");
    printf("                          E[score]   P(bust)    P(Flip 7)\n");
    printf("  expected-optimal play    %7.4f   %.5f    %.5f\n", ev_opt.e_score, ev_opt.p_bust, ev_opt.p_flip7);
    printf("  Flip-7-max (always hit)  %7.4f   %.5f    %.5f\n", allhit.e_score, allhit.p_bust, allhit.p_flip7);
    printf("  exact max P(Flip 7) DP   = %.5f  (matches always-hit: %s)\n",
           maxflip7, (maxflip7 - allhit.p_flip7 < 1e-9 && allhit.p_flip7 - maxflip7 < 1e-9) ? "yes" : "NO");
    printf("  => going for Flip 7 raises P(Flip 7) %.0fx (%.3f%% -> %.2f%%) but P(bust) %.4f -> %.4f\n",
           allhit.p_flip7 / ev_opt.p_flip7, 100 * ev_opt.p_flip7, 100 * allhit.p_flip7,
           ev_opt.p_bust, allhit.p_bust);
    printf("     and collapses E[score] %.2f -> %.2f.\n\n", ev_opt.e_score, allhit.e_score);

    // MC cross-check both policies (numbers only).
    SolitaireTurnDP allhit_dp;
    allhit_dp.hit.fill(true);
    const MCResult mc_opt = monte_carlo_solitaire(dp, 50'000'000ULL, 0xF117ULL);
    const MCResult mc_ah  = monte_carlo_solitaire(allhit_dp, 50'000'000ULL, 0xF117ULL);
    printf("  [MC] optimal : P(bust)=%.5f P(Flip7)=%.5f  (DP %.5f / %.5f)\n",
           mc_opt.p_bust, mc_opt.p_flip7, ev_opt.p_bust, ev_opt.p_flip7);
    printf("  [MC] alwayshit: P(bust)=%.5f P(Flip7)=%.5f  (DP %.5f / %.5f)\n\n",
           mc_ah.p_bust, mc_ah.p_flip7, allhit.p_bust, allhit.p_flip7);

    // ---- All 94 cards ----
    printf("--- all 94 cards ---\n");
    const MCFullResult full_ah = monte_carlo_all_alwayshit(50'000'000ULL, 0xF117ULL);
    printf("  expected-optimal play   : P(bust)=0.2935  P(Flip 7)=0.01259  (from the all-94 exact solve)\n");
    printf("  Flip-7-max (always hit) : P(bust)=%.4f  P(Flip 7)=%.4f  E[score]=%.4f\n",
           full_ah.p_bust, full_ah.p_flip7, full_ah.mean);
    printf("  Second Chance saves %.1f%% of always-hit turns, Flip Three forces draws in %.1f%%,\n",
           100 * full_ah.p_saved, 100 * full_ah.p_flip3);
    printf("  and a forced Freeze ends the attempt %.1f%% of the time.\n", 100 * full_ah.p_froze);
    printf("  => with all cards, going for it reaches Flip 7 ~%.1f%% of the time vs ~1.3%% playing for score.\n",
           100 * full_ah.p_flip7);
    return 0;
}
