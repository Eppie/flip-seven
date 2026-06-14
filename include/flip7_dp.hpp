// flip7_dp.hpp — exact dynamic programming for Flip 7 (grows per chapter).
//
// Chapter 1: numbers-only solitaire single turn. Maximize E[round score] by
// optimal stopping. State is the 13-bit set of held numbers; because we draw
// WITHOUT replacement and every held value was drawn exactly once, the held
// hand fully determines the remaining deck — so the mask is the entire state.
#pragma once
#include "flip7_core.hpp"
#include <array>
#include <cstdint>

namespace flip7 {

struct SolitaireTurnDP {
    static constexpr int kStates = 1 << kNumValues;  // 8192

    std::array<double, kStates> ev{};
    std::array<bool, kStates>   done{};
    std::array<bool, kStates>   hit{};  // optimal policy: true => Hit, false => Stay
    long states_evaluated = 0;

    // EV(S): expected round score under optimal play from held set S.
    //   pop == 7         -> forced Flip 7, terminal value = sum + 15
    //   else  EV_stay = sum(S)
    //         EV_hit  = sum over undrawn v of P(draw v) * EV(S | v)   [bust -> 0]
    //         EV(S)   = max(EV_stay, EV_hit)
    double solve(uint16_t S) {
        if (done[S]) return ev[S];
        const int pc = maskPop(S);
        double val;
        if (pc == kFlip7Target) {
            val = (double)(maskSum(S) + kFlip7Bonus);
        } else {
            const int T = kNumberDeckSize - pc;  // cards left = 79 - (cards drawn)
            double ev_hit = 0.0;
            for (int v = 0; v < kNumValues; ++v) {
                const uint16_t bit = (uint16_t)(1u << v);
                if (S & bit) continue;  // drawing a held value busts -> contributes 0
                ev_hit += (double)numberCount(v) / (double)T * solve((uint16_t)(S | bit));
            }
            const double ev_stay = (double)maskSum(S);
            if (ev_hit > ev_stay) { val = ev_hit;  hit[S] = true; }
            else                  { val = ev_stay; hit[S] = false; }
        }
        ev[S] = val;
        done[S] = true;
        ++states_evaluated;
        return val;
    }

    double optimal() { return solve(0); }

    // Forward pass over the optimal policy: outcome probabilities and E[score].
    // E[score] here must reproduce optimal() exactly (independent consistency check).
    struct Stats { double e_score, p_bust, p_flip7, p_stay; };

    Stats analyze() {
        std::array<double, kStates> reach{};
        reach.fill(0.0);
        reach[0] = 1.0;
        double e = 0, pb = 0, pf = 0, ps = 0;
        // Hits only raise popcount, so processing by ascending popcount is a valid order.
        for (int pc = 0; pc < kFlip7Target; ++pc) {
            for (uint32_t S = 0; S < (uint32_t)kStates; ++S) {
                if (maskPop((uint16_t)S) != pc) continue;
                const double p = reach[S];
                if (p == 0.0) continue;
                if (!hit[S]) {
                    ps += p;
                    e  += p * (double)maskSum((uint16_t)S);
                    continue;
                }
                const int T = kNumberDeckSize - pc;
                double bust_cards = 0;
                for (int v = 0; v < kNumValues; ++v)
                    if (S & (1u << v)) bust_cards += numberCount(v) - 1;
                pb += p * bust_cards / (double)T;
                for (int v = 0; v < kNumValues; ++v) {
                    const uint16_t bit = (uint16_t)(1u << v);
                    if (S & bit) continue;
                    const double pv = p * (double)numberCount(v) / (double)T;
                    const uint16_t Sn = (uint16_t)(S | bit);
                    if (maskPop(Sn) == kFlip7Target) {
                        pf += pv;
                        e  += pv * (double)(maskSum(Sn) + kFlip7Bonus);
                    } else {
                        reach[Sn] += pv;
                    }
                }
            }
        }
        return {e, pb, pf, ps};
    }
};

}  // namespace flip7
