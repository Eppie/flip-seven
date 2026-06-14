// flip7_dp.hpp — exact dynamic programming for Flip 7 (grows per chapter).
//
// Chapter 1: numbers-only solitaire single turn. Maximize E[round score] by
// optimal stopping. State is the 13-bit set of held numbers; because we draw
// WITHOUT replacement and every held value was drawn exactly once, the held
// hand fully determines the remaining deck — so the mask is the entire state.
#pragma once
#include "flip7_core.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

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

// -----------------------------------------------------------------------------
// Stage b: solitaire single turn with modifiers (+N, x2) and Second Chance.
//
// Deck = 79 numbers + 6 modifiers + num_sc Second Chance cards (num_sc in {0,3}).
// State = (number_mask, modifier_mask, sc_held, sc_seen):
//   sc_seen = Second Chance cards drawn so far (0..num_sc)  -> remaining = num_sc-sc_seen
//   sc_held = whether one Second Chance is currently in hand (0/1)
//
// Idealization: a Second Chance "save" returns the duplicate number card to the
// deck (so a held value's remaining copies stay count(v)-1). This keeps the held
// hand + sc_seen a sufficient statistic for the deck, making the DP exact for
// this model. The gap to the true game (where the duplicate is removed) is
// measured by the true-rules Monte-Carlo.
struct SolitaireFullDP {
    static constexpr int kBits   = 22;          // nm(13)|mm(6)|sc_held(1)|sc_seen(2)
    static constexpr int kStates = 1 << kBits;

    int  num_sc;
    int  deck_total;
    std::vector<double>  ev;
    std::vector<uint8_t> done;
    std::vector<uint8_t> hit;   // policy: 1 => Hit
    long states_evaluated = 0;

    explicit SolitaireFullDP(int second_chance_cards)
        : num_sc(second_chance_cards),
          deck_total(kNumberDeckSize + kNumModifiers + second_chance_cards),
          ev(kStates, 0.0), done(kStates, 0), hit(kStates, 0) {}

    static int enc(uint16_t nm, uint16_t mm, int sch, int scn) {
        return (int)nm | ((int)mm << 13) | (sch << 19) | (scn << 20);
    }

    double solve(uint16_t nm, uint16_t mm, int sch, int scn) {
        const int id = enc(nm, mm, sch, scn);
        if (done[id]) return ev[id];
        const int pcn = maskPop(nm);
        double val;
        if (pcn == kFlip7Target) {
            val = (double)fullScore(nm, mm);  // forced Flip 7 (includes +15)
        } else {
            const int drawn = pcn + maskPop(mm) + scn;  // idealized: saved dups not counted
            const int T = deck_total - drawn;
            double ev_hit = 0.0;
            // Number cards.
            for (int v = 0; v < kNumValues; ++v) {
                const uint16_t bit = (uint16_t)(1u << v);
                if (nm & bit) {
                    const double p = (double)(numberCount(v) - 1) / (double)T;  // duplicate
                    if (p > 0.0 && sch)                       // save: consume SC, dup returned
                        ev_hit += p * solve(nm, mm, 0, scn);
                    // else bust -> contributes 0
                } else {
                    const double p = (double)numberCount(v) / (double)T;
                    const uint16_t nn = (uint16_t)(nm | bit);
                    if (maskPop(nn) == kFlip7Target) ev_hit += p * (double)fullScore(nn, mm);
                    else                             ev_hit += p * solve(nn, mm, sch, scn);
                }
            }
            // Modifier cards (one of each; drawing one is always safe).
            for (int i = 0; i < kNumModifiers; ++i) {
                const uint16_t bit = (uint16_t)(1u << i);
                if (mm & bit) continue;
                ev_hit += (1.0 / (double)T) * solve(nm, (uint16_t)(mm | bit), sch, scn);
            }
            // Second Chance card: gain one (or discard the excess if already holding).
            if (num_sc > 0 && scn < num_sc) {
                const double p = (double)(num_sc - scn) / (double)T;
                ev_hit += p * solve(nm, mm, 1, scn + 1);
            }
            const double ev_stay = (double)fullScore(nm, mm);
            if (ev_hit > ev_stay) { val = ev_hit;  hit[id] = 1; }
            else                  { val = ev_stay; hit[id] = 0; }
        }
        ev[id] = val;
        done[id] = 1;
        ++states_evaluated;
        return val;
    }

    double optimal() { return solve(0, 0, 0, 0); }

    // Forward pass over the optimal policy (independent of value iteration):
    // reproduces optimal() and yields outcome probabilities.
    struct Stats { double e_score, p_bust, p_flip7, p_stay; };

    Stats analyze() {
        // Topological order: by total cards drawn ascending; within a level the
        // only intra-level edge is a save (sch:1->0), so process sch=1 first.
        std::vector<int> order;
        order.reserve(states_evaluated);
        for (int id = 0; id < kStates; ++id) {
            if (!done[id]) continue;
            const uint16_t nm = (uint16_t)(id & 0x1FFF);
            if (maskPop(nm) == kFlip7Target) continue;  // terminal, not a decision node
            order.push_back(id);
        }
        auto level = [&](int id) {
            const uint16_t nm = (uint16_t)(id & 0x1FFF);
            const uint16_t mm = (uint16_t)((id >> 13) & 0x3F);
            const int scn = (id >> 20) & 0x3;
            return maskPop(nm) + maskPop(mm) + scn;
        };
        auto key = [&](int id) { return level(id) * 2 + (((id >> 19) & 1) ? 0 : 1); };
        std::sort(order.begin(), order.end(), [&](int a, int b) { return key(a) < key(b); });

        std::vector<double> reach(kStates, 0.0);
        reach[enc(0, 0, 0, 0)] = 1.0;
        double e = 0, pb = 0, pf = 0, ps = 0;
        for (int id : order) {
            const double p = reach[id];
            if (p == 0.0) continue;
            const uint16_t nm = (uint16_t)(id & 0x1FFF);
            const uint16_t mm = (uint16_t)((id >> 13) & 0x3F);
            const int sch = (id >> 19) & 1;
            const int scn = (id >> 20) & 0x3;
            if (!hit[id]) { ps += p; e += p * (double)fullScore(nm, mm); continue; }
            const int pcn = maskPop(nm);
            const int T = deck_total - (pcn + maskPop(mm) + scn);
            for (int v = 0; v < kNumValues; ++v) {
                const uint16_t bit = (uint16_t)(1u << v);
                if (nm & bit) {
                    const double pd = (double)(numberCount(v) - 1) / (double)T;
                    if (pd <= 0.0) continue;
                    if (sch) reach[enc(nm, mm, 0, scn)] += p * pd;
                    else     pb += p * pd;
                } else {
                    const double pv = (double)numberCount(v) / (double)T;
                    const uint16_t nn = (uint16_t)(nm | bit);
                    if (maskPop(nn) == kFlip7Target) { pf += p * pv; e += p * pv * (double)fullScore(nn, mm); }
                    else                              reach[enc(nn, mm, sch, scn)] += p * pv;
                }
            }
            for (int i = 0; i < kNumModifiers; ++i) {
                const uint16_t bit = (uint16_t)(1u << i);
                if (mm & bit) continue;
                reach[enc(nm, (uint16_t)(mm | bit), sch, scn)] += p * (1.0 / (double)T);
            }
            if (num_sc > 0 && scn < num_sc)
                reach[enc(nm, mm, 1, scn + 1)] += p * ((double)(num_sc - scn) / (double)T);
        }
        return {e, pb, pf, ps};
    }
};

}  // namespace flip7
