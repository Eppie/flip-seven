// flip7_dp.hpp — exact dynamic programming for Flip 7 (grows per chapter).
//
// All solvers here model the REAL game rules (draw without replacement) exactly.
//   SolitaireTurnDP : numbers only.
//   SolitaireModDP  : numbers + modifiers (+N, x2).
//   SolitaireFullDP : numbers + modifiers + Second Chance.
//
// A player's number cards form a set, so the held numbers are a 13-bit mask over
// {0..12}. Because we draw without replacement and each held value was drawn
// exactly once, the held hand determines the remaining deck for the number/
// modifier games; Second Chance needs a little extra state (see below).
#pragma once
#include "flip7_core.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace flip7 {

// =============================================================================
// Chapter 1 milestone: numbers only.
// =============================================================================
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

// =============================================================================
// Stage b.1: numbers + modifiers (+N, x2). Deck = 85 (79 numbers + 6 modifiers).
//
// Modifiers are one of each and cannot bust, so the held hand (number_mask,
// modifier_mask) fully determines the remaining deck -- a clean exact DP.
// State index: number_mask(13) | modifier_mask(6).
// =============================================================================
struct SolitaireModDP {
    static constexpr int kStates    = 1 << 19;
    static constexpr int kDeckTotal = kNumberDeckSize + kNumModifiers;  // 85

    std::vector<double>  ev;
    std::vector<uint8_t> done;
    std::vector<uint8_t> hit;
    long states_evaluated = 0;

    SolitaireModDP() : ev(kStates, 0.0), done(kStates, 0), hit(kStates, 0) {}

    static int enc(uint16_t nm, uint16_t mm) { return (int)nm | ((int)mm << 13); }

    double solve(uint16_t nm, uint16_t mm) {
        const int id = enc(nm, mm);
        if (done[id]) return ev[id];
        const int pcn = maskPop(nm);
        double val;
        if (pcn == kFlip7Target) {
            val = (double)fullScore(nm, mm);  // forced Flip 7 (includes +15)
        } else {
            const int T = kDeckTotal - (pcn + maskPop(mm));
            double ev_hit = 0.0;
            for (int v = 0; v < kNumValues; ++v) {
                const uint16_t bit = (uint16_t)(1u << v);
                if (nm & bit) continue;  // duplicate -> bust -> 0
                const double p = (double)numberCount(v) / (double)T;
                const uint16_t nn = (uint16_t)(nm | bit);
                if (maskPop(nn) == kFlip7Target) ev_hit += p * (double)fullScore(nn, mm);
                else                             ev_hit += p * solve(nn, mm);
            }
            for (int i = 0; i < kNumModifiers; ++i) {
                const uint16_t bit = (uint16_t)(1u << i);
                if (mm & bit) continue;
                ev_hit += (1.0 / (double)T) * solve(nm, (uint16_t)(mm | bit));
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

    double optimal() { return solve(0, 0); }

    struct Stats { double e_score, p_bust, p_flip7, p_stay; };

    Stats analyze() {
        std::vector<double> reach(kStates, 0.0);
        reach[enc(0, 0)] = 1.0;
        // Every transition adds one card, so process by total drawn ascending.
        std::vector<int> order;
        order.reserve(states_evaluated);
        for (int id = 0; id < kStates; ++id) {
            if (!done[id]) continue;
            if (maskPop((uint16_t)(id & 0x1FFF)) == kFlip7Target) continue;
            order.push_back(id);
        }
        auto drawn = [&](int id) {
            return maskPop((uint16_t)(id & 0x1FFF)) + maskPop((uint16_t)((id >> 13) & 0x3F));
        };
        std::sort(order.begin(), order.end(), [&](int a, int b) { return drawn(a) < drawn(b); });

        double e = 0, pb = 0, pf = 0, ps = 0;
        for (int id : order) {
            const double p = reach[id];
            if (p == 0.0) continue;
            const uint16_t nm = (uint16_t)(id & 0x1FFF);
            const uint16_t mm = (uint16_t)((id >> 13) & 0x3F);
            if (!hit[id]) { ps += p; e += p * (double)fullScore(nm, mm); continue; }
            const int T = kDeckTotal - (maskPop(nm) + maskPop(mm));
            double bust_cards = 0;
            for (int v = 0; v < kNumValues; ++v)
                if (nm & (1u << v)) bust_cards += numberCount(v) - 1;
            pb += p * bust_cards / (double)T;
            for (int v = 0; v < kNumValues; ++v) {
                const uint16_t bit = (uint16_t)(1u << v);
                if (nm & bit) continue;
                const double pv = p * (double)numberCount(v) / (double)T;
                const uint16_t nn = (uint16_t)(nm | bit);
                if (maskPop(nn) == kFlip7Target) { pf += pv; e += pv * (double)fullScore(nn, mm); }
                else                              reach[enc(nn, mm)] += pv;
            }
            for (int i = 0; i < kNumModifiers; ++i) {
                const uint16_t bit = (uint16_t)(1u << i);
                if (mm & bit) continue;
                reach[enc(nm, (uint16_t)(mm | bit))] += p * (1.0 / (double)T);
            }
        }
        return {e, pb, pf, ps};
    }
};

// =============================================================================
// Stage b.2: numbers + modifiers + Second Chance. Deck = 88. Real rules.
//
// A Second Chance save removes the duplicate number card from the deck, so we
// track per-value extra discards: extra[v] = saved duplicates of value v beyond
// the one held. Drawn copies of v = [v in nm] + extra[v]; remaining = count(v)
// minus that. Because a save permanently removes a card, every transition adds
// exactly one drawn card -> the state graph is acyclic.
//
// State (nm, mm, sc_held, sc_seen, extra[]) is sparse (sum of extra <= 3) so it
// does not pack into a dense index; we memoize in a flat open-addressing hash
// map (no std::unordered_map). Key: nm(13)|mm(6)|sc_held(1)|sc_seen(2)|
// extra(26: 2 bits per value v at bit 2v).
// =============================================================================
struct SolitaireFullDP {
    static constexpr int kDeckTotal = kNumberDeckSize + kNumModifiers + kNumSecondChance;  // 88
    static constexpr uint64_t kEmpty = ~0ull;

    size_t cap, mask_;
    std::vector<uint64_t> hkey;
    std::vector<double>   hval;
    std::vector<uint8_t>  hpol;
    long states_evaluated = 0;

    explicit SolitaireFullDP(size_t capacity_pow2 = (size_t(1) << 26))
        : cap(capacity_pow2), mask_(capacity_pow2 - 1),
          hkey(capacity_pow2, kEmpty), hval(capacity_pow2, 0.0), hpol(capacity_pow2, 0) {}

    static uint64_t mix(uint64_t x) {
        x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33; return x;
    }
    static uint64_t pack(uint16_t nm, uint16_t mm, int sch, int scn, uint32_t extra) {
        return (uint64_t)nm | ((uint64_t)mm << 13) | ((uint64_t)sch << 19)
             | ((uint64_t)scn << 20) | ((uint64_t)extra << 22);
    }
    static int      extraOf(uint32_t e, int v)         { return (e >> (2 * v)) & 3u; }
    static uint32_t setExtra(uint32_t e, int v, int c) {
        return (e & ~(3u << (2 * v))) | ((uint32_t)c << (2 * v));
    }
    size_t slotOf(uint64_t key) const {
        size_t i = mix(key) & mask_;
        while (hkey[i] != kEmpty && hkey[i] != key) i = (i + 1) & mask_;
        return i;
    }

    double solve(uint16_t nm, uint16_t mm, int sch, int scn, uint32_t extra) {
        const uint64_t key = pack(nm, mm, sch, scn, extra);
        {
            const size_t i = slotOf(key);
            if (hkey[i] == key) return hval[i];
        }
        const int pcn = maskPop(nm);
        double val;
        uint8_t pol = 0;
        if (pcn == kFlip7Target) {
            val = (double)fullScore(nm, mm);
        } else {
            int E = 0;
            for (int v = 2; v <= 12; ++v) E += extraOf(extra, v);
            const int T = kDeckTotal - (pcn + E + maskPop(mm) + scn);
            double ev_hit = 0.0;
            for (int v = 0; v < kNumValues; ++v) {
                const uint16_t bit = (uint16_t)(1u << v);
                const int r = numberCount(v) - (int)((nm >> v) & 1) - extraOf(extra, v);
                if (nm & bit) {                                  // held -> duplicate risk
                    if (r > 0 && sch) {                          // save: remove duplicate
                        const double p = (double)r / (double)T;
                        ev_hit += p * solve(nm, mm, 0, scn, setExtra(extra, v, extraOf(extra, v) + 1));
                    }  // r>0 && !sch -> bust (0); r==0 -> impossible draw
                } else {
                    const double p = (double)r / (double)T;      // r == count(v)
                    const uint16_t nn = (uint16_t)(nm | bit);
                    if (maskPop(nn) == kFlip7Target) ev_hit += p * (double)fullScore(nn, mm);
                    else                             ev_hit += p * solve(nn, mm, sch, scn, extra);
                }
            }
            for (int i = 0; i < kNumModifiers; ++i) {
                const uint16_t bit = (uint16_t)(1u << i);
                if (mm & bit) continue;
                ev_hit += (1.0 / (double)T) * solve(nm, (uint16_t)(mm | bit), sch, scn, extra);
            }
            if (scn < kNumSecondChance) {
                const double p = (double)(kNumSecondChance - scn) / (double)T;
                ev_hit += p * solve(nm, mm, 1, scn + 1, extra);
            }
            const double ev_stay = (double)fullScore(nm, mm);
            if (ev_hit > ev_stay) { val = ev_hit;  pol = 1; }
            else                  { val = ev_stay; pol = 0; }
        }
        const size_t i = slotOf(key);  // re-find: recursion inserted other keys
        hkey[i] = key;
        hval[i] = val;
        hpol[i] = pol;
        ++states_evaluated;
        return val;
    }

    double optimal() { return solve(0, 0, 0, 0, 0); }

    uint8_t policy(uint16_t nm, uint16_t mm, int sch, int scn, uint32_t extra) const {
        const size_t i = slotOf(pack(nm, mm, sch, scn, extra));
        return (hkey[i] != kEmpty) ? hpol[i] : 0;
    }
    double load_factor() const { return (double)states_evaluated / (double)cap; }
};

}  // namespace flip7
