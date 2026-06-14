// flip7_duel.hpp — Chapter 5 Part C: the full 94-card, 2-player, shared-deck game
// with ORGANIC action cards and adversarial targeting (the real rules).
//
// This is the faithful-rules Monte-Carlo ground truth: a physical shuffled 94-card
// deck per round (so duplicate / save mechanics are exact with no bookkeeping --
// the deck array is the truth), real turn order, and the real action semantics:
//   * Freeze     -> the chosen active player Stays immediately (banks; cannot be
//                   blocked by Second Chance).
//   * Flip Three -> the chosen player flips 3 cards one at a time, ending early on
//                   bust or Flip 7. Action cards revealed during the three are set
//                   aside and resolved AFTER (re-targetable by the flipper) -- so a
//                   Second Chance revealed mid-sequence does NOT save a bust in that
//                   same sequence.
//   * Second Chance -> kept; a second copy is handed to an active player who lacks
//                   one, else discarded; saves the first duplicate bust.
//   * Flip 7 ends the round instantly for everyone.
//
// Players make Hit/Stay decisions with the numbers+modifier expected-score-optimal
// policy (SolitaireModDP, built instantly); the variable under study is the
// TARGETING policy. We measure how much adversarial targeting is worth by win rate
// vs. a naive (use-it-on-yourself) opponent, with symmetric matchups as sanities.
//
// Modeling note: a fresh shuffled deck per round (not the continuous shoe). The
// shoe-vs-fresh discrepancy is a separate, already-small effect (Chapter 4); the
// targeting question studied here is independent of deck cycling.
#pragma once
#include "flip7_core.hpp"
#include "flip7_dp.hpp"
#include "flip7_rng.hpp"

#include <cstdint>

namespace flip7 {

// Card codes in the physical deck.
enum : uint8_t { CARD_SC = 19, CARD_FLIP3 = 20, CARD_FREEZE = 21 };
// Player round status.
enum { ST_ACTIVE = 0, ST_STAYED = 1, ST_BUSTED = 2, ST_FLIP7 = 3 };
// Targeting policies.
enum { TP_SELF = 0, TP_RANDOM = 1, TP_ADVERSARIAL = 2 };

struct DuelStats {
    long games = 0;
    double p0_score = 0;               // p0 wins + 0.5 * ties
    long rounds = 0;
    long freezes = 0, freeze_at_opp = 0;
    long flip3s = 0,  flip3_at_opp = 0;
};

struct Duel {
    const SolitaireModDP& pol;   // numbers+modifier optimal Hit/Stay
    Xoshiro256pp&         rng;
    int                   tp[2]; // targeting policy per player
    int                   k_att; // adversarial: Flip-Three the opponent only when popcount >= k_att

    uint8_t deck[SolitaireAllDP::kDeckTotal];  // 94
    int     pos = 0;

    struct PState { uint16_t nm, mm; int sch; int status; } P[2];
    bool round_over = false;     // a Flip 7 ended the round
    DuelStats* st = nullptr;

    Duel(const SolitaireModDP& p, Xoshiro256pp& r, int tp0, int tp1, int katt, DuelStats* s)
        : pol(p), rng(r), k_att(katt), st(s) { tp[0] = tp0; tp[1] = tp1; }

    int popc(int i) const { return maskPop(P[i].nm); }

    void shuffle_deck() {
        int idx = 0;
        for (int v = 0; v < kNumValues; ++v)
            for (int k = 0, c = numberCount(v); k < c; ++k) deck[idx++] = (uint8_t)v;
        for (int i = 0; i < kNumModifiers; ++i)    deck[idx++] = (uint8_t)(13 + i);
        for (int i = 0; i < kNumSecondChance; ++i) deck[idx++] = CARD_SC;
        for (int i = 0; i < 3; ++i)                deck[idx++] = CARD_FLIP3;
        for (int i = 0; i < 3; ++i)                deck[idx++] = CARD_FREEZE;
        // Fisher-Yates over the whole 94 (one round never exhausts it).
        for (int i = SolitaireAllDP::kDeckTotal - 1; i > 0; --i) {
            const int j = (int)rng.bounded((uint64_t)(i + 1));
            const uint8_t t = deck[i]; deck[i] = deck[j]; deck[j] = t;
        }
        pos = 0;
    }

    bool wants_hit(int i) const { return pol.hit[SolitaireModDP::enc(P[i].nm, P[i].mm)] != 0; }

    // Choose the target of an action card the chooser `c` must assign.
    int choose_target(int c, bool is_freeze) {
        const int o = c ^ 1;
        const bool opp_active = (P[o].status == ST_ACTIVE);
        if (tp[c] == TP_SELF) return c;           // naive: always use it on yourself
        if (!opp_active) return c;                // must assign to an active player
        if (tp[c] == TP_RANDOM) return (rng.next() & 1) ? o : c;
        if (is_freeze) return o;                  // cap the opponent (denies EV(S)-sum(S))
        return (popc(o) >= k_att) ? o : c;        // Flip Three: attack a DEEP opp, else keep it
    }

    // Resolve a plain number/modifier card for player p (sets bust / Flip 7).
    void resolve_num_mod(int p, uint8_t card) {
        if (card < 13) {
            const uint16_t bit = (uint16_t)(1u << card);
            if (P[p].nm & bit) {                  // duplicate
                if (P[p].sch) P[p].sch = 0;       // Second Chance saves (duplicate discarded)
                else          P[p].status = ST_BUSTED;
            } else {
                P[p].nm |= bit;
                if (maskPop(P[p].nm) == kFlip7Target) { P[p].status = ST_FLIP7; round_over = true; }
            }
        } else {                                  // modifier 13..18
            P[p].mm |= (uint16_t)(1u << (card - 13));
        }
    }

    // Resolve a Second Chance gained by player p (keep, else hand to an active
    // player without one, else discard).
    void resolve_sc(int p) {
        if (!P[p].sch) { P[p].sch = 1; return; }
        const int o = p ^ 1;
        if (P[o].status == ST_ACTIVE && !P[o].sch) P[o].sch = 1;   // else discarded
    }

    // Resolve an action card (Flip Three / Freeze) that chooser `c` must assign.
    void resolve_action(int c, uint8_t card) {
        const bool is_freeze = (card == CARD_FREEZE);
        const int t = choose_target(c, is_freeze);
        if (st) {
            if (is_freeze) { st->freezes++; if (t != c) st->freeze_at_opp++; }
            else           { st->flip3s++;  if (t != c) st->flip3_at_opp++; }
        }
        if (is_freeze) { if (P[t].status == ST_ACTIVE) P[t].status = ST_STAYED; }
        else           do_flip_three(t);
    }

    // The target flips up to 3 cards; revealed action cards are set aside and
    // resolved (re-targeted by the flipper) after the sequence.
    void do_flip_three(int t) {
        if (P[t].status != ST_ACTIVE) return;
        uint8_t pend[8]; int np = 0;
        for (int d = 0; d < 3 && P[t].status == ST_ACTIVE && pos < SolitaireAllDP::kDeckTotal; ++d) {
            const uint8_t card = deck[pos++];
            if (card < CARD_SC) resolve_num_mod(t, card);
            else                pend[np++] = card;          // SC / Flip3 / Freeze: set aside
            if (round_over) return;
        }
        for (int i = 0; i < np; ++i) {                      // resolve set-aside, flipper assigns
            if (pend[i] == CARD_SC) resolve_sc(t);
            else                    resolve_action(t, pend[i]);
            if (round_over) return;
        }
    }

    // One player's turn: Stay, or draw a card and resolve it.
    void take_turn(int i) {
        if (P[i].status != ST_ACTIVE) return;
        if (!wants_hit(i)) { P[i].status = ST_STAYED; return; }
        if (pos >= SolitaireAllDP::kDeckTotal) { P[i].status = ST_STAYED; return; }
        const uint8_t card = deck[pos++];
        if      (card < CARD_SC)     resolve_num_mod(i, card);
        else if (card == CARD_SC)    resolve_sc(i);
        else                         resolve_action(i, card);    // assigned immediately on a free turn
    }

    // Play one round from a fresh deck; add banked scores to total[2].
    void play_round(long* total, int starter) {
        shuffle_deck();
        round_over = false;
        for (int i = 0; i < 2; ++i) P[i] = {0, 0, 0, ST_ACTIVE};
        for (int guard = 0; guard < 400; ++guard) {
            const int i = (starter + guard) & 1;
            if (P[0].status != ST_ACTIVE && P[1].status != ST_ACTIVE) break;
            take_turn(i);
            if (round_over) break;
        }
        for (int i = 0; i < 2; ++i)
            total[i] += (P[i].status == ST_BUSTED) ? 0 : (long)fullScore(P[i].nm, P[i].mm);
        if (st) st->rounds++;
    }

    // Play a first-to-`target` match; returns 1 if p0 wins, 0 tie, -1 p1 wins.
    int play_game(int target) {
        long total[2] = {0, 0};
        int starter = 0;
        for (int round = 0; round < 1000; ++round) {
            play_round(total, starter);
            starter ^= 1;                          // rotate the dealer
            if (total[0] >= target || total[1] >= target) {
                if (total[0] > total[1]) return 1;
                if (total[1] > total[0]) return -1;
                // tie at/above target: rules say play on until someone leads
            }
        }
        return 0;
    }
};

// Run G games of (p0 policy tp0) vs (p1 policy tp1); p0 gets win + 0.5*tie credit.
inline DuelStats run_duel(const SolitaireModDP& pol, int tp0, int tp1, int k_att,
                          int target, uint64_t G, uint64_t seed) {
    Xoshiro256pp rng; rng.seed(seed);
    DuelStats s;
    Duel d(pol, rng, tp0, tp1, k_att, &s);
    for (uint64_t g = 0; g < G; ++g) {
        const int r = d.play_game(target);
        s.games++;
        if (r == 1) s.p0_score += 1.0;
        else if (r == 0) s.p0_score += 0.5;
    }
    return s;
}

}  // namespace flip7
