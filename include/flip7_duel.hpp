// flip7_duel.hpp — the full 94-card, N-player, shared-deck game with ORGANIC
// action cards and adversarial targeting (the real rules). Chapter 5 Part C uses
// the 2-player case; the same engine runs an arbitrary number of players.
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
//                   one (the next such player in turn order), else discarded.
//   * Flip 7 ends the round instantly for everyone.
//
// Players make Hit/Stay decisions with the numbers+modifier expected-score-optimal
// policy (SolitaireModDP, built instantly); the variable under study is the
// TARGETING policy. We measure how much adversarial targeting is worth by win rate
// vs. a field of naive (use-it-on-yourself) / random opponents, with symmetric
// matchups as sanities (a symmetric field gives each player 1/n).
//
// Targeting among many opponents is "aim at the leader": Freeze the active
// opponent closest to winning the match (denying EV(S)-sum(S)); Flip Three the
// deepest active opponent (popcount >= k_att, most bust-prone), preferring the
// match leader on ties; keep it on yourself if no opponent is deep enough.
//
// Modeling note: a fresh shuffled deck per round (not the continuous shoe). The
// shoe-vs-fresh discrepancy is a separate, already-small effect (Chapter 4); the
// targeting question studied here is independent of deck cycling.
#pragma once
#include "flip7_core.hpp"
#include "flip7_dp.hpp"
#include "flip7_rng.hpp"

#include <algorithm>
#include <cstdint>
#include <thread>
#include <vector>

namespace flip7 {

// Card codes in the physical deck.
enum : uint8_t { CARD_SC = 19, CARD_FLIP3 = 20, CARD_FREEZE = 21 };
// Player round status.
enum { ST_ACTIVE = 0, ST_STAYED = 1, ST_BUSTED = 2, ST_FLIP7 = 3 };
// Targeting policies.
enum { TP_SELF = 0, TP_RANDOM = 1, TP_ADVERSARIAL = 2 };

struct DuelStats {
    long games = 0;
    double p0_score = 0;               // player 0 wins + tie credit (1/k on a top tie)
    long rounds = 0;
    long freezes = 0, freeze_at_opp = 0;
    long flip3s = 0,  flip3_at_opp = 0;
};

struct Duel {
    const SolitaireModDP& pol;   // numbers+modifier optimal Hit/Stay
    Xoshiro256pp&         rng;
    int                   n;     // number of players (>= 2)
    std::vector<int>      tp;    // targeting policy per player
    int                   k_att; // adversarial: Flip-Three the opponent only when popcount >= k_att

    uint8_t deck[SolitaireAllDP::kDeckTotal];  // 94
    int     pos = 0;

    struct PState { uint16_t nm, mm; int sch; int status; };
    std::vector<PState> P;        // per-player round state, size n
    std::vector<long>   total;    // running match totals, size n (for leader-aware targeting)
    bool round_over = false;      // a Flip 7 ended the round
    DuelStats* st = nullptr;

    Duel(const SolitaireModDP& p, Xoshiro256pp& r, const std::vector<int>& tps, int katt, DuelStats* s)
        : pol(p), rng(r), n((int)tps.size()), tp(tps), k_att(katt),
          P(tps.size()), total(tps.size(), 0), st(s) {}

    // 2-player convenience ctor (preserves the original Duel(p, r, tp0, tp1, ...) shape).
    Duel(const SolitaireModDP& p, Xoshiro256pp& r, int tp0, int tp1, int katt, DuelStats* s)
        : Duel(p, r, std::vector<int>{tp0, tp1}, katt, s) {}

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
        if (tp[c] == TP_SELF) return c;
        // 2-player fast path: identical behavior AND identical RNG stream to the
        // original Duel, so Chapter 5 Part C reproduces bit-for-bit.
        if (n == 2) {
            const int o = c ^ 1;
            if (P[o].status != ST_ACTIVE) return c;       // must assign to an active player
            if (tp[c] == TP_RANDOM) return (rng.next() & 1) ? o : c;
            if (is_freeze) return o;                      // cap the opponent (denies EV(S)-sum(S))
            return (popc(o) >= k_att) ? o : c;            // Flip Three: attack a DEEP opp, else keep it
        }
        // n >= 3.
        if (tp[c] == TP_RANDOM) {                          // uniform over active players (incl self)
            int act = 0;
            for (int i = 0; i < n; ++i) act += (P[i].status == ST_ACTIVE);
            int pick = (int)rng.bounded((uint64_t)act);
            for (int i = 0; i < n; ++i)
                if (P[i].status == ST_ACTIVE && pick-- == 0) return i;
            return c;
        }
        // adversarial: aim at the leader among active opponents.
        int best = -1;
        for (int o = 0; o < n; ++o) {
            if (o == c || P[o].status != ST_ACTIVE) continue;
            if (is_freeze) {                               // cap the match leader (tie: deepest hand)
                if (best < 0 || total[o] > total[best] ||
                    (total[o] == total[best] && popc(o) > popc(best))) best = o;
            } else {                                       // Flip Three: deepest qualifying opp (tie: leader)
                if (popc(o) < k_att) continue;
                if (best < 0 || popc(o) > popc(best) ||
                    (popc(o) == popc(best) && total[o] > total[best])) best = o;
            }
        }
        return best < 0 ? c : best;                        // none deep enough -> keep it on self
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

    // Resolve a Second Chance gained by player p (keep it; else hand to the next
    // active player in turn order who lacks one; else discard).
    void resolve_sc(int p) {
        if (!P[p].sch) { P[p].sch = 1; return; }
        for (int d = 1; d < n; ++d) {
            const int o = (p + d) % n;
            if (P[o].status == ST_ACTIVE && !P[o].sch) { P[o].sch = 1; return; }
        }
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

    // Play one round from a fresh deck; add banked scores to total[].
    void play_round(int starter) {
        shuffle_deck();
        round_over = false;
        for (int i = 0; i < n; ++i) P[i] = {0, 0, 0, ST_ACTIVE};
        const int guard_max = 200 * n;                      // n==2 -> 400 (matches the original)
        for (int guard = 0; guard < guard_max; ++guard) {
            const int i = (starter + guard) % n;
            bool any_active = false;
            for (int k = 0; k < n; ++k) any_active |= (P[k].status == ST_ACTIVE);
            if (!any_active) break;
            take_turn(i);
            if (round_over) break;
        }
        for (int i = 0; i < n; ++i)
            total[i] += (P[i].status == ST_BUSTED) ? 0 : (long)fullScore(P[i].nm, P[i].mm);
        if (st) st->rounds++;
    }

    // Play a first-to-`target` match; returns the winning player index, or -1 if
    // the round cap is hit with no decision. A tie at/above target is played on.
    int play_game(int target) {
        for (int i = 0; i < n; ++i) total[i] = 0;
        int starter = 0;
        for (int round = 0; round < 1000; ++round) {
            play_round(starter);
            starter = (starter + 1) % n;                    // rotate the dealer
            long mx = -1; for (int i = 0; i < n; ++i) mx = std::max(mx, total[i]);
            if (mx >= target) {
                int winner = -1, ties = 0;
                for (int i = 0; i < n; ++i) if (total[i] == mx) { ties++; winner = i; }
                if (ties == 1) return winner;
                // tie at/above target: rules say play on until someone leads.
            }
        }
        return -1;
    }
};

// Run G N-player games with the given per-player targeting policies; player 0 gets
// win credit (and 1/n on a no-decision). Symmetric fields give each player 1/n.
//
// Each game is seeded independently from `seed` via splitmix64(seed+g), so the
// result is deterministic and INDEPENDENT of the thread count -- games are split
// across hardware threads with per-thread Duel/RNG and the stats reduced.
inline DuelStats run_tournament(const SolitaireModDP& pol, const std::vector<int>& tps,
                                int k_att, int target, uint64_t G, uint64_t seed) {
    const int n = (int)tps.size();
    auto work = [&](uint64_t g0, uint64_t g1, DuelStats& out) {
        Xoshiro256pp rng;
        Duel d(pol, rng, tps, k_att, &out);
        for (uint64_t g = g0; g < g1; ++g) {
            uint64_t sm = seed + g;
            rng.seed(splitmix64(sm));                          // per-game independent stream
            const int r = d.play_game(target);
            out.games++;
            if (r == 0)     out.p0_score += 1.0;
            else if (r < 0) out.p0_score += 1.0 / (double)n;   // no decision: split evenly
        }
    };
    unsigned T = std::thread::hardware_concurrency();
    if (!T) T = 1;
    if (T == 1 || G < 8192) { DuelStats s; work(0, G, s); return s; }
    std::vector<DuelStats> parts(T);
    std::vector<std::thread> th;
    const uint64_t chunk = (G + T - 1) / T;
    for (unsigned t = 0; t < T; ++t) {
        const uint64_t g0 = (uint64_t)t * chunk, g1 = std::min(G, g0 + chunk);
        if (g0 >= g1) break;
        th.emplace_back([&, t, g0, g1] { work(g0, g1, parts[t]); });
    }
    for (auto& x : th) x.join();
    DuelStats s;
    for (const auto& p : parts) {
        s.games += p.games; s.p0_score += p.p0_score; s.rounds += p.rounds;
        s.freezes += p.freezes; s.freeze_at_opp += p.freeze_at_opp;
        s.flip3s += p.flip3s;   s.flip3_at_opp += p.flip3_at_opp;
    }
    return s;
}

// 2-player convenience wrapper (preserves Chapter 5 Part C's API and numbers).
inline DuelStats run_duel(const SolitaireModDP& pol, int tp0, int tp1, int k_att,
                          int target, uint64_t G, uint64_t seed) {
    return run_tournament(pol, std::vector<int>{tp0, tp1}, k_att, target, G, seed);
}

}  // namespace flip7
