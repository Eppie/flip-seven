// flip7_vengeance.hpp -- faithful-rules Monte-Carlo simulator for
// **Flip 7: With a Vengeance** (the standalone sequel; see flip7_vengeance_rules.md).
//
// The exact dynamic programming that solves the original game does NOT transfer:
// Lucky 13 lets a hand hold two 13s (the unique-set mask breaks), and Steal / Swap
// / Discard / Unlucky 7 move cards between players (so "your hand determines the
// deck" breaks). So this is a Monte-Carlo engine -- a physical shuffled 108-card
// deck per round and the real rules -- the same approach as flip7_duel.hpp, which
// is the project's ground truth for the messy multiplayer cases.
//
// Modeling choices (documented, not hidden):
//   * Hit/Stay is a transparent one-step expected-value rule over the ACTUAL
//     remaining deck (count-aware), not a claimed optimum -- Vengeance is unsolved.
//   * The take-that targeting (TP_ADVERSARIAL) uses sensible heuristics (cap/curse
//     the leader, Flip-Four the deepest, Swap to bust an opponent). A symmetric
//     field of any fixed policy gives each player 1/n (the sanity check), so the
//     engine itself is unbiased; the heuristics only affect the *edge* numbers.
//   * Players start a round empty (the one face-up deal is folded into normal
//     draws), matching flip7_duel.hpp.
#pragma once
#include "flip7_core.hpp"
#include "flip7_rng.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

// Optional logical-cost instrumentation (compiled out unless FLIP7_VENG_INSTR).
// Counts draws / decisions / deck-scan work / action frequencies / busts so the
// hotspots can be attributed without a PMU (see perf/profile_vengeance.cpp). Like
// FLIP7_BLK_INSTR, it perturbs timing, so it lives in a SEPARATE build from the
// PMU run.
#ifdef FLIP7_VENG_INSTR
#define VINSTR(x) do { x; } while (0)
#else
#define VINSTR(x) do {} while (0)
#endif

namespace flip7 {

// ---- card encoding ----------------------------------------------------------
// Regular number value v in 1..13 is its own code v. The rest:
enum : uint8_t {
    VC_ZERO     = 0,    // The Zero (counts as a number, value 0)
    VC_UNLUCKY7 = 14,   // Unlucky 7
    VC_LUCKY13  = 15,   // Lucky 13
    VC_M2 = 16, VC_M4, VC_M6, VC_M8, VC_M10, VC_DIV2,   // 16..21 modifiers
    VC_JUSTONE = 22, VC_FLIPFOUR, VC_SWAP, VC_STEAL, VC_DISCARD  // 22..26 actions
};
inline bool vc_is_number(uint8_t c) { return c <= 13 || c == VC_ZERO || c == VC_UNLUCKY7 || c == VC_LUCKY13; }
inline bool vc_is_modifier(uint8_t c) { return c >= VC_M2 && c <= VC_DIV2; }
inline bool vc_is_action(uint8_t c) { return c >= VC_JUSTONE; }

// modifier-mask bits (kept per player)
enum { MM_M2 = 0, MM_M4, MM_M6, MM_M8, MM_M10, MM_DIV2 };
inline int modmask_neg(uint8_t mm) {
    int s = 0;
    if (mm & (1u << MM_M2)) s += 2;   if (mm & (1u << MM_M4)) s += 4;
    if (mm & (1u << MM_M6)) s += 6;   if (mm & (1u << MM_M8)) s += 8;
    if (mm & (1u << MM_M10)) s += 10;
    return s;
}
inline int mm_bit_of(uint8_t card) {
    switch (card) {
        case VC_M2: return MM_M2; case VC_M4: return MM_M4; case VC_M6: return MM_M6;
        case VC_M8: return MM_M8; case VC_M10: return MM_M10; default: return MM_DIV2;
    }
}

enum { VST_ACTIVE = 0, VST_STAYED = 1, VST_BUSTED = 2, VST_FLIP7 = 3 };
enum { VTP_SELF = 0, VTP_RANDOM = 1, VTP_ADVERSARIAL = 2 };

struct VPlayer {
    uint8_t cnt[14] = {0};   // count of each number value 0..13 held (0 = The Zero)
    bool zero = false, lucky13 = false, unlucky7 = false;
    uint8_t modmask = 0;
    int status = VST_ACTIVE;

    int ncards() const { int s = 0; for (int v = 0; v <= 13; ++v) s += cnt[v]; return s; }
    int numsum() const { int s = 0; for (int v = 0; v <= 13; ++v) s += v * cnt[v]; return s; }
    int allowed(int v) const { return (v == 13 && lucky13) ? 2 : 1; }
    void reset() { *this = VPlayer{}; }
};

// Round score for a non-busted player (Vengeance scoring order).
inline int vengeance_score(const VPlayer& p) {
    if (p.status == VST_BUSTED) return 0;
    int s = p.numsum();
    if (p.modmask & (1u << MM_DIV2)) s /= 2;          // halve number total, rounded down
    s -= modmask_neg(p.modmask);                       // subtract negatives
    if (s < 0) s = 0;
    if (p.status == VST_FLIP7) return s + kFlip7Bonus; // Flip 7 lifts The Zero and adds +15
    if (p.zero) return 0;                              // The Zero: 0 unless you flipped 7
    return s;
}

struct VengeanceStats {
    long games = 0;
    double p0_score = 0;     // player 0 wins + 1/n on a no-decision
    long rounds = 0;
};

struct VengeanceGame {
    Xoshiro256pp& rng;
    int n;
    std::vector<int> tp;            // targeting policy per player
    std::vector<VPlayer> P;
    std::vector<long> total;
    std::vector<uint8_t> deck;
    int pos = 0;
    int rem[14] = {0};      // remaining REGULAR number cards of each value 1..13 in the draw pile
    bool round_over = false;

#ifdef FLIP7_VENG_INSTR
    // per-run logical counters (mutable so the const wants_hit can tally scans)
    mutable long ic_games = 0, ic_rounds = 0, ic_draws = 0, ic_decisions = 0, ic_scan = 0;
    mutable long ic_busts = 0, ic_flip7 = 0, ic_swap_evals = 0, ic_act[5] = {0};
    void instr_dump() const {
        const double G = ic_games ? (double)ic_games : 1.0;
        std::printf("  logical / game (n=%d):  rounds=%.2f  draws=%.1f  hit-decisions=%.1f\n",
                    n, ic_rounds / G, ic_draws / G, ic_decisions / G);
        std::printf("  wants_hit deck-scan:    %.0f card-reads/game  (%.1f per decision)\n",
                    ic_scan / G, ic_decisions ? (double)ic_scan / ic_decisions : 0.0);
        std::printf("  busts=%.2f/game  flip7=%.3f/game  swap-evals=%.1f/game\n",
                    ic_busts / G, ic_flip7 / G, ic_swap_evals / G);
        std::printf("  actions/game: JustOne=%.2f FlipFour=%.2f Swap=%.2f Steal=%.2f Discard=%.2f\n",
                    ic_act[0] / G, ic_act[1] / G, ic_act[2] / G, ic_act[3] / G, ic_act[4] / G);
    }
#endif

    VengeanceGame(Xoshiro256pp& r, const std::vector<int>& tps)
        : rng(r), n((int)tps.size()), tp(tps), P(tps.size()), total(tps.size(), 0) {}

    // --- deck: 108 cards ---
    void build_deck() {
        deck.clear();
        deck.push_back(VC_ZERO);                              // one Zero
        for (int v = 1; v <= 13; ++v) {                       // count(v)=v, minus the special copy of 7 and 13
            int copies = v;
            if (v == 7 || v == 13) copies -= 1;
            for (int k = 0; k < copies; ++k) deck.push_back((uint8_t)v);
        }
        deck.push_back(VC_UNLUCKY7);
        deck.push_back(VC_LUCKY13);
        for (uint8_t m = VC_M2; m <= VC_DIV2; ++m) deck.push_back(m);          // one each (6)
        for (uint8_t a = VC_JUSTONE; a <= VC_DISCARD; ++a) { deck.push_back(a); deck.push_back(a); }  // x2 (10)
    }
    void shuffle() {
        build_deck();
        for (int i = (int)deck.size() - 1; i > 0; --i) {
            int j = (int)rng.bounded((uint64_t)(i + 1));
            std::swap(deck[i], deck[j]);
        }
        pos = 0;
        for (int v = 1; v <= 13; ++v) rem[v] = (v == 7) ? 6 : (v == 13) ? 12 : v;  // regular-number counts
    }
    int draw_card() {
        VINSTR(++ic_draws);
        if (pos >= (int)deck.size()) return -1;
        uint8_t c = deck[pos++];
        if (c >= 1 && c <= 13) --rem[c];          // keep the remaining bust-card tally current
        return c;
    }

    bool active(int i) const { return P[i].status == VST_ACTIVE; }
    int  leader_other(int self) const {                       // highest-total non-busted opponent
        int best = -1;
        for (int o = 0; o < n; ++o) {
            if (o == self || P[o].status == VST_BUSTED) continue;
            if (best < 0 || total[o] > total[best]) best = o;
        }
        return best;
    }

    // --- resolve a number/special-number card landing on player p ---
    void give_number(int p, uint8_t card) {
        VPlayer& q = P[p];
        if (card == VC_ZERO) { if (q.cnt[0] == 0) { q.cnt[0] = 1; q.zero = true; } }
        else if (card == VC_UNLUCKY7) {                       // discard everything else, keep only this 7
            uint8_t keepmod = 0; (void)keepmod;
            q.reset();                                         // clears numbers + modifiers + flags
            q.cnt[7] = 1; q.unlucky7 = true;
        } else if (card == VC_LUCKY13) {
            q.lucky13 = true;
            if (q.cnt[13] + 1 > q.allowed(13)) { q.status = VST_BUSTED; return; }
            q.cnt[13]++;
        } else {                                              // regular number 1..13
            int v = card;
            if (q.cnt[v] + 1 > q.allowed(v)) { q.status = VST_BUSTED; VINSTR(++ic_busts); return; }
            q.cnt[v]++;
        }
        if (q.status != VST_BUSTED && q.ncards() == kFlip7Target) { q.status = VST_FLIP7; round_over = true; VINSTR(++ic_flip7); }
    }

    // re-check a player for a duplicate bust after a Swap/Steal moved cards in.
    void recheck_bust(int p) {
        VPlayer& q = P[p];
        if (q.status == VST_BUSTED) return;
        for (int v = 0; v <= 13; ++v) if (q.cnt[v] > q.allowed(v)) { q.status = VST_BUSTED; return; }
    }

    // --- targeting helpers (policy-dependent) ---
    int pick_modifier_target(int chooser) {                   // negatives/÷2: dump on an opponent
        if (tp[chooser] == VTP_SELF) return chooser;
        int L = leader_other(chooser);
        if (tp[chooser] == VTP_RANDOM) {
            // a random non-busted player (incl self)
            int act = 0; for (int i = 0; i < n; ++i) act += (P[i].status != VST_BUSTED);
            int pick = (int)rng.bounded((uint64_t)std::max(1, act));
            for (int i = 0; i < n; ++i) if (P[i].status != VST_BUSTED && pick-- == 0) return i;
            return chooser;
        }
        return L < 0 ? chooser : L;                           // adversarial: curse the leader
    }

    void resolve_modifier(int chooser, uint8_t card) {
        int t = pick_modifier_target(chooser);
        P[t].modmask |= (uint8_t)(1u << mm_bit_of(card));
    }

    // forced single draw (Just One More on target), then stay.
    void forced_one_then_stay(int t) {
        if (P[t].status != VST_ACTIVE) return;
        int c = draw_card();
        if (c >= 0) resolve_drawn(t, c, /*chooser_for_actions=*/t);
        if (P[t].status == VST_ACTIVE) P[t].status = VST_STAYED;
    }
    // forced up-to-4 draws (Flip Four on target).
    void forced_flip_four(int t) {
        if (P[t].status != VST_ACTIVE) return;
        for (int k = 0; k < 4 && P[t].status == VST_ACTIVE; ++k) {
            int c = draw_card(); if (c < 0) break;
            resolve_drawn(t, c, /*chooser_for_actions=*/t);
            if (round_over) return;
        }
    }

    // Swap: among face-up number cards of two different non-busted players, choose
    // the swap that does the most damage to the chooser's opponents (busts > score
    // loss), least self-harm. Returns false if no legal swap exists.
    void best_swap(int chooser) {
        // collect (player, value) of every face-up regular-number card we can move
        struct Slot { int p, v; };
        std::vector<Slot> slots;
        for (int p = 0; p < n; ++p) {
            if (P[p].status == VST_BUSTED) continue;
            for (int v = 0; v <= 13; ++v) for (int k = 0; k < P[p].cnt[v]; ++k) slots.push_back({p, v});
        }
        if (tp[chooser] == VTP_SELF) return;                  // naive: decline take-that swaps
        if (tp[chooser] == VTP_RANDOM) {                      // symmetric: a random legal swap
            std::vector<std::pair<int, int>> pairs;
            for (size_t i = 0; i < slots.size(); ++i) for (size_t j = i + 1; j < slots.size(); ++j)
                if (slots[i].p != slots[j].p) pairs.push_back({(int)i, (int)j});
            if (pairs.empty()) return;
            auto [i, j] = pairs[(int)rng.bounded((uint64_t)pairs.size())];
            int pa = slots[i].p, va = slots[i].v, pb = slots[j].p, vb = slots[j].v;
            P[pa].cnt[va]--; P[pa].cnt[vb]++; P[pb].cnt[vb]--; P[pb].cnt[va]++;
            recheck_bust(pa); recheck_bust(pb);
            return;
        }
        auto eval = [&](int pa, int va, int pb, int vb) -> double {
            // simulate moving va from pa to pb and vb from pb to pa; score = damage to
            // opponents minus harm to chooser. Bust is worth a lot.
            VPlayer A = P[pa], B = P[pb];
            A.cnt[va]--; A.cnt[vb]++; B.cnt[vb]--; B.cnt[va]++;
            auto busts = [&](const VPlayer& q) { for (int v = 0; v <= 13; ++v) if (q.cnt[v] > q.allowed(v)) return true; return false; };
            auto val = [&](int who, const VPlayer& before, const VPlayer& after) {
                bool bb = busts(after);
                double d = (bb ? 200.0 : (double)(/*after score proxy*/ after.numsum() - before.numsum()));
                // damage to opponents is good (+), to self/teammates bad
                return (who == chooser) ? -d : d;
            };
            double s = 0;
            s += val(pa, P[pa], A);
            s += val(pb, P[pb], B);
            return s;
        };
        VINSTR(ic_swap_evals += (long)(slots.size() * slots.size()));
        double best = 0.0; int bpa = -1, bva = 0, bpb = -1, bvb = 0; bool found = false;
        for (size_t i = 0; i < slots.size(); ++i)
            for (size_t j = 0; j < slots.size(); ++j) {
                if (slots[i].p == slots[j].p) continue;
                double s = eval(slots[i].p, slots[i].v, slots[j].p, slots[j].v);
                if (!found || s > best) { found = true; best = s; bpa = slots[i].p; bva = slots[i].v; bpb = slots[j].p; bvb = slots[j].v; }
            }
        if (!found || best <= 0.0) return;                    // no profitable swap -> decline (no harmless-but-mandatory model)
        P[bpa].cnt[bva]--; P[bpa].cnt[bvb]++; P[bpb].cnt[bvb]--; P[bpb].cnt[bva]++;
        recheck_bust(bpa); recheck_bust(bpb);
    }

    void best_steal(int chooser) {                            // take leader's highest number we can hold
        if (tp[chooser] == VTP_SELF) return;                  // naive: decline
        int L = (tp[chooser] == VTP_ADVERSARIAL) ? leader_other(chooser) : -1;
        if (L < 0) {                                          // random: steal from a random opponent
            std::vector<int> opp; for (int o = 0; o < n; ++o) if (o != chooser && P[o].status != VST_BUSTED && P[o].ncards() > 0) opp.push_back(o);
            if (opp.empty()) return; L = opp[(int)rng.bounded((uint64_t)opp.size())];
        }
        int bestv = -1;
        for (int v = 13; v >= 0; --v) if (P[L].cnt[v] > 0 && P[chooser].cnt[v] + 1 <= P[chooser].allowed(v)) { bestv = v; break; }
        if (bestv < 0) {                                      // nothing safe -> take their highest anyway (denies them)
            for (int v = 13; v >= 0; --v) if (P[L].cnt[v] > 0) { bestv = v; break; }
            if (bestv < 0) return;
        }
        P[L].cnt[bestv]--; P[chooser].cnt[bestv]++;
        recheck_bust(chooser);
        if (P[chooser].status != VST_BUSTED && P[chooser].ncards() == kFlip7Target) { P[chooser].status = VST_FLIP7; round_over = true; }
    }

    void best_discard(int chooser) {                          // make leader drop their highest number
        if (tp[chooser] == VTP_SELF) return;                  // naive: decline (no self benefit)
        int L = (tp[chooser] == VTP_ADVERSARIAL) ? leader_other(chooser) : -1;
        if (L < 0) {
            std::vector<int> opp; for (int o = 0; o < n; ++o) if (o != chooser && P[o].status != VST_BUSTED && P[o].ncards() > 0) opp.push_back(o);
            if (opp.empty()) return; L = opp[(int)rng.bounded((uint64_t)opp.size())];
        }
        for (int v = 13; v >= 0; --v) if (P[L].cnt[v] > 0) { P[L].cnt[v]--; if (v == 13 && P[L].cnt[13] == 0) P[L].lucky13 = false; return; }
    }

    // resolve any drawn card for player p; chooser assigns action cards.
    void resolve_drawn(int p, uint8_t card, int chooser) {
        if (vc_is_number(card)) { give_number(p, card); return; }
        if (vc_is_modifier(card)) { resolve_modifier(p, card); return; }
        // action card: the drawing player p is the chooser (organic play)
        resolve_action(chooser, card);
    }

    void resolve_action(int chooser, uint8_t card) {
        VINSTR(++ic_act[card - VC_JUSTONE]);
        // who can be targeted? if no non-busted opponent, self where possible
        switch (card) {
            case VC_JUSTONE: {
                int t;
                if (tp[chooser] == VTP_SELF) t = chooser;
                else if (tp[chooser] == VTP_RANDOM) {                   // random active player (incl self)
                    std::vector<int> a; for (int o = 0; o < n; ++o) if (active(o)) a.push_back(o);
                    t = a.empty() ? chooser : a[(int)rng.bounded((uint64_t)a.size())];
                } else { t = leader_other(chooser); if (t < 0) t = chooser; }  // adversarial: cap the leader
                forced_one_then_stay(t);
                break;
            }
            case VC_FLIPFOUR: {
                int t = chooser;
                if (tp[chooser] == VTP_ADVERSARIAL) {          // the deepest active opponent busts most easily
                    int best = -1;
                    for (int o = 0; o < n; ++o) if (o != chooser && active(o) && (best < 0 || P[o].ncards() > P[best].ncards())) best = o;
                    if (best >= 0) t = best;
                } else if (tp[chooser] == VTP_RANDOM) {
                    std::vector<int> a; for (int o = 0; o < n; ++o) if (active(o)) a.push_back(o);
                    if (!a.empty()) t = a[(int)rng.bounded((uint64_t)a.size())];
                }
                forced_flip_four(t);
                break;
            }
            case VC_SWAP:    best_swap(chooser); break;
            case VC_STEAL:   best_steal(chooser); break;
            case VC_DISCARD: best_discard(chooser); break;
        }
    }

    // --- Hit/Stay: transparent one-step EV over the actual remaining deck ---
    bool wants_hit(int i) const {
        VINSTR(++ic_decisions);
        const VPlayer& q = P[i];
        if (q.zero) return true;                               // The Zero forces hitting
        const int nc = q.ncards();
        if (nc == 0) return true;
        // bust cards remaining = regular copies of values held at their allowed limit,
        // read from the incremental `rem` tally (no deck rescan).
        const long remaining = (long)deck.size() - pos;
        if (remaining <= 0) return false;
        long bust = 0;
        for (int v = 1; v <= 13; ++v) { VINSTR(++ic_scan); if (q.cnt[v] >= q.allowed(v)) bust += rem[v]; }
        const double pb = (double)bust / (double)remaining;
        const int sum = q.numsum();
        const double bonus = (nc == kFlip7Target - 1) ? (double)kFlip7Bonus : 0.0;  // one card from Flip 7
        // hit iff expected gain beats expected loss (avg safe number ~6.5). (The
        // per-decision FP divide here is NOT on the critical path -- cross-multiplying
        // it away measured neutral-to-worse, as the OoO core hides its latency.)
        return (1.0 - pb) * (6.5 + bonus) > pb * (double)sum;
    }

    void take_turn(int i) {
        if (!active(i)) return;
        if (!wants_hit(i)) { P[i].status = VST_STAYED; return; }
        int c = draw_card();
        if (c < 0) { P[i].status = VST_STAYED; return; }
        resolve_drawn(i, (uint8_t)c, /*chooser=*/i);
    }

    void play_round(int starter) {
        VINSTR(++ic_rounds);
        shuffle();
        round_over = false;
        for (int i = 0; i < n; ++i) { P[i].reset(); }
        const int guard_max = 300 * n;
        for (int g = 0; g < guard_max; ++g) {
            int i = (starter + g) % n;
            bool any = false; for (int k = 0; k < n; ++k) any |= active(k);
            if (!any) break;
            take_turn(i);
            if (round_over) break;
        }
        for (int i = 0; i < n; ++i) total[i] += vengeance_score(P[i]);
    }

    int play_game(int target) {
        VINSTR(++ic_games);
        for (int i = 0; i < n; ++i) total[i] = 0;
        int starter = 0;
        for (int round = 0; round < 1000; ++round) {
            play_round(starter);
            starter = (starter + 1) % n;
            long mx = -1; for (long t : total) mx = std::max(mx, t);
            if (mx >= target) {
                int win = -1, ties = 0;
                for (int i = 0; i < n; ++i) if (total[i] == mx) { ties++; win = i; }
                if (ties == 1) return win;
            }
        }
        return -1;
    }
};

// Run G N-player Vengeance games; player 0 gets win credit (1/n on no-decision).
// Per-game splitmix64 seeding => reproducible regardless of thread count.
inline VengeanceStats run_vengeance_tournament(const std::vector<int>& tps, int target,
                                               uint64_t G, uint64_t seed) {
    const int n = (int)tps.size();
    auto work = [&](uint64_t g0, uint64_t g1, VengeanceStats& out) {
        Xoshiro256pp rng;
        VengeanceGame game(rng, tps);
        for (uint64_t g = g0; g < g1; ++g) {
            uint64_t sm = seed + g; rng.seed(splitmix64(sm));
            int r = game.play_game(target);
            out.games++;
            if (r == 0) out.p0_score += 1.0;
            else if (r < 0) out.p0_score += 1.0 / (double)n;
        }
    };
    unsigned T = std::thread::hardware_concurrency(); if (!T) T = 1;
    if (T == 1 || G < 8192) { VengeanceStats s; work(0, G, s); return s; }
    std::vector<VengeanceStats> parts(T);
    std::vector<std::thread> th;
    const uint64_t chunk = (G + T - 1) / T;
    for (unsigned t = 0; t < T; ++t) {
        uint64_t g0 = (uint64_t)t * chunk, g1 = std::min(G, g0 + chunk);
        if (g0 >= g1) break;
        th.emplace_back([&, t, g0, g1] { work(g0, g1, parts[t]); });
    }
    for (auto& x : th) x.join();
    VengeanceStats s;
    for (const auto& p : parts) { s.games += p.games; s.p0_score += p.p0_score; s.rounds += p.rounds; }
    return s;
}

}  // namespace flip7
