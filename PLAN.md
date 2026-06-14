# Flip 7 — Exact & Competitive Analysis: Project Plan

## Goal

Provide the **first exact and competitive treatment** of Flip 7. Prior public work is a
Monte-Carlo simulation blog (dirck.dev); we add exact dynamic programming (optimal
stopping), the separability analysis, exact tail probabilities, and a competitive
first-to-200 / equilibrium treatment — each cross-validated against an independent
Monte-Carlo simulator.

**Discipline:** every headline number is produced by an **exact DP** *and* confirmed by an
**independent Monte-Carlo** run before we report it. We prove the small cases first and
build up. We do not start with the full 94-card multiplayer game.

---

## Deck facts (verified against the rules)

- **94 cards total.**
- **Number cards (79):** one `0`, one `1`, two `2`s, three `3`s, … twelve `12`s
  (`count(0)=1`, `count(v)=v` for `v=1..12`; sum = 79).
- **Modifier cards (6):** one each of `+2, +4, +6, +8, +10, ×2`.
- **Action cards (9):** three each of `Freeze`, `Flip Three`, `Second Chance`.

### Reshuffle / shoe behavior (real game = continuous shoe)

- Played cards go to a **discard pile kept aside** at end of round — *not* reshuffled in.
- The **draw pile persists across rounds** (passed to the next dealer).
- The discard pile is reshuffled into a new draw pile **only when the draw pile is
  exhausted** (can occur mid-round).
- ⇒ **Card counting is real** in the physical game (high cards that are "out" stay out).

**Modeling decision (resolved):**
- **Chapters 1–3 (single turn):** reshuffle policy is irrelevant — one turn never nears
  deck exhaustion. Draw **without replacement from one fresh deck**. For numbers-only the
  remaining composition is *fully determined by the held hand*, so the DP is exact with no
  extra deck state.
- **Chapter 4 (across rounds):** the MC simulator models the **true continuous shoe**
  (ground truth). The DP uses a **fresh-deck-per-round approximation** for tractability,
  and we **measure the discrepancy with MC** rather than assume it away. We do **not**
  build a with-replacement idealization (the user specified without replacement).

---

## Deliverables (chapters)

1. **Single-turn optimal stopping (solitaire).** Exact DP maximizing `E[round score]`.
   State ≈ (set of held numbers, modifiers held incl. ×2, Second-Chance flag); remaining
   deck determined by what's been drawn (draw without replacement). Report the optimal
   expected round score and the Hit/Stay policy.
2. **Separability question.** Is optimal play a simple threshold ("stop at k uniques" or
   "stop at score ≥ T"), or non-separable? Expectation: **non-separable** — holding *low*
   numbers (few copies left) is far safer than *high* numbers (many copies), so the
   decision depends on *which* numbers you hold, not just how many or your score.
   Quantify how far the best simple threshold is from optimal.
3. **Tail probabilities (the "perfect game" analog).** `P(bust)` and `P(Flip 7)` under the
   expected-optimal policy vs. a policy that *maximizes* `P(Flip 7)`. Show they are
   strategy-dependent.
4. **Competitive first-to-200.** Win-probability DP where the optimal action depends on
   your running total vs. opponents' (push variance when behind, play safe when ahead).
   Best response to a field of expected-optimal players, and a symmetric Nash equilibrium
   via fictitious play. Validate with a multiplayer Monte-Carlo tournament.
5. **Action-card interaction.** Freeze / Flip Three targeting is adversarial (hand Freeze
   to the leader). Richest and hardest part — tackled last, inside the competitive chapter.

---

## Staging (prove small first)

- **(a) Numbers-only solitaire single-turn DP** → clean exact expected score + MC agreement.
- **(b) Add modifiers (+N, ×2), then Second Chance.**
- **(c) Add across-rounds first-to-200 win-probability DP** (solitaire-to-target, then 2-player).
- **(d) Add action-card adversarial targeting + N-player equilibrium.**

At each stage: a **small proof run with verbose streaming progress** before any long
compute, and **DP numbers must match the independent Monte-Carlo** before we trust them.

---

## Chapter-1 DP (concrete formulation)

State `S` = 13-bit `number_mask` over values `{0..12}`. Numbers-only, no modifiers/actions.

- Cards drawn so far `d = popcount(S)`; remaining total `T = 79 − d`.
- Remaining count of value `v`: `r(v) = count(v) − [v ∈ S]`.
  (Note `count(0)−1 = 0` and `count(1)−1 = 0`: holding `0` or `1` can **never** bust you;
  holding `v` exposes `v−1` bust cards — the core of the separability result.)

Recursion:
```
if popcount(S) == 7:            EV(S) = sum(S) + 15            // forced Flip 7, terminal
else:
    EV_stay(S) = sum(S)
    EV_hit(S)  = Σ_{v ∉ S}  (count(v) / T) · EV(S | (1<<v))    // bust branch contributes 0
    EV(S)      = max(EV_stay(S), EV_hit(S))
    policy(S)  = HIT  iff  EV_hit(S) > EV_stay(S)
```

**Headline number:** `EV(∅)` = optimal expected round score from an empty hand. State count
≤ Σ_{k=0}^{7} C(13,k) = 5,812 — trivially exact.

**MC cross-check:** simulate many turns following `policy`, drawing without replacement from
a fresh 79-card number deck (partial Fisher–Yates). Report mean ± standard error and the
agreement (z-score) vs. `EV(∅)`.

### Correctness wrinkle for stage (b)+ (record now)

The held hand captures `number_mask` and `modifier_mask` exactly (modifiers are one-of-each;
numbers are a set). It does **not** capture how many *traceless* cards have been consumed:
discarded duplicate **Second Chance** cards and resolved **Flip Three** cards leave no mark
in the hand yet change the remaining deck. For an exact DP at stage (b)+, extend the state
with small consumed-counts (e.g. `sc_consumed ∈ 0..3`, `flip3_consumed ∈ 0..3`). Freeze is
terminal, so it can't have been consumed while still deciding. (Chapter 1 numbers-only is
free of this wrinkle.)

---

## Core technical design (performance-first)

### State representation — a fistful of bits
A player's number cards form a **set** (duplicate ⇒ bust), so the collection is a **13-bit
mask**. Hot ops are single-instruction:
- bust test on value `v`: `(mask >> v) & 1`
- add card: `mask |= 1u << v`
- Flip-7 test: `popcount(mask) == 7`
- number sum: precomputed `sum_lut[mask]`

Packed `PlayerState` in one `uint32`:
```
bits  0..12  number_mask        (13)
bits 13..18  modifier_mask      (6: +2,+4,+6,+8,+10,×2)
bit  19      second_chance_held (1)
bits 20..22  status             (ACTIVE/STAYED/BUSTED/FROZEN/FLIP7)
bits 23..31  spare
```

Branchless score:
```
mult  = 1 + ((modmask >> 5) & 1)                 // ×2 bit
score = sum_lut[number_mask]*mult + addmod_lut[modmask] + 15*(popcount(number_mask)==7)
```
(`×2` multiplies number cards only — not the +15, not the +N modifiers.)

### Lookup tables (L1-resident, constexpr)
- `sum_lut[8192]` (u8 ≤ 78) — Σ of set bit-values.
- `popcnt_lut[8192]` (u8) — or ARM `cnt`.
- `addmod_lut[64]` (u8 ≤ 30) — additive-modifier sum.

### Deck & draw — two representations
- **Shoe array** (MC forward sim): `uint8 deck[N]` + cursor; **partial Fisher–Yates**
  (swap front with random remaining, advance cursor) gives draw-without-replacement at
  **O(1)/draw with zero upfront shuffle** — only as many swaps as cards actually drawn.
- **Composition counts** (DP / exact probability): small count arrays; bust probability is
  a masked horizontal sum `Σ_{v ∈ mask} count(v) / total` (vectorizable).

### PRNG
- **xoshiro256++** as the scalar workhorse (4×u64, passes BigCrush, far faster/better than
  `mt19937`); `jump()` for independent streams.
- **Philox4×32-10** (counter-based) behind the same interface for **bit-reproducible**
  runs (regression) and the lane-parallel path.
- **Lemire nearly-divisionless** bounded integers for unbiased shuffling (no modulo, no
  divide).

### SIMD / NEON (Apple M4 Pro, 128-bit NEON)
- Vectorize the **regular math kernels**: batched score, bust-probability/EV reductions
  over composition vectors, popcount/sum, policy sweeps over the state table.
- Optional **masked-lockstep MC** of independent single-player rollouts (per-lane `alive`
  predicate) for EV-by-simulation.
- Keep divergent control flow (multiplayer + targeting) **scalar**; thread across games
  later. Don't force the whole engine into SIMD.

### Memory & containers (no `unordered_map` in hot paths)
- **Dense flat arrays indexed by the packed state** = perfect hashing, zero collisions,
  cache-friendly (the DP/EV tables).
- If a sparse integer-keyed map is ever needed: **flat open-addressing, linear probing**,
  power-of-two capacity — never `std::unordered_map`.
- `std::vector` only as an owning contiguous buffer; hot data via raw pointers / `span`.
- **Arena/bump allocator** for per-thread scratch (reset per batch).
- Align NEON buffers to 16 B; pad shared hot structures to the **128-byte Apple-Silicon
  cache line** (false-sharing safety once threaded).

### M4 build & tuning
- C++20, `-O3` with **arch auto-detect** (`-mcpu=apple-m4`, fall back to
  `-mcpu=apple-latest`/`native`), `-flto`, `-fno-exceptions -fno-rtti` in the hot core.
- Confirm cache geometry on the box: `sysctl -a | grep -i cache` / `hw.cachelinesize`.
- Verify auto-vectorization: `-Rpass=loop-vectorize -Rpass-missed=loop-vectorize`.
- Microbench timing via `mach_absolute_time` / `cntvct_el0`; Instruments for hotspots.

---

## Tech & conventions

- **C++20**, `-O3` with arch auto-detect.
- **Exact DP lives in a shared header**; **one small program per chapter**.
- **Makefile** with `make` (build), `make run` (print headline numbers), `make test`
  (assert them).
- **Monte-Carlo simulators single-threaded for now** (multithreading deferred).
- New repo at `~/claude_projects/flip_seven`; **`git init`**.
- **Work directly on `main`; commit/push to `main`; no feature branches.**
- **README** with: headline-results table, methods, and an honest **"Relation to prior
  work"** section citing the dirck.dev simulation posts — stating precisely that only a
  Monte-Carlo blog exists and we provide the first exact / competitive treatment (no more).

---

## Rigor guardrails

- Exact DP and Monte-Carlo must **independently agree** before any headline number is
  reported.
- State the optimal expected score, the bust / Flip-7 odds, and the equilibrium results
  **plainly**, each with its MC cross-check.
- Be precise about prior work: only a Monte-Carlo blog exists; we provide the first exact /
  competitive treatment — say exactly that, no more, no less.

## Performance reporting

Everything is performance-first: each program **prints/logs timing and throughput**
(states evaluated, rollouts/sec, wall-clock per phase, DP solve time, MC samples and
rate). Performance stats are part of every chapter's output.

---

## First milestone (concrete)

Implement the **numbers-only solitaire single-turn DP**, print the **exact optimal expected
round score** `EV(∅)`, and confirm it with a **quick Monte-Carlo run** (mean ± stderr,
agreement). **Show that number and the agreement before building further.**

---

## File layout (initial)

```
PLAN.md                     this plan
flip7_rules.md              verified ruleset
README.md                   headline table, methods, prior-work note  (added with ch.1)
Makefile                    make / make run / make test
include/
  flip7_core.hpp            packed state, LUTs, score/bust (shared)
  flip7_deck.hpp            shoe (partial Fisher–Yates) + composition counts
  flip7_rng.hpp             xoshiro256++, Philox, Lemire-bounded
  flip7_dp.hpp              exact DP (grows per chapter)
ch1_solitaire_turn/         numbers-only single-turn DP + MC cross-check
ch2_separability/
ch3_tails/
ch4_competitive/            first-to-200 win DP, fictitious play, MC tournament
ch5_actions/                (folded into ch4) adversarial targeting
tests/                      DP-vs-MC asserts, golden enumeration, RNG/shuffle stats
```

## Roadmap

0. Repo + `git init` on `main`; core header + LUTs + xoshiro; correctness tests.
1. **Chapter 1** — numbers-only single-turn DP + MC agreement (the milestone).
2. Chapter 2 — separability (best threshold vs. optimal, quantified).
3. Chapter 3 — tail probabilities under expected-optimal vs. Flip-7-maximizing policies.
4. Stage (b) — modifiers + Second Chance (with the consumed-count state extension).
5. Chapter 4 — first-to-200 win DP, best response, fictitious-play Nash, MC tournament.
6. Chapter 5 — action-card adversarial targeting inside the competitive model.
7. Later — NEON kernels where regular, multithreading across games.
```
