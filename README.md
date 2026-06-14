# Flip 7 — Exact & Competitive Analysis

An exact dynamic-programming and game-theoretic treatment of the card game
**Flip 7**, with every headline number independently confirmed by Monte-Carlo
simulation. See [`flip7_rules.md`](flip7_rules.md) for the verified ruleset and
[`PLAN.md`](PLAN.md) for the full roadmap.

## Headline results

### Chapter 1 — numbers-only solitaire, single turn

Optimal stopping that maximizes the expected round score, using only the 79
number cards (one `0`, one `1`, two `2`s, … twelve `12`s), drawn **without
replacement**. Stay banks the sum of your held numbers; a duplicate busts you
(score 0); collecting 7 unique numbers ends the turn with a +15 bonus.

| Quantity | Exact (DP) | Monte-Carlo (10⁸ rollouts) |
|---|---:|---:|
| **Optimal E[round score]** | **18.5652176** | 18.565918 (±0.0013, z = +0.5σ) |
| P(bust) under optimal play | 0.3134965 | 0.313470 |
| P(Flip 7) under optimal play | 0.0028113 | 0.002816 |
| P(stay) under optimal play | 0.6836922 | — |

The DP evaluates all 5,812 reachable states in ~0.16 ms; the simulator runs at
~80 M rollouts/s single-threaded. A forward pass over the optimal policy
reproduces the exact optimum to 1.6 × 10⁻¹³, and the outcome probabilities sum
to 1 — independent of the value-iteration that produced the policy.

**The optimal policy is non-separable.** It is *not* a threshold on the number of
cards held nor on the running score. Holding *low* numbers is far safer than
*high* ones — holding value `v` exposes exactly `v − 1` bust cards, so `0` and
`1` can never bust you, while a `12` can bust you 11 ways. Hence:

```
{1,2,3}        sum= 6  -> HIT     {10,11,12}     sum=33  -> STAY
{0,1,2,3,4}    sum=10  -> HIT     {6,7,8,9,10}   sum=40  -> STAY
```

Same hand size, opposite decisions. Quantifying the gap between optimal play and
the best simple threshold is Chapter 2.

### Chapter 1, Stage b — adding modifiers and Second Chance

The same single-turn optimal stopping, with the real deck's modifier cards
(`+2,+4,+6,+8,+10,×2`) and Second Chance cards, drawn without replacement. All
results are **exact** and Monte-Carlo confirmed.

| Deck | Exact optimal E[round score] | Monte-Carlo |
|---|---:|---:|
| numbers only (79) | 18.5652 | z = +0.5σ ✓ |
| + modifiers (85) | **20.2291** | 20.2296 (z = +0.2σ) ✓ |
| + Second Chance (88) | **22.2504** | 22.2531 (z = +1.1σ) ✓ |
| + Freeze + Flip Three (94, all cards) | **20.2980** | 20.2990 (z = +0.4σ) ✓ |

Modifiers are one-of-each and cannot bust, so the held hand determines the deck —
a dense exact DP (`×2` doubles the number total only, not the +N modifiers or the
+15 bonus). **Second Chance is subtler:** when a save fires, the duplicate number
card is also discarded, removing a *second* copy of that value from the deck with
no trace in your hand. The exact DP tracks per-value "extra discards" (saves total
≤ 3) so the remaining deck is always exact; because every transition permanently
removes a card, the state graph is acyclic. That state is sparse, so the full
solver memoizes in a flat open-addressing hash map (≈22M states, ~3 s) rather than
a dense array. Every result uses the exact game rules.

**All 94 cards (solitaire).** Freeze and Flip Three are self-targeted: Freeze
forces an immediate Stay, Flip Three forces 3 draws (a bust or 7th unique ends it
early; further Flip Threes stack; a Freeze drawn during it is pending and Stays
after). These **lower** the optimal expected score (22.2504 → **20.2980**, −1.95):
Freeze caps your upside (you freeze ~12% of turns) and Flip Three forces risky
draws (~11%). The exact DP adds a forced-draw counter and a pending-freeze flag,
which multiply the state space to **1,208,732,216 states** — solved in ~5.3 min
using a 34 GB open-addressing table. This count is not a surprise: it factors as
**21,980,032 base configs × 55 action-card "modes"** (an upper bound of
1,208,901,760), and the measured value is lower by just 0.014% — i.e. on average
54.99 of the 55 modes are reachable per config. Because of its size this DP is
opt-in (`make all-cards` / `make test-all-cards`), separate from the fast suite.

## Build & run

Requires a C++20 compiler (tested with Apple Clang on an Apple M4 Pro; the
Makefile auto-detects the best `-mcpu`).

```sh
make            # build all binaries + tests
make run        # fast headline numbers (numbers, +modifiers, +Second Chance)
make test       # assert them (DP<->MC agreement + regression)
make all-cards  # the full 94-card solitaire DP (~5 min, ~34 GB) -- opt-in
make test-all-cards
```

`./bin/ch1_solitaire_turn [num_rollouts] [seed]` and
`./bin/ch1b_modifiers_sc [num_rollouts] [seed]` to vary the Monte-Carlo runs.
The full solve streams live progress (states, %, rate, ETA) to stderr.

## Methods

- **Exact DP.** A player's number cards form a *set*, so the state is a 13-bit
  mask over `{0..12}`. Because we draw without replacement and each held value
  was drawn exactly once, the held hand fully determines the remaining deck — so
  the mask is the entire state. Value iteration:
  `EV(S) = max(sum(S), Σ_{undrawn v} P(v)·EV(S∪{v}))`, with a duplicate draw
  contributing 0 and a 7th unique giving `sum + 15`.
- **Independent Monte-Carlo.** A separate simulator draws without replacement
  from a fresh 79-card deck via partial Fisher–Yates (the deck is restored by
  undoing the ≤7 swaps, so there is no per-rollout shuffle cost), follows the DP
  policy, and reports mean ± standard error. PRNG is xoshiro256++ with Lemire
  bounded sampling.
- **Discipline.** No headline number is reported until the exact DP and the
  Monte-Carlo agree.

## Performance

Performance is a first-class goal: bit-packed state, dense flat arrays indexed
directly by the packed state where the state allows it (numbers, numbers+
modifiers), a hand-rolled flat open-addressing hash map only where it does not
(the sparse Second Chance state — never `std::unordered_map`), xoshiro256++
rather than `mt19937`, and partial Fisher–Yates. Every program prints timing and
throughput. NEON/SIMD kernels and multithreading are planned (see `PLAN.md`) but
not yet needed at this scale.

## Relation to prior work

The only prior public analysis we are aware of is a **Monte-Carlo simulation**
treatment on **dirck.dev**. This project provides, to our knowledge, the first
**exact** (dynamic-programming) and **competitive** (win-probability /
equilibrium) treatment of Flip 7. We use Monte-Carlo only to *cross-check* the
exact results, not to produce them.

## Status

The complete single solitaire turn is solved exactly for **all 94 cards**
(numbers, modifiers, Second Chance, Freeze, Flip Three) and Monte-Carlo confirmed.
Action cards are self-targeted here; their *adversarial* targeting is the
multiplayer problem (Chapter 5). Chapters 2–5 — separability, tail probabilities,
the first-to-200 win-probability DP and Nash equilibrium via fictitious play, and
adversarial action-card targeting — are specified in `PLAN.md` and not yet
implemented.
