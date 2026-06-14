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

## Build & run

Requires a C++20 compiler (tested with Apple Clang on an Apple M4 Pro; the
Makefile auto-detects the best `-mcpu`).

```sh
make          # build chapter binaries + tests
make run      # print the Chapter 1 headline numbers
make test     # assert them (DP<->MC agreement + regression)
```

`./bin/ch1_solitaire_turn [num_rollouts] [seed]` to vary the Monte-Carlo run.

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

Performance is a first-class goal: bit-packed state, constexpr-friendly
primitives, flat arrays indexed directly by the packed state (no hashing),
xoshiro256++ rather than `mt19937`, and partial Fisher–Yates. Every program
prints timing and throughput. NEON/SIMD kernels and multithreading are planned
(see `PLAN.md`) but not yet needed at this scale.

## Relation to prior work

The only prior public analysis we are aware of is a **Monte-Carlo simulation**
treatment on **dirck.dev**. This project provides, to our knowledge, the first
**exact** (dynamic-programming) and **competitive** (win-probability /
equilibrium) treatment of Flip 7. We use Monte-Carlo only to *cross-check* the
exact results, not to produce them.

## Status

Chapter 1 (this milestone) is complete and verified. Chapters 2–5 — separability,
tail probabilities, the first-to-200 win-probability DP and Nash equilibrium via
fictitious play, and adversarial action-card targeting — are specified in
`PLAN.md` and not yet implemented.
