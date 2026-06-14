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

### Chapter 2 — what the optimal strategy actually is

For the numbers-only game we can state the optimal policy in one line and measure
how close simple rules come:

> **Keep flipping until your chance of busting on the next card reaches ≈26%, then stop.**

That single bust-probability threshold earns **99.8%** of the optimal expected
score. By contrast, **"stop after k cards" is a poor rule** — the best fixed count
(k=3) gets only **95.1%**. So the decision is driven by *risk*, not by how many
cards you hold.

| Simple rule | best setting | % of optimal |
|---|---|---:|
| stop at k unique cards | k = 3 | 95.1% |
| stop at banked sum ≥ T | T = 23 | 99.76% |
| stop at P(bust) ≥ θ | θ ≈ 0.26 | **99.81%** |

If you want a "stop at score ≥ N" rule, **N = 23** is best, and the plateau is broad
— any cutoff in **21–25 stays above 99%** of optimal (e.g. 20→98.3%, 22→99.6%,
24→99.4%, 26→98.5%), so the choice is forgiving.

The last ~0.2% is genuinely **non-separable** — no single threshold is exactly
optimal. The clinching evidence: the optimal policy **hits** `{3,4,5,6,8,9}` at a
**39.7%** bust risk (it's one card from the +15 Flip-7 bonus, worth the gamble) yet
**stays** on `{10,12}` at only **26.0%** (two high cards, little upside). A
higher-risk hand hits while a lower-risk hand stays — so the exact call needs the
whole hand, not just one number. (For the full 94-card game the optimal policy is
the billion-entry table, not a one-liner, but the same shape holds: Second Chance
makes you push harder, big modifiers raise the stakes, and Freeze/Flip Three you
can't control.)

### Chapter 3 — tail probabilities (the "perfect game")

Flip 7 — collecting 7 unique numbers for the +15 — is the game's perfect hand, and
how often it happens is entirely strategy-dependent. Maximizing P(Flip 7) is
provably just **"always hit"** (staying gives zero chance), confirmed by an exact
Flip-7-maximizing DP.

| Policy | E[score] | P(bust) | P(Flip 7) |
|---|---:|---:|---:|
| **numbers only**, play for score | 18.5652 | 0.3135 | **0.28%** |
| **numbers only**, go for Flip 7 (always hit) | 5.7433 | 0.9134 | **8.66%** |
| **all 94**, play for score | 20.2980 | 0.2935 | **1.26%** |
| **all 94**, go for Flip 7 (always hit) | 9.84 | 0.742 | **8.83%** |

Going for it raises the Flip-7 rate **~31×** (numbers only), at the cost of busting
~91% of the time and gutting the expected score. With all 94 cards the perfect-game
ceiling is **~8.8%**: Second Chance (saves ~11.5% of attempts) and Flip Three (forces
draws toward 7) push it up, while a forced Freeze ends ~17% of attempts — and the
action cards actually *lower* the bust rate (a Freeze ends in a Stay, not a bust).
All figures are exact (numbers-only) or Monte-Carlo with the exact policies, cross-checked.

### Chapter 4 — competitive first-to-200 (win probability)

Across rounds, the goal is no longer the most points — it's to reach 200 first.
Optimizing for *winning* differs from optimizing for *score*. (Model: independent
rounds; action-card targeting is deferred to Ch. 5.)

- **Solitaire-to-target:** it takes on average **11.51 rounds** (numbers only,
  18.57/round) or **10.68 rounds** with all 94 cards (20.30/round) to reach 200.
  Both MC-confirmed. The all-94 round distribution comes from the exact 94-card
  optimal policy (`data/round_pmf_all94.txt`, regenerated by `make all-cards`).
- **Both players greedy (play for score):** by symmetry W(0,0) = **0.5** exactly.
  A lead is worth more later — an ~18-point (one-round) edge is ≈63% early but
  **≈82%** near the finish (numbers only; the all-94 curves are slightly flatter
  because action cards add round-score variance).
- **Best response to a greedy field** (re-optimize each round for win probability
  instead of score): **W(0,0) = 0.5593** — a win-probability player beats a
  score-maximizing player **56%–44%**, an edge of **+5.9 points of win
  probability** purely from *how* it manages risk:

  | your total | opp total | round P(bust) | E[round] | stance |
  |---:|---:|---:|---:|---|
  | 110 | 160 | 0.567 | 15.9 | far behind → **push** (more risk, *lower* mean) |
  | 110 | 135 | 0.438 | 17.9 | behind → push |
  | 110 | 110 | 0.341 | 18.5 | even → ≈ greedy |
  | 110 | 85 | 0.257 | 18.4 | ahead → play safe |
  | 110 | 60 | 0.211 | 18.0 | far ahead → **safest** |

  When far behind it deliberately accepts a **57%** bust risk (vs greedy's 31%) and
  even a *lower* expected score, trading mean for the variance it needs to catch up;
  when ahead it cuts risk to ~21%. The best-response grid is exact, cross-checked by
  a Monte-Carlo tournament (0.5589 vs 0.5593). Still to come (Ch. 4 cont.): the
  symmetric Nash equilibrium via fictitious play.

## Build & run

Requires a C++20 compiler (tested with Apple Clang on an Apple M4 Pro; the
Makefile auto-detects the best `-mcpu`).

```sh
make            # build all binaries + tests
make run        # fast headline numbers (Ch.1 progression, Ch.2 strategy, Ch.3 tails)
make test       # assert them (DP<->MC agreement + regression), incl. Ch.4 A-C
make competitive  # Ch.4 best-response grid + push/safe + MC (~24 s) -- opt-in
make all-cards    # the full 94-card solitaire DP (~5 min, ~34 GB) -- opt-in
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

### PMU profiling (`make profile`)

`perf/profile.cpp` instruments the hot kernels with [`perf.h`](third_party/perf.h)
(a vendored single-header Apple-Silicon PMU library) to find optimization
opportunities. `make profile` then `sudo ./bin/profile` (programming counters
needs root) reports IPC and per-op cost plus stall sources. On the M4 Pro:

| kernel | ns/op | IPC | dTLB/kI | L2TLB/kI | bound by |
|---|---:|---:|---:|---:|---|
| dense DPs (numbers, +mods) | 44–215 ns/state | 3.0–4.0 | ~0 | ~0 | **compute** (healthy) |
| hashed DP (+Second Chance) | 157 ns/state | 1.95 | **31.4** | **10.9** | **TLB** (random hash probes) |
| MC numbers / +mods | 13–16 ns/roll | 3.6–3.8 | ~0 | ~0 | **compute** (healthy) |
| MC +Second Chance | 57 ns/roll | 1.87 | 21.1 | 4.4 | hash policy lookup per decision |
| xoshiro256++ next / bounded | 0.7–0.8 ns | 4.9–6.3 | 0 | 0 | not a bottleneck |
| `round_solve` (Ch.4 inner) | 47 µs/solve | 3.63 | 0.3 | 0.1 | **compute** (full 8K-state re-solve) |

Findings: the **hashed DPs are TLB-bound** (the ~1 GB / 34 GB tables thrash the
TLB — 2 MB superpages or a denser table are the lever); the **Ch.4 best response
is compute-bound on re-solving the within-round DP** (only `g[0]` changes between
fixed-point iterations, so an incremental solve would cut the ~24 s); the dense
DPs, Monte-Carlo, and PRNG are already near peak IPC.

## Relation to prior work

The only prior public analysis we are aware of is a **Monte-Carlo simulation**
treatment on **dirck.dev**. This project provides, to our knowledge, the first
**exact** (dynamic-programming) and **competitive** (win-probability /
equilibrium) treatment of Flip 7. We use Monte-Carlo only to *cross-check* the
exact results, not to produce them.

## Status

The complete single solitaire turn is solved exactly for **all 94 cards**
(numbers, modifiers, Second Chance, Freeze, Flip Three) and Monte-Carlo confirmed.
Chapters 1–3 (the solitaire round: optimal score, strategy/separability, tail
probabilities) and Chapter 4 layers A–D (across-rounds win probability + best
response to a greedy field) are complete and verified. Remaining: the symmetric
**Nash equilibrium** via fictitious play (Ch. 4 cont.) and **adversarial
action-card targeting** in the N-player game (Ch. 5) — specified in `PLAN.md`.
