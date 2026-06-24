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
which multiply the state space to **1,208,732,216 states**. This count is not a
surprise: it factors as **21,980,032 base configs × 55 action-card "modes"** (an
upper bound of 1,208,901,760), and the measured value is lower by just 0.014% —
i.e. on average 54.99 of the 55 modes are reachable per config.

That factorization is also the storage layout: rather than a flat 1.2-billion-slot
hash (32 GB, TLB-bound — ~13.7 page-walks/state), we **hash only the base config
`(nm, mm, sch, scn, extra)` and store its 55 modes `(f3, fz, forced)` contiguously**,
indexed directly with no per-value key. That removes the 8-byte key from every value
slot, keeps the action-card transitions (Flip Three / Freeze / forced) inside one
~440-byte block, and shrinks the page tables enough to stay cached — cutting the
solve from **~5.3 min / 34 GB to ~69 s / ~11 GB (≈4.6× faster, exact result
unchanged**, MC-confirmed). Because of its size this DP is still opt-in
(`make all-cards` / `make test-all-cards`), separate from the fast suite.

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
  a Monte-Carlo tournament (0.5589 vs 0.5593).

- **Symmetric Nash equilibrium** (fictitious self-play, `make nash`, ~3 min): the
  value is **0.5** by symmetry — and an MC tournament with both players using the
  converged policy returns **0.49986**, confirming it. The equilibrium *policy* is
  the same push/safe schedule, slightly *more* polarized than the best-response to
  greedy (behind 50 → **59%** bust risk; ahead 50 → **17%**; even ≈ greedy), because
  at equilibrium the opponent adapts too. Notably the stage game is **mixed**
  (matching-pennies-like: vs an aggressive opponent you play safe to let them bust,
  vs a safe opponent you push for variance), so randomizing your round-aggression
  matters — the best *deterministic* round-policy is held to only ~**0.39** against
  the equilibrium. (Fictitious play converges the policy as ~1/k; the per-iteration
  pure-best-response value is not a 0.5 signal, since pure play can't mirror a mixed
  strategy.)

### Chapter 5 — adversarial action-card targeting (Freeze / Flip Three)

When you flip a **Freeze** or **Flip Three** you must hand it to an active player —
yourself or an opponent — so the targeting is adversarial. We solve the *value* of
each card exactly (numbers-only DP, MC-confirmed), then measure targeting in the
real 94-card 2-player game (`make actions`).

- **Flip Three is never a gift.** Forcing the target to draw 3 cards changes their
  expected round score by `dMean ≤ 0` everywhere — it is only a *weapon*. But its
  bite depends on how deep the target already is:

  | target holds (k uniques) | 0 | 1 | 2 | **4** | 5 | 6 |
  |---|---:|---:|---:|---:|---:|---:|
  | Δ to target's E[round] | −0.1 | −1.7 | −6.0 | **−14.4** | −7.4 | −0.5 |
  | resulting P(bust) | 0.34 | 0.48 | 0.63 | **0.79** | 0.62 | 0.37 |
  | resulting P(Flip 7) | 0.00 | 0.00 | 0.00 | 0.21 | 0.38 | **0.63** |

  It is **~neutral when the target is shallow** (the forced draws are ones an optimal
  player would take anyway), **most damaging at k≈4** (−14 points), then **weaker
  again near Flip 7** — forcing draws on a 5–6 card hand often *completes their +15
  bonus*. So aim it at a mid-deep opponent, never at someone one card from Flip 7.
  Flipping an optimal opponent right as they would bank (`Flip Three @ stop`) cuts
  their round mean **18.57 → 7.89** and busts them **85%** of the time (DP=MC exactly).

- **Freeze caps the target** at their current `sum(S)`; the points it denies are
  `EV(S) − sum(S) ≥ 0`. The competitive catch: it also removes the target's *bust
  risk*, so against an opponent who might bust on their own it is a double-edge.

- **Whom to Flip Three (exact win-prob DP).** In first-to-200 the answer is
  **unconditional: always the opponent** — attacking only lowers their mean (to ~7.9),
  so self/none are dominated at every score. The standings change only *how much* the
  swing is worth, not the direction. (Granting a free, optimally-aimed Flip Three
  every round is worth **W(0,0)=0.942** — an idealized ceiling that isolates the
  direction; the real card frequency is measured below.)

- **The real 94-card 2-player game** (organic action draws, exact rules — set-aside
  nested actions, Flip 7 ends the round; players use the numbers+modifier optimal
  Hit/Stay, only targeting differs; 300 k games/matchup):

  | matchup | agent win rate |
  |---|---:|
  | adversarial vs. random targeting | **0.571** (+7.1%) |
  | adversarial vs. naive (use-on-self) | **0.642** (+14.2%) |
  | random vs. random | 0.501 (sanity) |
  | adversarial vs. adversarial | 0.502 (sanity) |

  Adversarial targeting is worth **+7 points of win probability** over random and
  **+14** over the naive self-target baseline. The symmetric matchups return ~0.5 as
  they must. Numbers-only exact layer (Parts A–B) is in the fast test suite; Part C
  is the faithful-rules Monte-Carlo ground truth.

### Arbitrary number of players

The competitive analysis generalizes from 2 players to any *n*. The single-turn
solitaire DP is player-count-independent; the across-rounds machinery is the part
that scales. Where the win-probability DP stays **exact** is set by the joint-total
state, which is `target^n` cells: 40 K at n=2, 8 M at n=3, intractable beyond. So:

| layer | n=2 | n=3 | n≥4 |
|---|---|---|---|
| greedy win grid | exact | **exact** (`win_prob_greedy_n`) | Monte-Carlo |
| best response vs field | exact | **exact** (`best_response_grid_n`) | Monte-Carlo |
| Flip-Three targeting (idealized) | exact | **exact** (`win_prob_flip3_target_n3`) | MC |
| symmetric Nash | exact (fictitious play) | value 1/n by symmetry, MC-confirmed | MC |
| real 94-card targeting | MC | MC | MC |

By symmetry the *value* of the symmetric game is exactly **1/n**, and the exact
3-player grid reproduces it (`W(0,0,0) = 1/3` to 1e-9) — a clean cross-check the
2-player case (0.5) can't distinguish from a coding error. Best-responding to a
greedy field still beats the field: at n=3 a win-probability re-optimizer reaches
**~0.44–0.46** (vs 1/3), and the real-rules adversarial-targeting edge persists as
the table fills, diluting gracefully (+7%→+5.6%→+4.4% vs a random field at n=2→3→4;
+14%→+13.9%→+12.5% vs a naive field). Every exact n=3 number is confirmed by an
independent Monte-Carlo tournament, and every symmetric n-player MC returns 1/n.

The exact Flip-Three targeting DP also generalizes: with one optimally-aimed Flip
Three per round, the 3-player win probability from `(0,0,0)` is **0.7916** (vs 1/3
— worth **+0.458**), MC-confirmed at 0.7920, and the optimal aim is at the leading
opponent. (As in the 2-player case this is an idealized one-free-Flip-Three-per-
round model; the `~3/94` organic frequency is what the real 94-card tournament
measures, where the targeting edge is the +5.6%/+13.9% above.)

The exact n=3 grids are **parallelized across cores** (each coordinate-sum layer is
independent), as is the Monte-Carlo (per-game-seeded, so results are reproducible
regardless of thread count). Run them with:

```
./bin/ch4_competitive N [target]      # greedy + best response (exact for N<=3)
./bin/ch4_nash players=N [target]     # symmetric value 1/N, MC-confirmed
./bin/ch5_actions players=N [target]  # real-rules targeting vs an N-field
make competitive-3p                   # the full exact 3-player first-to-200
```

The 2-player chapters, outputs, and asserted headline numbers are unchanged — the
generic code is validated against the frozen 2-player path (`win_prob_greedy_n` at
n=2 matches `win_prob_greedy` to <1e-9, and the n=2 best response reproduces
`W_br(0,0)=0.5593`).

### Decision oracle (`decide`)

`bin/decide` turns the analysis into an in-the-moment recommendation. You hand it
the live situation — number of players, your hand, your match total, the opponents'
totals (and hands, if you want targeting advice), the cards you've already seen, and
whether you hold a Freeze / Flip Three — and it returns the **win-probability-optimal
Hit/Stay** (plus the expected-score-optimal call for contrast), the **count-aware
P(bust)** of hitting now, and whom to target.

```
./bin/decide --players 2 --my-hand 9,11,12 --my-total 120 --opp 168
  WIN-OPTIMAL : STAY   (win prob  hit=0.1492  stay=0.1631)
  count-aware P(bust) if you hit now = 0.3816   (fresh-deck 0.3816)

# ...but if you've watched five 11s and six 12s go by, your bust cards are gone:
./bin/decide --players 2 --my-hand 9,11,12 --my-total 120 --opp 168 \
             --seen 11,11,11,11,11,12,12,12,12,12,12
  WIN-OPTIMAL : HIT    (win prob  hit=0.1668  stay=0.1631)
  count-aware P(bust) if you hit now = 0.2769   (fresh-deck 0.3816)
```

A single exact table over the full state (your hand × every opponent's hand × all
totals × cards-seen × n) is impossible, so the oracle exploits the same factoring
the rest of the project rests on. It composes three layers:

1. a **win-value backbone** `winval(total)` = P(you win the match if you bank to this
   total) — read from the exact best-response grid for **n≤3** (built once and cached
   to `data/winbr_n*_t*.bin`; the n=3 grid takes a few minutes the first time),
   Monte-Carlo for **n≥4**;
2. a **count-aware within-round solve** from your current hand using the *actual*
   remaining deck (your hand + every visible hand + `--seen` cards removed) — so
   seeing your bust cards leave the shoe lowers P(bust) and can flip the call, exactly
   as above;
3. a **targeting** evaluation that scores aiming Freeze / Flip Three at each opponent
   through the win grid (Freeze denies a deep-but-not-finished leader's upside; Flip
   Three attacks a mid-deep hand).

Honest bounds, also printed by the tool: it is a **one-step best response against a
greedy field**, not the exact Nash of the full multiplayer game; the count-aware
layer corrects the *current* round exactly while the multi-round continuation still
assumes a fresh deck (the measured-small shoe effect); and card-counting is
numbers-focused (which number cards are gone → bust risk).

## Build & run

Requires a C++20 compiler (tested with Apple Clang on an Apple M4 Pro; the
Makefile auto-detects the best `-mcpu`).

```sh
make            # build all binaries + tests
make run        # fast headline numbers (Ch.1 progression, Ch.2 strategy, Ch.3 tails)
make test       # assert them (DP<->MC agreement + regression), incl. Ch.4 A-C
make competitive  # Ch.4 best-response grid + push/safe + MC (~9 s) -- opt-in
make nash         # Ch.4 symmetric Nash via fictitious self-play (~3 min) -- opt-in
make actions      # Ch.5 action-card targeting: exact A/B + real 94-card duel (~6 s)
make all-cards    # the full 94-card solitaire DP (~5 min, ~34 GB) -- opt-in
make test-all-cards
make profile      # PMU profiling (sudo ./bin/profile for counters)
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
throughput.

**NEON-vectorized Monte-Carlo** (`include/flip7_sim_neon.hpp`): the numbers-only
rollout doesn't actually need a physical deck — the held mask `S` (popcount `p`)
fully determines the remaining 79-card deck, so we run **8 independent rollouts in
lockstep** (`uint16x8`, one per lane) with per-lane alive predicates and persistent
lane refill, and a vectorized xoshiro128++×8. The next card is drawn by **rejection
over the full deck**: a uniform position `k ∈ [0,79)`, its `(value, first-copy
flag)` from a 79-byte LUT (one cache line), with each held value's single reserved
first-copy position treated as a no-op redraw — any other copy of a held value is a
bust, a copy of an unheld value is kept. That reproduces draw-without-replacement
exactly with one small gather instead of a 13-step categorical decode. It is a
*different sampler* (not a bit-for-bit replay of the scalar deck), but provably the
*same distribution* — verified by matching the exact score pmf bin-by-bin to **max
deviation < 10⁻⁴** across all 79 bins, plus the exact DP's mean/bust/Flip-7. Net
**~2.4× over the already-fast scalar simulator** (13 → ~5.5 ns/rollout), via the
LUT gather, an incrementally-tracked popcount, and deferred horizontal reductions
(folded to scalars every 8192 steps). Multithreading across games is the remaining
lever (see `PLAN.md`).

### PMU profiling (`make profile`)

Every chapter program prints its own wall-clock and throughput (DP states/s, MC
rollouts/s, games/s, solve times), and `perf/profile.cpp` instruments the hot
kernels of **all five chapters** with [`perf.h`](third_party/perf.h) (a vendored
single-header Apple-Silicon PMU library) to find optimization opportunities.
`make profile` then `sudo ./bin/profile` (programming counters needs root) reports
IPC and per-op cost plus stall sources. On the M4 Pro:

| kernel | ns/op | IPC | dTLB/kI | L2TLB/kI | bound by |
|---|---:|---:|---:|---:|---|
| dense DPs (numbers, +mods) | 44–215 ns/state | 3.0–4.0 | ~0 | ~0 | **compute** (healthy) |
| hashed DP (+Second Chance) | 157 ns/state | 1.95 | **31.4** | **10.9** | **TLB** (random hash probes) |
| MC numbers / +mods | 13–16 ns/roll | 3.6–3.8 | ~0 | ~0 | **compute** (healthy) |
| MC numbers (NEON ×8) | ~5.5 ns/roll | — | ~0 | ~0 | **~2.4×** vs scalar (8 lockstep lanes) |
| MC +Second Chance | 57 ns/roll | 1.87 | 21.1 | 4.4 | hash policy lookup per decision |
| xoshiro256++ next / bounded | 0.7–0.8 ns | 4.9–6.3 | 0 | 0 | not a bottleneck |
| `round_solve` (Ch.4 inner) | 47 µs/solve | 3.63 | 0.3 | 0.1 | **compute** (full 8K-state re-solve) |
| Ch.5 pmf builders (A/B) | 35–77 µs/call | — | — | — | dense forward enumeration |
| Ch.5 94-card duel (C) | ~2.6 µs/game | — | — | — | divergent control + dense lookup |

(The last two rows are the Chapter 5 kernels — the `[chapter 5 …]` sections of
`make profile`; the IPC/TLB/branch columns populate under `sudo`.) Findings: the
**hashed DPs are TLB-bound** (the ~1 GB / 34 GB tables thrash the TLB — the
`[TLB detail]` line shows one hardware page-table walk per L2 TLB miss, ~one walk
per state, so most of the 647 cyc/state is address translation). 2 MB superpages
are impossible on Apple Silicon (16 KB base pages, no userspace large pages), but
the **denser-table lever paid off for the all-94 solve**: the base-config/mode
block layout (above) shrank it 34 GB → ~11 GB and made the page tables cacheable,
for ≈4.6× (the +Second-Chance DP could get the same treatment). The dense DPs,
Monte-Carlo, PRNG, and the Ch.5 duel are already near peak IPC.

The Ch.4 best response *was* compute-bound on re-solving the within-round DP, but
that is now fixed: only `g[0]` changes between fixed-point iterations and the
solver is linear in `g[0]` within a policy region, so the self-loop is solved in
closed form (`w* = (U0 − B0·D0·w)/(1 − B0·D0)`, where `B0 = P(bust)`), cutting it
from ~6–20 solves/state to ~3.4 and the grid from ~24 s to **~7 s** (identical
W_br, MC-confirmed).

## Relation to prior work

The only prior public analysis we are aware of is a **Monte-Carlo simulation**
treatment on **dirck.dev**. This project provides, to our knowledge, the first
**exact** (dynamic-programming) and **competitive** (win-probability /
equilibrium) treatment of Flip 7. We use Monte-Carlo only to *cross-check* the
exact results, not to produce them.

## Status

The complete single solitaire turn is solved exactly for **all 94 cards**
(numbers, modifiers, Second Chance, Freeze, Flip Three) and Monte-Carlo confirmed.
**All five chapters are complete and verified:** the solitaire round (optimal
score, strategy/separability, tail probabilities), the across-rounds competition
(win probability, best response, symmetric Nash via fictitious play), and
adversarial action-card targeting (exact Freeze/Flip-Three values, optimal
targeting, and the real 94-card 2-player game). Every headline number is produced
by an exact DP and independently confirmed by Monte-Carlo, per the project's
discipline.

The competitive analysis further extends to **an arbitrary number of players**: the
win-probability DP is exact for up to 3 players (parallelized across cores) and
Monte-Carlo beyond, with the symmetric value 1/n reproduced exactly at n=3 and
every result cross-checked by an n-player tournament (see *Arbitrary number of
players* above). The 2-player headline numbers are unchanged.

Finally, `bin/decide` packages all of this into an **interactive decision oracle**:
given a live situation (players, your hand/total, opponents' totals and hands, the
cards seen so far, and any action card you hold) it returns the win-optimal Hit/Stay,
a count-aware bust probability, and a targeting recommendation — composing the exact
sub-results where they exist and Monte-Carlo where they don't (see *Decision oracle*
above).

## License

MIT — see [`LICENSE`](LICENSE). The vendored Apple-Silicon PMU helper in
`third_party/` carries its own license ([`third_party/perf.LICENSE`](third_party/perf.LICENSE)).
