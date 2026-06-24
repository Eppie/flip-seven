# Flip 7: With a Vengeance — Complete Ruleset

**Players:** 3–18 · **Goal:** Be the first to reach **200+ points**, scored across
multiple rounds. **Tagline:** *"No one's safe!"*

A standalone sequel to [Flip 7](flip7_rules.md). Same push-your-luck core — keep
flipping number cards, but a **duplicate number busts you** (lose the round) — with a
meaner, take-that twist: the modifiers are now **negative cards you play on
opponents**, there's **no Second Chance**, and **staying no longer makes you safe**
(steal/swap/forced-draw cards can still hit a player who has stayed).

---

## Deck Composition (108 cards)

### Number cards (92 total)

The number deck is the original's, **extended to 13**: one `0`, one `1`, two `2`s, …
up to thirteen `13`s — i.e. **`count(v) = v`** for `v = 1..13`, plus a single `0`.
Two of those slots are special versions: **one of the seven `7`s is the Unlucky 7**,
and **one of the thirteen `13`s is the Lucky 13**.

| Value | Copies | | Value | Copies |
|------|--------|---|------|--------|
| 0 (The Zero) | 1 | | 7  | 7  (6 regular + 1 **Unlucky 7**) |
| 1    | 1      | | 8  | 8  |
| 2    | 2      | | 9  | 9  |
| 3    | 3      | | 10 | 10 |
| 4    | 4      | | 11 | 11 |
| 5    | 5      | | 12 | 12 |
| 6    | 6      | | 13 | 13 (12 regular + 1 **Lucky 13**) |

Sum = 1 + (1 + 2 + … + 13) = **92**.

### Modifier cards (6 total) — one each

All negative / divide, all **"Play on any player"**:
**−2, −4, −6, −8, −10, and ÷2.**

### Action cards (10 total) — two each

**Just One More, Flip Four, Swap, Steal, Discard.**

(92 number + 6 modifier + 10 action = **108**. Note what's **gone** versus the
original: there are **no positive `+N` modifiers, no `×2`, no Second Chance, no
Freeze, and no Flip Three** — Just One More folds in a freeze, Flip Four supplants
Flip Three, and modifiers went negative.)

---

## How a Round Works

1. The dealer deals **one card face-up to each player**, going around the table.
2. On your turn you choose **"Hit"** (take another card) or **"Stay"** (bank your
   points and sit out the rest of the round).
3. Play continues until every player has **stayed, busted, or hit Flip 7** — but
   because action cards can target stayed players, **a stayed player is not safe
   until the round actually ends**.
4. Everyone scores, points are added to running totals, and a new round begins with
   a reshuffled deck.

### Busting

Flip a **number card that duplicates one you already have** and you **bust**: out of
the round, **0 points**. (There is **no Second Chance** in this game to save you.)
Modifier and action cards never cause a bust by themselves. *Exception:* **Lucky 13**
lets you hold one extra `13` (see below).

**Busting isn't only on your own flip.** Because **Swap** and **Steal** move number
cards between players, you can be handed a **duplicate of a number you already hold —
and bust — even after you've stayed**. After any such move, every affected player's
face-up numbers are rechecked, and a single **Swap can bust both players at once**
(e.g. each holds a 10 and an 11; swapping one player's 10 for the other's 11 leaves
both with a duplicate). This is the heart of *"no one's safe."*

### The "Flip 7" bonus 🎉

Collect **7 number cards** without busting and you immediately **Flip 7**: score a
**+15 bonus** and the round ends for you. Only *number* cards count toward the 7
(modifiers don't). The Zero and Unlucky 7 **do** count as number cards.

---

## Special Number Cards

- **0 / The Zero** — your round score becomes **0 regardless of your other cards,
  *unless* you achieve Flip 7**. It still counts as **one of your seven** number
  cards, and **while you hold it you must keep hitting** (you can't bank a guaranteed
  zero). So it's a trap that turns into a jackpot only if you complete the set.

- **Unlucky 7** — when you get it, **discard all your other number *and* modifier
  cards**; you keep **only the Unlucky 7**. It counts as a `7` from then on (so a
  second `7` will bust you).

- **Lucky 13** — counts as a `13`, but lets you **safely hold one *other* `13`**
  without busting. A **third `13` still busts** you. *(This is the rule that breaks
  the original's "your numbers are a set" invariant.)*

---

## Modifier Cards

Modifiers **don't count toward Flip 7**, **never bust**, and **normally can't push a
score below 0**. When drawn you may keep one **or give it to another non-busted
player**. They apply at scoring (not to busted players).

| Card | Effect |
|------|--------|
| **−2 / −4 / −6 / −8 / −10** | Subtract that many points from that player's round score. |
| **÷2** | **Halve that player's number-card total** at scoring (**rounded down**), applied **before** the negative modifiers. |

---

## Action Cards

Action cards **resolve immediately, then are discarded**. They can target **any
non-busted player — including yourself and players who have already stayed**. If you
are the only non-busted player, you must use it on yourself.

| Card | Effect |
|------|--------|
| **Just One More** | Chosen player **draws one more card** (resolve it if needed), then **must stay**. |
| **Flip Four** | Chosen player takes **up to 4 cards, one at a time**, stopping early on **bust** or **Flip 7**; action/modifier cards drawn during it resolve normally. |
| **Swap** | **Swap any two face-up cards** — yours ↔ another's, or two other players' (number and modifier cards). **After the swap, recheck every affected player: a player now holding a duplicate number busts — even one who had already stayed, and both players can bust from a single swap.** |
| **Steal** | **Take any face-up card** from another player into your own. (If the stolen number duplicates one you hold, you bust.) |
| **Discard** | Force a non-busted player to **discard one of their face-up cards**. |

---

## Scoring (strict order)

For each player who **stayed** (busted players score 0):

1. **Sum your number cards.**
2. **Apply ÷2** if a ÷2 card is on you (number total only, **rounded down**).
3. **Subtract the negative modifiers** (−2 / −4 / −6 / −8 / −10); not below 0.
4. **Add +15** if you achieved Flip 7.
5. **The Zero** forces the round score to **0** unless Flip 7 was achieved.

*Example:* numbers 4 + 8 + 11 = 23 → **÷2 → 11** (rounded down) → **−4 modifier → 7**.

---

## Winning

At the **end of any round** in which at least one player has reached **200+ points**,
the **highest total wins**. Ties at/above 200 play on until someone leads.

---

## How this differs from the original (and why it's harder to solve exactly)

Versus [base Flip 7](flip7_rules.md), Vengeance changes the cards that made the
original cleanly solvable:

- **Lucky 13 breaks the unique-set state** — a hand can legally hold two `13`s, so a
  player's numbers are no longer a plain 13-bit set/mask.
- **Steal / Swap / Discard / Unlucky 7 move cards between players or out of hand** —
  so "your held hand determines the remaining deck" (the property that made the
  single-turn DP exact) no longer holds.
- **÷2 and negative modifiers** reshape scoring and are **take-that** (played on
  opponents), coupling players far more tightly than the original's targeting.
- **No Second Chance** — busting is unforgiving.

The exact dynamic-programming machinery in this repo doesn't transfer directly, but
the **faithful-rules Monte-Carlo engine** (`include/flip7_duel.hpp`) — which already
simulates organic action cards and inter-player targeting for any number of players —
is the natural basis for a "Vengeance mode."

---

## Sources

- [The Op — Flip 7: With a Vengeance FAQs](https://theop.games/pages/flip-7-wav-faqs)
- [Happy Piranha — How to Play Flip 7 With a Vengeance](https://happypiranha.com/blogs/board-game-rules/how-to-play-flip-7-with-a-vengeance-board-game-rules-instructions)
- [The Board Game Family — Flip 7 & Flip 7 With a Vengeance review](https://www.theboardgamefamily.com/2026/05/flip-7-and-flip-7-with-a-vengeance-card-game-review/)
- [Meeple Mountain — Flip 7: With a Vengeance review](https://www.meeplemountain.com/reviews/flip-7-with-a-vengeance/)
- [BoardGameGeek — Flip 7: With A Vengeance](https://boardgamegeek.com/boardgame/463441/flip-7-with-a-vengeance)
- Card faces and deck-count breakdown verified from the publisher's card gallery and
  distributor contents listing.
