// flip7_core.hpp — shared primitives for the Flip 7 analysis.
//
// A player's number cards form a SET (a duplicate value busts you), so the
// whole collection is a 13-bit mask over values {0..12}. This header holds the
// deck composition and the handful of bit ops the engine leans on.
#pragma once
#include <cstdint>

namespace flip7 {

inline constexpr int kNumValues      = 13;  // number values 0..12
inline constexpr int kNumberDeckSize = 79;  // one 0, one 1, two 2s, ... twelve 12s
inline constexpr int kFlip7Target    = 7;   // 7 unique numbers => round-ending bonus
inline constexpr int kFlip7Bonus     = 15;

// Copies of number value v in a fresh deck: one 0, otherwise v copies of v.
inline constexpr int numberCount(int v) { return v == 0 ? 1 : v; }

// Sum of the number-card values present in mask m (a weighted popcount).
inline int maskSum(uint16_t m) {
    int s = 0;
    while (m) { s += __builtin_ctz(m); m &= (uint16_t)(m - 1); }
    return s;
}

// Number of distinct number cards held.
inline int maskPop(uint16_t m) { return __builtin_popcount(m); }

// --- Modifier cards (Stage b) -------------------------------------------------
// One each of +2,+4,+6,+8,+10,x2. Bits 0..4 are the additive +N (value 2*(i+1));
// bit 5 is the x2 multiplier. x2 doubles the NUMBER-card sum only -- not the +N
// modifiers and not the +15 Flip 7 bonus.
inline constexpr int kNumModifiers = 6;
inline constexpr int kX2Bit        = 5;

inline int modAdditive(uint16_t mm) {
    int s = 0;
    for (int i = 0; i < 5; ++i)
        if (mm & (1u << i)) s += 2 * (i + 1);
    return s;
}
inline int modMult(uint16_t mm) { return (mm & (1u << kX2Bit)) ? 2 : 1; }

// Round score for a non-busted hand: numbers (x2 first), then +N modifiers,
// then the +15 bonus iff 7 unique numbers are held.
inline int fullScore(uint16_t nm, uint16_t mm) {
    const int pc = maskPop(nm);
    return maskSum(nm) * modMult(mm) + modAdditive(mm) + (pc == kFlip7Target ? kFlip7Bonus : 0);
}

inline constexpr int kNumSecondChance = 3;  // Second Chance cards in a full deck

}  // namespace flip7
