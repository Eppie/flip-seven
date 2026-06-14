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

}  // namespace flip7
