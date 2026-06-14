// flip7_rng.hpp — fast PRNG for the Monte-Carlo simulators.
//
// xoshiro256++ (Blackman & Vigna): 4x u64 state, passes BigCrush, far faster
// and statistically better than std::mt19937. Seeded via splitmix64. Bounded
// integers use Lemire's nearly-divisionless method (unbiased, no modulo/divide
// on the hot path).
#pragma once
#include <cstdint>

namespace flip7 {

inline uint64_t splitmix64(uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

struct Xoshiro256pp {
    uint64_t s[4];

    static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

    void seed(uint64_t seed) {
        uint64_t sm = seed;
        for (int i = 0; i < 4; ++i) s[i] = splitmix64(sm);
    }

    uint64_t next() {
        const uint64_t result = rotl(s[0] + s[3], 23) + s[0];
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return result;
    }

    // Uniform integer in [0, n) via Lemire's nearly-divisionless algorithm.
    uint64_t bounded(uint64_t n) {
        __uint128_t m = (__uint128_t)next() * (__uint128_t)n;
        uint64_t lo = (uint64_t)m;
        if (lo < n) {
            uint64_t thresh = (uint64_t)(-(int64_t)n) % n;  // (2^64 - n) % n
            while (lo < thresh) {
                m = (__uint128_t)next() * (__uint128_t)n;
                lo = (uint64_t)m;
            }
        }
        return (uint64_t)(m >> 64);
    }
};

}  // namespace flip7
