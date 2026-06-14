// test_ch2 — assert the separability / optimal-strategy findings (numbers only).
#include "flip7_core.hpp"
#include "flip7_dp.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <vector>

using namespace flip7;

static int failures = 0;
static void check(bool ok, const char* what) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

static double pbust(uint16_t S) {
    const int T = kNumberDeckSize - maskPop(S);
    int b = 0;
    for (int v = 0; v < kNumValues; ++v) if (S & (1u << v)) b += numberCount(v) - 1;
    return (double)b / (double)T;
}
template <class Hit>
static double evaluate(Hit should_hit) {
    std::vector<double> reach(1 << kNumValues, 0.0);
    reach[0] = 1.0;
    double e = 0.0;
    for (int p = 0; p < kFlip7Target; ++p)
        for (uint32_t S = 0; S < (1u << kNumValues); ++S) {
            if (maskPop((uint16_t)S) != p) continue;
            const double pr = reach[S];
            if (pr == 0.0) continue;
            if (!should_hit((uint16_t)S)) { e += pr * (double)maskSum((uint16_t)S); continue; }
            const int T = kNumberDeckSize - p;
            for (int v = 0; v < kNumValues; ++v) {
                const uint16_t bit = (uint16_t)(1u << v);
                if (S & bit) continue;
                const double pv = pr * (double)numberCount(v) / (double)T;
                const uint16_t Sn = (uint16_t)(S | bit);
                if (maskPop(Sn) == kFlip7Target) e += pv * (double)(maskSum(Sn) + kFlip7Bonus);
                else                             reach[Sn] += pv;
            }
        }
    return e;
}
static uint16_t mk(std::initializer_list<int> vs) { uint16_t m = 0; for (int v : vs) m |= (uint16_t)(1u << v); return m; }

int main() {
    printf("Chapter 2 tests\n");
    SolitaireTurnDP dp;
    const double opt = dp.optimal();

    // Best bust-probability threshold is near-optimal; count threshold is not.
    double best_th = 0, best_k = 0;
    for (int i = 1; i < 100; ++i) { const double th = i / 100.0;
        best_th = std::max(best_th, evaluate([th](uint16_t S){ return pbust(S) < th; })); }
    for (int k = 3; k <= 7; ++k)
        best_k = std::max(best_k, evaluate([k](uint16_t S){ return maskPop(S) < k; }));

    printf("  (opt=%.4f, best P(bust)-threshold=%.4f=%.1f%%, best count-threshold=%.4f=%.1f%%)\n",
           opt, best_th, 100*best_th/opt, best_k, 100*best_k/opt);
    check(best_th / opt > 0.995, "a single P(bust) threshold captures >99.5% of optimal");
    check(best_k  / opt < 0.96,  "the best card-count threshold is poor (<96%)");

    // Non-separability: a higher-risk hand hits while a lower-risk hand stays.
    check(dp.hit[mk({3,4,5,6,8,9})] == true,  "HIT {3,4,5,6,8,9} (P(bust)=0.40, chasing the +15)");
    check(dp.hit[mk({10,12})]       == false, "STAY {10,12} (P(bust)=0.26)");
    check(dp.hit[mk({1,2,3})]       == true,  "HIT {1,2,3}");
    check(dp.hit[mk({10,11,12})]    == false, "STAY {10,11,12} (same count, opposite call)");

    printf("%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
