// test_oracle -- the decision oracle (decide). All n=2 grid work is fast; the
// checks below use small situations and the (instant) n=2 best-response grid.
#include "flip7_oracle.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace flip7;

static int failures = 0;
static void check(bool ok, const char* what) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}
static bool close(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

// fresh-deck remaining counts for a held hand (no seen cards).
static void fresh_counts(uint16_t hand, int r0[kNumValues]) {
    for (int v = 0; v < kNumValues; ++v) r0[v] = numberCount(v) - ((hand >> v) & 1);
}

int main() {
    printf("Oracle tests\n");
    Oracle oracle(2, 200);  // 2-player grid is instant

    const uint16_t hand = (1u << 9) | (1u << 11) | (1u << 12);  // {9,11,12} -- deep, high
    std::vector<int> opp{100}; std::vector<char> act{1}; std::vector<uint16_t> oh{0};

    // 1. fresh-deck reduction: count-aware P(bust) with no seen cards == the closed form
    {
        int r0[kNumValues]; fresh_counts(hand, r0);
        OracleDecision d = oracle.decide(hand, 0, false, 120, opp, act, oh, r0, 0);
        long T = kNumberDeckSize - maskPop(hand); long b = 0;
        for (int v = 0; v < kNumValues; ++v) if ((hand >> v) & 1) b += numberCount(v) - 1;  // remaining copies
        check(close(d.p_bust, (double)b / T, 1e-12), "fresh-deck P(bust) matches closed form Sigma (count(v)-1)/T");
    }

    // 2. card counting moves bust odds the right way
    {
        int rfresh[kNumValues]; fresh_counts(hand, rfresh);
        OracleDecision base = oracle.decide(hand, 0, false, 120, opp, act, oh, rfresh, 0);
        // remove the dangerous high copies we could bust on (more 11s and 12s gone)
        int rsafe[kNumValues]; fresh_counts(hand, rsafe);
        rsafe[11] -= 6; rsafe[12] -= 8; if (rsafe[11] < 0) rsafe[11] = 0; if (rsafe[12] < 0) rsafe[12] = 0;
        OracleDecision safe = oracle.decide(hand, 0, false, 120, opp, act, oh, rsafe, 0);
        check(safe.p_bust < base.p_bust - 1e-6, "seeing our bust cards leave the deck LOWERS count-aware P(bust)");
        // removing low cards we can't bust on (0,1) barely moves it
        int rlow[kNumValues]; fresh_counts(hand, rlow); rlow[0] = 0; rlow[1] = 0;
        OracleDecision low = oracle.decide(hand, 0, false, 120, opp, act, oh, rlow, 0);
        check(low.p_bust > base.p_bust, "removing low cards (can't bust us) slightly RAISES P(bust) (smaller deck)");
    }

    // 3. standings: deep behind pushes, far ahead plays safe (win-optimal vs EV)
    {
        const uint16_t mid = (1u << 6) | (1u << 7) | (1u << 8) | (1u << 9);  // {6,7,8,9}=30, deepish
        int r0[kNumValues]; fresh_counts(mid, r0);
        OracleDecision behind = oracle.decide(mid, 0, false, 40, std::vector<int>{180},
                                              std::vector<char>{1}, std::vector<uint16_t>{0}, r0, 0);
        OracleDecision ahead  = oracle.decide(mid, 0, false, 180, std::vector<int>{40},
                                              std::vector<char>{1}, std::vector<uint16_t>{0}, r0, 0);
        // when far behind the win-optimal call should be at least as aggressive as when far ahead
        check(!(ahead.hit_winopt && !behind.hit_winopt),
              "far-behind is at least as likely to HIT as far-ahead (push when behind)");
    }

    // 4. targeting: holding Freeze with two opponents, aim at the leader.
    // Small target keeps the (cached) n=3 grid build to ~seconds.
    {
        Oracle o3(3, 60);
        const uint16_t myh = (1u << 3) | (1u << 4);             // shallow
        int r0[kNumValues]; fresh_counts(myh, r0);
        // opp1 is a harmless trailer; opp2 leads with a SHALLOW hand (high deniable
        // upside, ~one round from winning) -> Freeze should cap opp2, not opp1.
        std::vector<int> ot{15, 42};
        std::vector<char> oa{1, 1};
        std::vector<uint16_t> ohh{(uint16_t)(1u << 2), (uint16_t)(1u << 1)};
        OracleDecision d = o3.decide(myh, 0, false, 40, ot, oa, ohh, r0, /*Freeze*/ 1);
        check(d.has_action, "freeze decision produced a target");
        check(d.target == 1, "Freeze caps the threatening leader (#2), not the harmless trailer");
    }

    printf("%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
