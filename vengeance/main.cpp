// Flip 7: With a Vengeance -- faithful-rules Monte-Carlo summary.
//
// The exact DP that solves the original game does not transfer (Lucky 13 breaks the
// unique-set state; Steal/Swap/Discard/Unlucky 7 move cards between players), so we
// measure the sequel by simulation -- the same ground-truth approach the repo uses
// for the messy multiplayer cases. See flip7_vengeance_rules.md and
// include/flip7_vengeance.hpp (which documents the heuristic hit/stay and targeting).
#include "flip7_vengeance.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace flip7;

int main() {
    constexpr int kT = 200;
    const uint64_t G = 500'000;
    const uint64_t S = 0x9E3779B97F4A7C15ULL;   // spread seeds far apart (independent streams)

    printf("=== Flip 7: With a Vengeance -- Monte-Carlo (faithful rules) ===\n");
    printf("deck: 108 cards = 92 numbers (count(v)=v for 1..13, +Zero, with one 7=Unlucky, one 13=Lucky)\n");
    printf("      + 6 negative/divide modifiers (one each) + 5 action types x2.\n");
    printf("model: heuristic one-step-EV hit/stay + take-that targeting. A symmetric field\n");
    printf("       gives each player 1/n (engine sanity); only the *edge* is heuristic.\n\n");

    const auto t0 = std::chrono::steady_clock::now();

    printf("--- symmetric field (everyone plays the same policy) -> should be 1/n ---\n");
    for (int n = 2; n <= 6; ++n) {
        auto s = run_vengeance_tournament(std::vector<int>(n, VTP_RANDOM), kT, G, S * (uint64_t)n);
        printf("   n=%d: player-0 win rate = %.4f   (1/%d = %.4f)\n", n, s.p0_score / s.games, n, 1.0 / n);
    }

    printf("\n--- one ADVERSARIAL player (aims take-that at the leader) vs a field ---\n");
    printf("    edge = player-0 win rate minus the fair share 1/n\n");
    printf("    n |  vs naive (self-target)  |  vs random targeting\n");
    for (int n = 2; n <= 6; ++n) {
        std::vector<int> vs_self(n, VTP_SELF);   vs_self[0]  = VTP_ADVERSARIAL;
        std::vector<int> vs_rand(n, VTP_RANDOM); vs_rand[0]  = VTP_ADVERSARIAL;
        auto a = run_vengeance_tournament(vs_self, kT, G, S * (uint64_t)(n + 10));
        auto b = run_vengeance_tournament(vs_rand, kT, G, S * (uint64_t)(n + 20));
        const double ra = a.p0_score / a.games, rb = b.p0_score / b.games, fair = 1.0 / n;
        printf("    %d |  %.4f  (%+.4f)        |  %.4f  (%+.4f)\n", n, ra, ra - fair, rb, rb - fair);
    }

    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("\n   (%llu games/matchup; %.1f s total)\n", (unsigned long long)G, sec);
    printf("   Takeaway: the extra take-that (Steal/Swap/Discard + negatives) makes targeting\n");
    printf("   worth even more than in the original, and -- as there -- the edge dilutes as the\n");
    printf("   table fills. Exact play is unsolved; these are heuristic-policy estimates.\n");
    return 0;
}
