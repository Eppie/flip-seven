// test_vengeance -- the Flip 7: With a Vengeance Monte-Carlo engine.
#include "flip7_vengeance.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace flip7;

static int failures = 0;
static void check(bool ok, const char* what) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

int main() {
    printf("Vengeance tests\n");
    Xoshiro256pp rng; rng.seed(1);
    VengeanceGame g(rng, std::vector<int>{0, 0});

    // 1. deck composition (108)
    g.build_deck();
    int cnt[27] = {0}; for (uint8_t c : g.deck) cnt[c]++;
    check(g.deck.size() == 108, "deck has 108 cards");
    check(cnt[VC_ZERO] == 1 && cnt[VC_UNLUCKY7] == 1 && cnt[VC_LUCKY13] == 1, "one each: Zero, Unlucky 7, Lucky 13");
    check(cnt[7] == 6 && cnt[13] == 12, "six regular 7s + twelve regular 13s (specials make 7 and 13)");
    int numtotal = cnt[VC_ZERO]; for (int v = 1; v <= 13; ++v) numtotal += cnt[v]; numtotal += cnt[VC_UNLUCKY7] + cnt[VC_LUCKY13];
    check(numtotal == 92, "92 number/special-number cards total");
    bool mods = true; for (uint8_t m = VC_M2; m <= VC_DIV2; ++m) mods &= (cnt[m] == 1);
    check(mods, "one each of the 6 negative/divide modifiers");
    bool acts = true; for (uint8_t a = VC_JUSTONE; a <= VC_DISCARD; ++a) acts &= (cnt[a] == 2);
    check(acts, "two each of the 5 action cards");

    // 2. scoring order: numbers -> /2 (floor) -> negatives -> Flip7 +15 / Zero
    {
        VPlayer p; p.add(4); p.add(8); p.add(11); p.status = VST_STAYED;  // sum 23
        check(vengeance_score(p) == 23, "plain sum {4,8,11} == 23");
        p.modmask |= (1u << MM_DIV2);
        check(vengeance_score(p) == 11, "/2 floors 23 -> 11");
        p.modmask |= (1u << MM_M4);
        check(vengeance_score(p) == 7, "then -4 -> 7 (divide before negatives)");
        VPlayer z; z.add(0); z.zero = true; z.add(5); z.status = VST_STAYED;
        check(vengeance_score(z) == 0, "The Zero forces 0 without Flip 7");
        z.status = VST_FLIP7;
        check(vengeance_score(z) == 5 + kFlip7Bonus, "Flip 7 lifts The Zero and adds +15");
    }

    // 3. Lucky 13 lets you hold two 13s; a third busts
    {
        g.P[0] = VPlayer{}; g.give_number(0, 13); g.give_number(0, VC_LUCKY13);
        check(g.P[0].status == VST_ACTIVE && g.P[0].cnt[13] == 2, "Lucky 13 + a 13 = two 13s, no bust");
        g.give_number(0, 13);
        check(g.P[0].status == VST_BUSTED, "a third 13 busts");
        // without Lucky 13, a second 13 busts
        g.P[0] = VPlayer{}; g.give_number(0, 13); g.give_number(0, 13);
        check(g.P[0].status == VST_BUSTED, "two 13s without Lucky 13 busts");
    }

    // 4. Unlucky 7 discards everything else, keeps one 7
    {
        g.P[0] = VPlayer{}; g.give_number(0, 5); g.give_number(0, 9); g.P[0].modmask = 0xFF;
        g.give_number(0, VC_UNLUCKY7);
        check(g.P[0].ncards() == 1 && g.P[0].cnt[7] == 1 && g.P[0].modmask == 0,
              "Unlucky 7 discards all other number + modifier cards");
    }

    // 5. symmetric field -> 1/n (the engine-unbiasedness sanity)
    {
        const uint64_t G = 400'000;
        for (int n = 2; n <= 4; ++n) {
            auto s = run_vengeance_tournament(std::vector<int>(n, VTP_RANDOM), 200, G, 0xABCDEFULL * n);
            const double rate = s.p0_score / s.games, se = std::sqrt((1.0 / n) * (1 - 1.0 / n) / (double)G);
            char msg[96]; snprintf(msg, sizeof msg, "symmetric %d-player field == 1/%d (within 5 se)", n, n);
            check(std::fabs(rate - 1.0 / n) < 5 * se, msg);
        }
    }

    // 6. adversarial targeting beats a naive field
    {
        std::vector<int> t{VTP_ADVERSARIAL, VTP_SELF, VTP_SELF};
        auto s = run_vengeance_tournament(t, 200, 300'000, 0x5151ULL);
        check(s.p0_score / s.games > 1.0 / 3.0, "adversarial player beats a naive 3-player field");
    }

    printf("%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
