// decide -- the Flip 7 decision oracle CLI.
//
//   decide --players N [--target 200] \
//          --my-hand 3,7,9 [--my-mods x2,+4] [--my-sc] --my-total 110 \
//          --opp 95:5,8 --opp 130:2,3,4 ...   (total[:held-numbers] per opponent) \
//          [--seen 12,12,11] [--have-freeze | --have-flip3]
//
// Prints the win-probability-optimal Hit/Stay (and the expected-score-optimal call
// for contrast), the count-aware P(bust) of hitting, and -- if you hold an action
// card -- whom to target. See flip7_oracle.hpp for the model and its honest bounds.
#include "flip7_oracle.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace flip7;

static uint16_t parse_hand(const char* s) {
    uint16_t m = 0;
    if (!s || !*s) return 0;
    std::string t(s);
    size_t i = 0;
    while (i < t.size()) {
        size_t j = t.find(',', i);
        if (j == std::string::npos) j = t.size();
        if (j > i) { int v = atoi(t.substr(i, j - i).c_str()); if (v >= 0 && v < kNumValues) m |= (uint16_t)(1u << v); }
        i = j + 1;
    }
    return m;
}

static uint16_t parse_mods(const char* s) {
    uint16_t mm = 0;
    if (!s || !*s) return 0;
    std::string t(s); size_t i = 0;
    while (i < t.size()) {
        size_t j = t.find(',', i); if (j == std::string::npos) j = t.size();
        std::string tok = t.substr(i, j - i);
        if (tok == "x2" || tok == "X2") mm |= (uint16_t)(1u << kX2Bit);
        else if (tok == "+2")  mm |= 1u << 0;
        else if (tok == "+4")  mm |= 1u << 1;
        else if (tok == "+6")  mm |= 1u << 2;
        else if (tok == "+8")  mm |= 1u << 3;
        else if (tok == "+10") mm |= 1u << 4;
        i = j + 1;
    }
    return mm;
}

int main(int argc, char** argv) {
    int players = 2, target = 200; long my_total = 0;
    uint16_t my_hand = 0, my_mods = 0; bool my_sc = false;
    int action = 0;  // 0 none, 1 Freeze, 2 Flip Three
    std::vector<int> opp_total;
    std::vector<uint16_t> opp_hand;
    std::vector<char> opp_active;
    int seen[kNumValues] = {0};

    for (int i = 1; i < argc; ++i) {
        auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
        if (!strcmp(argv[i], "--players")) players = atoi(next());
        else if (!strcmp(argv[i], "--target")) target = atoi(next());
        else if (!strcmp(argv[i], "--my-hand")) my_hand = parse_hand(next());
        else if (!strcmp(argv[i], "--my-mods")) my_mods = parse_mods(next());
        else if (!strcmp(argv[i], "--my-sc")) my_sc = true;
        else if (!strcmp(argv[i], "--my-total")) my_total = atol(next());
        else if (!strcmp(argv[i], "--have-freeze")) action = 1;
        else if (!strcmp(argv[i], "--have-flip3")) action = 2;
        else if (!strcmp(argv[i], "--seen")) {
            std::string t(next()); size_t k = 0;
            while (k < t.size()) { size_t j = t.find(',', k); if (j == std::string::npos) j = t.size();
                int v = atoi(t.substr(k, j - k).c_str()); if (v >= 0 && v < kNumValues) seen[v]++; k = j + 1; }
        } else if (!strcmp(argv[i], "--opp")) {
            std::string t(next());
            size_t colon = t.find(':');
            opp_total.push_back(atoi(t.substr(0, colon).c_str()));
            opp_hand.push_back(colon == std::string::npos ? 0 : parse_hand(t.c_str() + colon + 1));
            opp_active.push_back(1);
        } else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }

    if ((int)opp_total.size() != players - 1) {
        fprintf(stderr, "error: --players %d needs %d --opp entries (got %d)\n",
                players, players - 1, (int)opp_total.size());
        return 2;
    }

    // remaining number-deck counts: fresh minus my hand, all visible opp hands, seen.
    int r0[kNumValues];
    for (int v = 0; v < kNumValues; ++v) {
        int used = ((my_hand >> v) & 1) + seen[v];
        for (uint16_t h : opp_hand) used += (h >> v) & 1;
        r0[v] = numberCount(v) - used;
        if (r0[v] < 0) r0[v] = 0;
    }

    printf("=== Flip 7 decision oracle (%d players, first to %d) ===\n", players, target);
    printf("you hold {");
    bool first = true;
    for (int v = 0; v < kNumValues; ++v) if ((my_hand >> v) & 1) { printf("%s%d", first ? "" : ",", v); first = false; }
    printf("} score-so-far=%d  total=%ld%s\n", fullScore(my_hand, my_mods), my_total, my_sc ? "  [Second Chance]" : "");
    if (players <= 3) printf("(win values: exact best-response grid; building/caching may take minutes the first time)\n");
    else              printf("(win values: Monte-Carlo estimate -- n>=4 is past the exact wall)\n");
    fflush(stdout);

    Oracle oracle(players, target);
    OracleDecision d = oracle.decide(my_hand, my_mods, my_sc, my_total, opp_total, opp_active,
                                     opp_hand, r0, action);

    printf("\n  WIN-OPTIMAL : %s   (win prob  hit=%.4f  stay=%.4f)\n",
           d.hit_winopt ? "HIT " : "STAY", d.w_hit, d.w_stay);
    printf("  score-optimal: %s   (E[round]  hit=%.3f  stay=%.3f)\n",
           d.hit_evopt ? "HIT " : "STAY", d.ev_hit, d.ev_stay);
    printf("  count-aware P(bust) if you hit now = %.4f", d.p_bust);
    {
        // fresh-deck P(bust) for contrast
        long Tf = kNumberDeckSize - maskPop(my_hand); long bf = 0;
        for (int v = 0; v < kNumValues; ++v) if ((my_hand >> v) & 1) bf += numberCount(v) - 1;
        printf("   (fresh-deck %.4f)\n", Tf > 0 ? (double)bf / Tf : 0.0);
    }
    if (d.hit_winopt != d.hit_evopt)
        printf("  note: win-optimal and score-optimal DISAGREE -- standings change the call.\n");

    if (action != 0) {
        const char* an = (action == 1) ? "Freeze" : "Flip Three";
        if (d.target == -2) printf("  %s: best NOT to use it now (no profitable target)\n", an);
        else if (d.target == -1) printf("  %s: aim at YOURSELF\n", an);
        else printf("  %s: aim at opponent #%d (total %d)  -> your win prob %.4f\n",
                    an, d.target + 1, opp_total[d.target], d.w_action);
    }
    return 0;
}
