// decide -- the Flip 7 decision oracle CLI.
//
// Hand it the live situation and it recommends the best action. See
// flip7_oracle.hpp for the model and its honest bounds; run `decide --help`.
#include "flip7_oracle.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace flip7;

static void usage(FILE* f) {
    std::fprintf(f,
        "decide -- Flip 7 decision oracle\n"
        "\n"
        "Usage:\n"
        "  decide --players N [--target T] \\\n"
        "         --my-hand 3,7,9 [--my-mods x2,+4] [--my-sc] --my-total V \\\n"
        "         --opp TOTAL[:held-numbers] ...   (one --opp per opponent) \\\n"
        "         [--seen 12,12,11] [--have-freeze | --have-flip3]\n"
        "\n"
        "Options:\n"
        "  --players N      number of players (>= 2)\n"
        "  --target  T      first-to-T match target (default 200)\n"
        "  --my-hand a,b,c  the number cards you hold (values 0..12)\n"
        "  --my-mods ...    modifiers held: x2, +2, +4, +6, +8, +10\n"
        "  --my-sc          you hold a Second Chance\n"
        "  --my-total V     your current match total (0 <= V < target)\n"
        "  --opp T[:h,h]    an opponent's match total, optionally their held numbers\n"
        "                   (give exactly N-1 --opp; held numbers enable targeting)\n"
        "  --seen ...       number cards already gone from the shoe (card counting)\n"
        "  --have-freeze    you hold a Freeze (asks who to target)\n"
        "  --have-flip3     you hold a Flip Three (asks who to target)\n"
        "  -h, --help       this message\n"
        "\n"
        "Example:\n"
        "  decide --players 3 --my-hand 3,7,9 --my-total 110 \\\n"
        "         --opp 95:5,8 --opp 130:2,3,4 --seen 12,12 --have-freeze\n"
        "\n"
        "Win values are exact for <=3 players (the n=3 grid is built and cached to\n"
        "data/ on first use, a few minutes) and Monte-Carlo for >=4. This is a\n"
        "one-step best response to a greedy field, not the full-game equilibrium.\n");
}

// Parse "3,7,9" into a 13-bit number mask; warn on out-of-range / non-numeric tokens.
static uint16_t parse_hand(const char* s, const char* what) {
    uint16_t m = 0;
    if (!s || !*s) return 0;
    std::string t(s);
    size_t i = 0;
    while (i < t.size()) {
        size_t j = t.find(',', i);
        if (j == std::string::npos) j = t.size();
        if (j > i) {
            std::string tok = t.substr(i, j - i);
            int v = atoi(tok.c_str());
            if (v >= 0 && v < kNumValues) {
                if (m & (1u << v)) std::fprintf(stderr, "warning: %s lists %d twice (a duplicate would bust)\n", what, v);
                m |= (uint16_t)(1u << v);
            } else {
                std::fprintf(stderr, "warning: ignoring out-of-range %s value '%s' (want 0..12)\n", what, tok.c_str());
            }
        }
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
        if      (tok == "x2" || tok == "X2") mm |= (uint16_t)(1u << kX2Bit);
        else if (tok == "+2")  mm |= 1u << 0;
        else if (tok == "+4")  mm |= 1u << 1;
        else if (tok == "+6")  mm |= 1u << 2;
        else if (tok == "+8")  mm |= 1u << 3;
        else if (tok == "+10") mm |= 1u << 4;
        else if (!tok.empty()) std::fprintf(stderr, "warning: ignoring unknown modifier '%s' (want x2,+2,+4,+6,+8,+10)\n", tok.c_str());
        i = j + 1;
    }
    return mm;
}

static void die(const char* msg) { std::fprintf(stderr, "error: %s\n\n", msg); usage(stderr); }

int main(int argc, char** argv) {
    int players = 2, target = 200; long my_total = -1;
    uint16_t my_hand = 0, my_mods = 0; bool my_sc = false, have_total = false;
    int action = 0;  // 0 none, 1 Freeze, 2 Flip Three
    std::vector<int> opp_total;
    std::vector<uint16_t> opp_hand;
    std::vector<char> opp_active;
    int seen[kNumValues] = {0};

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        auto val = [&](void) -> const char* {
            if (i + 1 >= argc) { die((std::string("missing value for ") + a).c_str()); std::exit(2); }
            return argv[++i];
        };
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(stdout); return 0; }
        else if (!strcmp(a, "--players")) players = atoi(val());
        else if (!strcmp(a, "--target")) target = atoi(val());
        else if (!strcmp(a, "--my-hand")) my_hand = parse_hand(val(), "--my-hand");
        else if (!strcmp(a, "--my-mods")) my_mods = parse_mods(val());
        else if (!strcmp(a, "--my-sc")) my_sc = true;
        else if (!strcmp(a, "--my-total")) { my_total = atol(val()); have_total = true; }
        else if (!strcmp(a, "--have-freeze")) action = 1;
        else if (!strcmp(a, "--have-flip3")) action = 2;
        else if (!strcmp(a, "--seen")) {
            std::string t(val()); size_t k = 0;
            while (k < t.size()) { size_t j = t.find(',', k); if (j == std::string::npos) j = t.size();
                int v = atoi(t.substr(k, j - k).c_str());
                if (v >= 0 && v < kNumValues) seen[v]++;
                else std::fprintf(stderr, "warning: ignoring out-of-range --seen value (want 0..12)\n");
                k = j + 1; }
        } else if (!strcmp(a, "--opp")) {
            std::string t(val());
            size_t colon = t.find(':');
            opp_total.push_back(atoi(t.substr(0, colon).c_str()));
            opp_hand.push_back(colon == std::string::npos ? 0 : parse_hand(t.c_str() + colon + 1, "--opp hand"));
            opp_active.push_back(1);
        } else { die((std::string("unknown argument '") + a + "'").c_str()); return 2; }
    }

    // ---- validation ----
    if (players < 2)               { die("--players must be >= 2"); return 2; }
    if (target <= 0)               { die("--target must be > 0"); return 2; }
    if (!have_total)               { die("--my-total is required"); return 2; }
    if (my_total < 0 || my_total >= target) { die("--my-total must be in [0, target)"); return 2; }
    if ((int)opp_total.size() != players - 1) {
        std::fprintf(stderr, "error: --players %d needs %d --opp entries (got %d)\n\n",
                     players, players - 1, (int)opp_total.size());
        usage(stderr); return 2;
    }
    for (int ot : opp_total)
        if (ot < 0 || ot >= target) { die("each --opp total must be in [0, target)"); return 2; }

    // remaining number-deck counts: fresh minus my hand, all visible opp hands, seen.
    int r0[kNumValues];
    for (int v = 0; v < kNumValues; ++v) {
        int used = ((my_hand >> v) & 1) + seen[v];
        for (uint16_t h : opp_hand) used += (h >> v) & 1;
        r0[v] = numberCount(v) - used;
        if (r0[v] < 0) { std::fprintf(stderr, "warning: more %d's accounted for than exist; clamping\n", v); r0[v] = 0; }
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
        long Tf = kNumberDeckSize - maskPop(my_hand); long bf = 0;
        for (int v = 0; v < kNumValues; ++v) if ((my_hand >> v) & 1) bf += numberCount(v) - 1;
        printf("   (fresh-deck %.4f)\n", Tf > 0 ? (double)bf / Tf : 0.0);
    }
    if (d.hit_winopt != d.hit_evopt)
        printf("  note: win-optimal and score-optimal DISAGREE -- standings change the call.\n");

    if (action != 0) {
        const char* an = (action == 1) ? "Freeze" : "Flip Three";
        if (d.target == -2) printf("  %s: best NOT to use it on an opponent now (no profitable target)\n", an);
        else printf("  %s: aim at opponent #%d (total %d)  -> your win prob %.4f\n",
                    an, d.target + 1, opp_total[d.target], d.w_action);
    }
    return 0;
}
