// Chapter 2: what IS the optimal strategy, and is it a simple threshold?
//
// Works in the interpretable numbers-only game (the modifier/Second-Chance/
// action-card variants shift the boundary but keep the same shape). Evaluates
// the exact optimal policy and the best simple heuristics, and quantifies how
// far each heuristic is from optimal -- i.e. how non-separable optimal play is.
#include "flip7_core.hpp"
#include "flip7_dp.hpp"

#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <string>
#include <vector>

using namespace flip7;

static double pbust(uint16_t S) {                 // P(next flip busts) at held set S
    const int p = maskPop(S);
    const int T = kNumberDeckSize - p;
    int b = 0;
    for (int v = 0; v < kNumValues; ++v)
        if (S & (1u << v)) b += numberCount(v) - 1;
    return (double)b / (double)T;
}

// Expected round score of an arbitrary Hit/Stay policy (numbers-only), via the
// same forward reach-probability pass used to verify the DP.
template <class Hit>
static double evaluate(Hit should_hit) {
    std::vector<double> reach(1 << kNumValues, 0.0);
    reach[0] = 1.0;
    double e = 0.0;
    for (int p = 0; p < kFlip7Target; ++p) {
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
    }
    return e;
}

int main() {
    SolitaireTurnDP dp;
    const double opt = dp.optimal();
    printf("=== Flip 7 - Chapter 2: the optimal strategy & separability (numbers only) ===\n\n");
    printf("Exact optimal E[score] = %.6f\n", opt);
    printf("(forward-eval of the DP policy = %.6f, sanity)\n\n",
           evaluate([&](uint16_t S){ return dp.hit[S]; }));

    // --- Heuristic 1: stop after k unique cards ---
    printf("--- Heuristic: \"hit until you hold k unique numbers, then stay\" ---\n");
    double best_k_ev = 0; int best_k = 0;
    for (int k = 3; k <= 7; ++k) {
        const double ev = evaluate([k](uint16_t S){ return maskPop(S) < k; });
        printf("  k=%d : E[score] = %7.4f   (%.1f%% of optimal)\n", k, ev, 100.0 * ev / opt);
        if (ev > best_k_ev) { best_k_ev = ev; best_k = k; }
    }
    printf("  best: k=%d at %.4f  (gap to optimal: %.4f, %.2f%%)\n\n",
           best_k, best_k_ev, opt - best_k_ev, 100.0 * (opt - best_k_ev) / opt);

    // --- Heuristic 2: stop once banked score >= T (full sweep) ---
    printf("--- Heuristic: \"hit until your banked sum >= T, then stay\" (which T?) ---\n");
    double best_T_ev = 0; int best_T = 0;
    double ev_T[80] = {0};
    for (int T = 1; T <= 60; ++T) {
        ev_T[T] = evaluate([T](uint16_t S){ return maskSum(S) < T; });
        if (ev_T[T] > best_T_ev) { best_T_ev = ev_T[T]; best_T = T; }
    }
    printf("    T  : E[score]  (%% of optimal)\n");
    for (int T = 1; T <= 40; ++T)
        printf("    %2d : %7.4f   %5.1f%%%s\n", T, ev_T[T], 100.0 * ev_T[T] / opt,
               T == best_T ? "   <- best" : "");
    printf("  best: T=%d at %.4f  (gap to optimal: %.4f, %.2f%%)\n", best_T, best_T_ev,
           opt - best_T_ev, 100.0 * (opt - best_T_ev) / opt);
    printf("  (broad plateau: any T in 21..25 stays above ~99%%; 22..24 within ~0.6%% of optimal.)\n\n");

    // --- Heuristic 3: stop once P(bust) >= theta ---
    printf("--- Heuristic: \"hit while P(bust on next flip) < theta\" ---\n");
    double best_th_ev = 0, best_th = 0;
    for (int i = 1; i < 100; ++i) {
        const double th = i / 100.0;
        const double ev = evaluate([th](uint16_t S){ return pbust(S) < th; });
        if (ev > best_th_ev) { best_th_ev = ev; best_th = th; }
    }
    printf("  best: theta=%.2f at %.4f  (gap to optimal: %.4f, %.2f%%)\n\n",
           best_th, best_th_ev, opt - best_th_ev, 100.0 * (opt - best_th_ev) / opt);

    // --- Why no fixed bust-probability threshold is optimal (non-separability) ---
    printf("--- Non-separability: the bust-risk you should accept depends on your score ---\n");
    double max_hit_pb = 0.0, min_stay_pb = 1.0;
    uint16_t hit_witness = 0, stay_witness = 0;
    for (uint32_t S = 0; S < (1u << kNumValues); ++S) {
        if (maskPop((uint16_t)S) >= kFlip7Target) continue;
        const double pb = pbust((uint16_t)S);
        if (dp.hit[S]) { if (pb > max_hit_pb) { max_hit_pb = pb; hit_witness = (uint16_t)S; } }
        else           { if (pb < min_stay_pb) { min_stay_pb = pb; stay_witness = (uint16_t)S; } }
    }
    auto bits = [](uint16_t m){ static char b[64]; int n=0; b[n++]='{';
        for(int v=0;v<kNumValues;++v) if(m&(1u<<v)) n+=snprintf(b+n,8,"%s%d", n>1?",":"", v);
        b[n++]='}'; b[n]=0; return std::string(b); };
    auto show = [&](uint16_t m){
        printf("    %-16s cards=%d  sum=%2d  P(bust)=%.3f  ->  %s\n",
               bits(m).c_str(), maskPop(m), maskSum(m), pbust(m), dp.hit[m] ? "HIT" : "STAY");
    };
    printf("  highest P(bust) the optimal policy still HITS:\n");
    show(hit_witness);
    printf("  lowest P(bust) at which it already STAYS:\n");
    show(stay_witness);
    printf("  these overlap (%.3f > %.3f), so no single P(bust) cutoff is exactly optimal.\n",
           max_hit_pb, min_stay_pb);
    printf("  and count is no better -- equal hand size, opposite call:\n");
    auto mk = [](std::initializer_list<int> vs){ uint16_t m=0; for(int v:vs) m|=(uint16_t)(1u<<v); return m; };
    show(mk({1,2,3}));
    show(mk({10,11,12}));

    printf("\n--- one-line summary ---\n");
    printf("  Keep flipping until your bust risk hits ~%.0f%%, then stop. That single rule\n", best_th * 100);
    printf("  earns %.1f%% of optimal. It is NOT \"stop at k cards\" (best k only %.1f%%).\n",
           100.0 * best_th_ev / opt, 100.0 * best_k_ev / opt);
    printf("  The last ~%.1f%% needs the exact hand (chase the +15 at 6 cards, mind exact values).\n",
           100.0 * (opt - best_th_ev) / opt);
    return 0;
}
