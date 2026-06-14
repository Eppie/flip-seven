// PMU profiling of the hot kernels (via vendored third_party/perf.h).
//
// Reports IPC and per-operation cost plus the dominant stall sources (L1 / TLB
// misses, branch mispredicts) so we can see which kernels are memory-bound vs
// compute-bound. Programming the counters needs root:
//     make profile && sudo ./bin/profile
// Without root it still runs and prints wall-clock time only.
#include "perf.h"

#include "flip7_compete.hpp"
#include "flip7_core.hpp"
#include "flip7_dp.hpp"
#include "flip7_rng.hpp"
#include "flip7_sim.hpp"

#include <cstdint>
#include <cstdio>
#include <optional>
#include <utility>
#include <vector>

using namespace flip7;

// Run f under a counter set; print wall time, per-op cost, IPC and stall stats.
//   units = number of work items (rollouts / DP states), 0 to skip per-op stats.
template <class F>
static void profile(const char* label, perf::CounterSet set, double units, const char* unit, F&& f) {
    PerfMeasurement m = PerfMeasure(set, std::forward<F>(f));
    std::printf("  %-24s %8.1f ms", label, m.wall_ns / 1e6);
    if (units > 0) std::printf("  %6.1f ns/%s", m.wall_ns / units, unit);
    if (!m.valid) { std::printf("   [no PMU counters — run with sudo]\n"); return; }
    const auto cyc = m.Get(CYCLES);
    const auto ins = m.Get(INSTRUCTIONS);
    if (cyc && ins && *cyc) std::printf("  IPC=%.2f", (double)*ins / (double)*cyc);
    if (units > 0 && cyc) std::printf("  %.0f cyc/%s", *cyc / units, unit);
    if (units > 0 && ins) std::printf("  %.0f ins/%s", *ins / units, unit);
    if (auto v = m.Get(L1_LOAD_MISS); v && ins && *ins) std::printf("  L1/kI=%.1f", *v * 1000.0 / *ins);
    if (auto v = m.Get(DTLB_MISS);    v && ins && *ins) std::printf("  dTLB/kI=%.1f", *v * 1000.0 / *ins);
    if (auto v = m.Get(L2_TLB_MISS);  v && ins && *ins) std::printf("  L2TLB/kI=%.2f", *v * 1000.0 / *ins);
    if (auto v = m.Get(BRANCH_MISS); v) {
        if (auto br = m.Get(BRANCHES); br && *br) std::printf("  brMiss/kBr=%.1f", *v * 1000.0 / *br);
    }
    std::printf("\n");
}

int main() {
    std::printf("=== Flip 7 kernel profiling (perf.h %s) ===\n", PERF_VERSION);
    std::printf("Fixed counters (IPC) need root here too; run `sudo ./bin/profile`.\n");
    std::printf("Reads: IPC>3 compute-bound; high dTLB/L2TLB per kI => TLB-bound (random table access).\n\n");

    SolitaireTurnDP num_dp;  num_dp.optimal();
    SolitaireModDP  mod_dp;  mod_dp.optimal();
    SolitaireFullDP full_dp; full_dp.optimal();
    const double full_states = (double)full_dp.states_evaluated;
    const std::vector<double> D = round_pmf_numbers();
    const uint64_t N = 50'000'000ULL;
    const double Nd = (double)N;

    std::printf("[exact DP solves] cache profile (per state)\n");
    profile("numbers DP (8K)",      CACHE_PROFILE, 8192,        "st", [&]{ SolitaireTurnDP d; d.optimal(); });
    profile("modifiers DP (262K)",  CACHE_PROFILE, 262144,      "st", [&]{ SolitaireModDP d;  d.optimal(); });
    profile("+2ndChance DP (22M)",  CACHE_PROFILE, full_states, "st", [&]{ SolitaireFullDP d; d.optimal(); });

    // Direct attribution: each L2 TLB miss triggers a hardware page-table walk
    // (a multi-level dependent memory chase, ~100+ cycles). MMU_TABLE_WALK_DATA
    // counts them, so walks/state x walk-latency shows where the cycles go.
    std::printf("\n[TLB detail] hashed +2ndChance DP — page-walk attribution\n");
    {
        const auto WALK = perf::Counter::Named("MMU_TABLE_WALK_DATA");
        PerfMeasurement m = PerfMeasure(CYCLES | INSTRUCTIONS | DTLB_MISS | L2_TLB_MISS | WALK,
                                        [&]{ SolitaireFullDP d; d.optimal(); });
        if (!m.valid) { std::printf("  [no PMU counters — run with sudo]\n"); }
        else {
            const auto cyc = m.Get(CYCLES); const auto ins = m.Get(INSTRUCTIONS);
            const auto w = m.Get(WALK); const auto l2 = m.Get(L2_TLB_MISS);
            std::printf("  ");
            if (cyc) std::printf("%.0f cyc/state   ", *cyc / full_states);
            if (l2)  std::printf("L2TLB-miss/state=%.1f   ", *l2 / full_states);
            if (w)   std::printf("page-walks/state=%.1f", *w / full_states);
            if (w && ins && *ins) std::printf("  (walks/kI=%.1f)", *w * 1000.0 / *ins);
            std::printf("\n  => walks ~ L2 TLB misses; each is a multi-level page chase, so most of\n");
            std::printf("     the cyc/state is address translation, not arithmetic.\n");
        }
    }

    std::printf("\n[Monte-Carlo %lluM rollouts] cache profile (per rollout)\n", (unsigned long long)(N / 1'000'000));
    profile("MC numbers",       CACHE_PROFILE, Nd, "roll", [&]{ volatile auto r = monte_carlo_solitaire(num_dp, N, 1).mean; (void)r; });
    profile("MC +modifiers",    CACHE_PROFILE, Nd, "roll", [&]{ volatile auto r = monte_carlo_mod(mod_dp, N, 1).mean; (void)r; });
    profile("MC +2ndChance",    CACHE_PROFILE, Nd, "roll", [&]{ volatile auto r = monte_carlo_full(full_dp, N, 1).mean; (void)r; });

    std::printf("\n[branch profile]\n");
    profile("MC numbers",       BRANCH_PROFILE, Nd, "roll", [&]{ volatile auto r = monte_carlo_solitaire(num_dp, N, 2).mean; (void)r; });
    profile("MC +2ndChance",    BRANCH_PROFILE, Nd, "roll", [&]{ volatile auto r = monte_carlo_full(full_dp, N, 2).mean; (void)r; });
    profile("+2ndChance DP",    BRANCH_PROFILE, full_states, "st", [&]{ SolitaireFullDP d; d.optimal(); });

    std::printf("\n[PRNG] xoshiro256++ (per value)\n");
    {
        const uint64_t R = 1'000'000'000ULL;
        profile("next()", CACHE_PROFILE, (double)R, "val", [&]{
            Xoshiro256pp rng; rng.seed(7); volatile uint64_t s = 0;
            for (uint64_t i = 0; i < R; ++i) s ^= rng.next(); (void)s; });
        profile("bounded(94)", CACHE_PROFILE, (double)R, "val", [&]{
            Xoshiro256pp rng; rng.seed(7); volatile uint64_t s = 0;
            for (uint64_t i = 0; i < R; ++i) s ^= rng.bounded(94); (void)s; });
    }

    std::printf("\n[within-round solver] round_solve (the best-response inner kernel)\n");
    {
        init_round_tables();
        std::vector<double> g(kRoundScoreMax + 1), U(1 << kNumValues);
        for (int s = 0; s <= kRoundScoreMax; ++s) g[s] = s;  // reward = score
        const uint64_t K = 200'000;                          // ~ one slice of the Ch.4 grid
        profile("round_solve x200K", CACHE_PROFILE, (double)K, "solve", [&]{
            volatile double acc = 0;
            for (uint64_t i = 0; i < K; ++i) { g[0] = (double)(i & 7); acc += round_solve(g.data(), U.data(), nullptr); }
            (void)acc; });
    }

    std::printf("\n[competitive stages] cache profile\n");
    profile("round_pmf (numbers)",   CACHE_PROFILE, 0, "", [&]{ volatile auto x = round_pmf_numbers()[0]; (void)x; });
    profile("expected_rounds_200",   CACHE_PROFILE, 0, "", [&]{ volatile auto x = expected_rounds_to_target(D, 200); (void)x; });
    profile("win_prob_greedy(num)",  CACHE_PROFILE, 0, "", [&]{ volatile auto w = win_prob_greedy(D, 200)[0]; (void)w; });
    std::vector<double> Dall = load_pmf("data/round_pmf_all94.txt");
    if (!Dall.empty())
        profile("win_prob_greedy(all94)", CACHE_PROFILE, 0, "", [&]{ volatile auto w = win_prob_greedy(Dall, 200)[0]; (void)w; });

    return 0;
}
