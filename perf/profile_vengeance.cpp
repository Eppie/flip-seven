// PMU + logical profiling of the Flip 7: With a Vengeance Monte-Carlo engine.
//
//   make profile-vengeance
//   ./bin/profile_vengeance_instr        # logical per-game work attribution (no root)
//   sudo ./bin/profile_vengeance         # IPC / cache / branch (needs root)
//
// The engine is profiled SINGLE-THREADED (one VengeanceGame, per-game-seeded) so the
// thread-scoped counters capture all the work. The adversarial field is profiled
// because it exercises the heaviest path (best_swap's O(cards^2) damage search).
#include "perf.h"

#include "flip7_rng.hpp"
#include "flip7_vengeance.hpp"

#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

using namespace flip7;

template <class F>
static void profile(const char* label, perf::CounterSet set, double units, const char* unit, F&& f) {
    PerfMeasurement m = PerfMeasure(set, std::forward<F>(f));
    std::printf("  %-26s %8.1f ms", label, m.wall_ns / 1e6);
    if (units > 0) std::printf("  %7.0f ns/%s", m.wall_ns / units, unit);
    if (!m.valid) { std::printf("   [no PMU — run with sudo]\n"); return; }
    const auto cyc = m.Get(CYCLES); const auto ins = m.Get(INSTRUCTIONS);
    if (cyc && ins && *cyc) std::printf("  IPC=%.2f", (double)*ins / (double)*cyc);
    if (units > 0 && cyc) std::printf("  %.0f cyc/%s", *cyc / units, unit);
    if (units > 0 && ins) std::printf("  %.0f ins/%s", *ins / units, unit);
    if (auto v = m.Get(L1_LOAD_MISS); v && ins && *ins) std::printf("  L1/kI=%.1f", *v * 1000.0 / *ins);
    if (auto v = m.Get(DTLB_MISS);    v && ins && *ins) std::printf("  dTLB/kI=%.1f", *v * 1000.0 / *ins);
    if (auto v = m.Get(BRANCH_MISS); v) {
        if (auto br = m.Get(BRANCHES); br && *br) std::printf("  brMiss/kBr=%.1f", *v * 1000.0 / *br);
    }
    std::printf("\n");
}

// One single-threaded batch of games (per-game seeded, matching the engine).
static long run_games(const std::vector<int>& tps, uint64_t G, uint64_t seed, bool dump) {
    Xoshiro256pp rng;
    VengeanceGame game(rng, tps);
    long acc = 0;
    for (uint64_t g = 0; g < G; ++g) {
        uint64_t sm = seed + g * 0x9E3779B97F4A7C15ULL;
        rng.seed(splitmix64(sm));
        acc += game.play_game(200);
    }
#ifdef FLIP7_VENG_INSTR
    if (dump) game.instr_dump();
#else
    (void)dump;
#endif
    return acc;
}

int main() {
    std::printf("=== Flip 7: With a Vengeance — MC profiling (perf.h %s) ===\n", PERF_VERSION);
    const uint64_t G = 200'000;

#ifdef FLIP7_VENG_INSTR
    std::printf("[logical instrumentation — per-game work attribution; timing is PERTURBED]\n\n");
    for (int n : {2, 3, 4, 6}) {
        std::printf("n=%d (adversarial field):\n", n);
        run_games(std::vector<int>(n, VTP_ADVERSARIAL), G, 0x1234u + n, true);
        std::printf("\n");
    }
#else
    std::printf("[PMU build — wall time always; IPC/cache/branch need `sudo`]\n");
    std::printf("Reads: IPC>3 compute-bound; high L1/kI or dTLB/kI => memory-bound.\n\n");
    std::printf("[per game — cache profile, adversarial field]\n");
    for (int n : {2, 3, 4, 6}) {
        char lbl[48]; std::snprintf(lbl, sizeof lbl, "vengeance n=%d", n);
        profile(lbl, CACHE_PROFILE, (double)G, "game",
                [&] { volatile long a = run_games(std::vector<int>(n, VTP_ADVERSARIAL), G, 0x1234u + n, false); (void)a; });
    }
    std::printf("\n[per game — branch profile]\n");
    for (int n : {2, 4}) {
        char lbl[48]; std::snprintf(lbl, sizeof lbl, "vengeance n=%d", n);
        profile(lbl, BRANCH_PROFILE, (double)G, "game",
                [&] { volatile long a = run_games(std::vector<int>(n, VTP_ADVERSARIAL), G, 0x5678u + n, false); (void)a; });
    }
    // a RANDOM field (no best_swap damage search) for contrast with the adversarial path
    std::printf("\n[per game — cache profile, RANDOM field (lighter targeting)]\n");
    for (int n : {2, 4}) {
        char lbl[48]; std::snprintf(lbl, sizeof lbl, "vengeance n=%d rand", n);
        profile(lbl, CACHE_PROFILE, (double)G, "game",
                [&] { volatile long a = run_games(std::vector<int>(n, VTP_RANDOM), G, 0x9abcu + n, false); (void)a; });
    }
#endif
    return 0;
}
