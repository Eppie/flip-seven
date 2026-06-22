// Re-profile of the all-94 blocked DP (SolitaireAllDPBlocked) — the >60 s solve.
//
// Two builds from this one source:
//   bin/profile_blocked        (PMU mode)     — counters on the UNMODIFIED solve.
//   bin/profile_blocked_instr  (-DFLIP7_BLK_INSTR) — logical counts (no PMU).
//
// PMU mode programs the counters (needs root) around ONE optimal() solve. The
// table (~11 GB) is built BEFORE the measured region so the value-array fill does
// not pollute the per-state numbers. Pick a counter set on the command line:
//   sudo ./bin/profile_blocked cachewalk   (cyc, IPC, L1/dTLB/L2TLB miss, page-walks)
//   sudo ./bin/profile_blocked cache        (same, no walks — fallback if cachewalk
//                                            can't program 4 configurable counters)
//   sudo ./bin/profile_blocked walk         (page-walk attribution, known-good combo)
//   sudo ./bin/profile_blocked branch       (branch mispredicts)
//   sudo ./bin/profile_blocked exec         (uops/cycle, retire — compute-bound check)
//
// Logical mode just runs once and prints the memo fan-in and same-base edge split.
#include "perf.h"

#include "flip7_core.hpp"
#include "flip7_dp.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>

using namespace flip7;
using clk = std::chrono::steady_clock;
static double secs(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

#ifndef FLIP7_BLK_INSTR
// ---- PMU mode -------------------------------------------------------------
using perf::Counter;

static void report(const char* label, const PerfMeasurement& m, double states) {
    std::printf("\n[%s]  %.2f s solve   %.0f M states   %.1f ns/state\n",
                label, m.wall_ns / 1e9, states / 1e6, m.wall_ns / states);
    if (!m.valid) { std::printf("  [no PMU counters — run with sudo]\n"); return; }
    const auto cyc = m.Get(CYCLES);
    const auto ins = m.Get(INSTRUCTIONS);
    if (cyc) std::printf("  %.1f cyc/state", *cyc / states);
    if (ins) std::printf("   %.1f ins/state", *ins / states);
    if (cyc && ins && *cyc) std::printf("   IPC=%.2f", (double)*ins / (double)*cyc);
    std::printf("\n");
    if (auto v = m.Get(L1_LOAD_MISS)) std::printf("  L1d-miss/state=%.2f   (L1/kI=%.1f)\n",
                                                  *v / states, ins && *ins ? *v * 1000.0 / *ins : 0);
    if (auto v = m.Get(DTLB_MISS))    std::printf("  dTLB-miss/state=%.2f\n", *v / states);
    if (auto v = m.Get(L2_TLB_MISS))  std::printf("  L2TLB-miss/state=%.2f\n", *v / states);
    if (auto v = m.Get(Counter::Named("MMU_TABLE_WALK_DATA")))
        std::printf("  page-walks/state=%.2f   <- each is a multi-level dependent chase\n", *v / states);
    if (auto v = m.Get(BRANCHES)) {
        std::printf("  branches/state=%.1f", *v / states);
        if (auto mb = m.Get(BRANCH_MISS); mb && *v)
            std::printf("   miss/state=%.2f   brMiss/kBr=%.1f", *mb / states, *mb * 1000.0 / *v);
        std::printf("\n");
        if (auto cm = m.Get(BRANCH_COND_MISS)) std::printf("  cond-miss/state=%.2f\n", *cm / states);
        if (auto im = m.Get(BRANCH_INDIR_MISS)) std::printf("  indir-miss/state=%.3f\n", *im / states);
    }
    if (auto v = m.Get(RETIRE_UOP)) {
        std::printf("  uops/state=%.1f", *v / states);
        if (cyc && *cyc) std::printf("   uops/cyc=%.2f", (double)*v / (double)*cyc);
        std::printf("\n");
    }
}

int main(int argc, char** argv) {
    const char* which = argc > 1 ? argv[1] : "cachewalk";
    std::printf("=== blocked all-94 DP re-profile (perf.h %s) — set='%s' ===\n", PERF_VERSION, which);

    perf::CounterSet set;
    if      (!std::strcmp(which, "cachewalk"))
        set = CYCLES | INSTRUCTIONS | L1_LOAD_MISS | DTLB_MISS | L2_TLB_MISS
              | Counter::Named("MMU_TABLE_WALK_DATA");
    else if (!std::strcmp(which, "cache"))  set = CACHE_PROFILE;
    else if (!std::strcmp(which, "walk"))
        set = CYCLES | INSTRUCTIONS | DTLB_MISS | L2_TLB_MISS | Counter::Named("MMU_TABLE_WALK_DATA");
    else if (!std::strcmp(which, "branch")) set = BRANCH_PROFILE;
    else if (!std::strcmp(which, "exec"))   set = EXECUTION_PROFILE;
    else { std::printf("unknown set '%s'\n", which); return 1; }

    auto tb = clk::now();
    SolitaireAllDPBlocked dp;   // ~11 GB; built (and value-array filled) BEFORE measuring
    std::printf("(arena built in %.1f s: %d modes/block, %.2fM-slot base hash)\n",
                secs(tb, clk::now()), dp.n_modes, (double)dp.base_cap / 1e6);
    std::fflush(stdout);

    double opt = 0.0;
    PerfMeasurement m = PerfMeasure(set, [&] { opt = dp.optimal(); });
    report(which, m, (double)dp.states_evaluated);
    std::printf("  E[score]=%.10f   states=%ld   base blocks=%.2fM   load=%.3f\n",
                opt, dp.states_evaluated, dp.next_block / 1e6, dp.load_factor());
    return 0;
}

#else
// ---- logical-counts mode (-DFLIP7_BLK_INSTR) ------------------------------
int main() {
    std::printf("=== blocked all-94 DP — logical instrumentation ===\n");
    SolitaireAllDPBlocked dp;
    auto t0 = clk::now();
    const double opt = dp.optimal();
    const double sec = secs(t0, clk::now());

    const double S  = (double)dp.states_evaluated;   // computed states
    const double C  = (double)dp.solve_calls;        // total solve() invocations
    const double H  = (double)dp.memo_hits;          // early-return invocations
    const double sb = (double)(dp.sb_f3 + dp.sb_fz);

    std::printf("\nE[score]=%.10f   wall=%.1f s   (%.1f M states/s)\n", opt, sec, S / sec / 1e6);
    std::printf("computed states            = %ld\n", dp.states_evaluated);
    std::printf("solve() calls (= base probes) = %ld   (%.2f calls / computed state)\n",
                dp.solve_calls, C / S);
    std::printf("  memo hits (early return)  = %ld   (%.1f%% of calls)\n", dp.memo_hits, 100.0 * H / C);
    std::printf("  -> every call pays 1 base-hash probe + 1 value-array index BEFORE\n");
    std::printf("     it knows it's a hit; that lookup path runs %.2fx more than the compute path.\n", C / S);
    std::printf("base-probe collisions      = %ld   (%.3f extra slots / probe; load=%.3f)\n",
                dp.probe_steps, (double)dp.probe_steps / C, dp.load_factor());
    std::printf("same-base child edges      = %ld   (f3=%ld, fz=%ld)\n",
                dp.sb_f3 + dp.sb_fz, dp.sb_f3, dp.sb_fz);
    std::printf("  -> %.1f%% of all solve() calls arrive via a same-base edge; their base\n", 100.0 * sb / C);
    std::printf("     probe is redundant (pass-the-block-down ceiling).\n");
    const double cross = (C - 1.0) - sb;
    std::printf("cross-base child edges     = %.0f   (%.1f%% of calls — need a genuine new base)\n",
                cross, 100.0 * cross / C);
    return 0;
}
#endif
