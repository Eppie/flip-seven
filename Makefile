# Flip 7 — exact & competitive analysis.
#   make        build all chapter binaries + tests
#   make run    print the headline numbers (Chapter 1)
#   make test   assert the headline numbers (DP <-> MC agreement + regression)
#   make clean

CXX  ?= clang++
STD  := -std=c++20
OPT  := -O3 -flto -funroll-loops
WARN := -Wall -Wextra
INC  := -Iinclude

# Architecture auto-detect: pick the most specific -mcpu the compiler accepts.
ARCH := $(shell uname -m)
ifeq ($(ARCH),arm64)
  MCPU := $(shell for f in apple-m4 apple-m3 apple-m2 apple-m1 native; do \
            if echo 'int main(){}' | $(CXX) -x c++ -mcpu=$$f -o /dev/null - 2>/dev/null; \
            then echo -mcpu=$$f; break; fi; done)
else
  MCPU := -march=native
endif

CXXFLAGS := $(STD) $(OPT) $(WARN) $(INC) $(MCPU) -pthread

BIN  := bin
HDRS := $(wildcard include/*.hpp)

CH1  := $(BIN)/ch1_solitaire_turn
CH1B := $(BIN)/ch1b_modifiers_sc
CH2  := $(BIN)/ch2_separability
CH3  := $(BIN)/ch3_tails
CH4  := $(BIN)/ch4_competitive
NASH := $(BIN)/ch4_nash
CH5  := $(BIN)/ch5_actions
ALL  := $(BIN)/solitaire_all_cards
DECIDE := $(BIN)/decide
VENG := $(BIN)/vengeance
PROF := $(BIN)/profile
PROFB := $(BIN)/profile_blocked
PROFBI:= $(BIN)/profile_blocked_instr
PROFV := $(BIN)/profile_vengeance
PROFVI:= $(BIN)/profile_vengeance_instr
TST1 := $(BIN)/test_ch1
TST1B:= $(BIN)/test_ch1b
TST2 := $(BIN)/test_ch2
TST3 := $(BIN)/test_ch3
TST4 := $(BIN)/test_ch4
TST5 := $(BIN)/test_ch5
TSTN := $(BIN)/test_neon
TSTA := $(BIN)/test_all_cards
TSTO := $(BIN)/test_oracle
TSTV := $(BIN)/test_vengeance

all: $(CH1) $(CH1B) $(CH2) $(CH3) $(CH4) $(NASH) $(CH5) $(ALL) $(DECIDE) $(VENG) $(PROF) $(TST1) $(TST1B) $(TST2) $(TST3) $(TST4) $(TST5) $(TSTN) $(TSTA) $(TSTO) $(TSTV)
	@echo "built with: $(CXX) $(MCPU)"

$(BIN):
	@mkdir -p $(BIN)

$(CH1): ch1_solitaire_turn/main.cpp $(HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(CH1B): ch1b_modifiers_sc/main.cpp $(HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(CH2): ch2_separability/main.cpp $(HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(CH3): ch3_tails/main.cpp $(HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(CH4): ch4_competitive/main.cpp $(HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(NASH): ch4_nash/main.cpp $(HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(CH5): ch5_actions/main.cpp $(HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(ALL): solitaire_all_cards/main.cpp $(HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

# decide: the interactive decision oracle CLI.
$(DECIDE): oracle/main.cpp $(HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(TSTO): tests/test_oracle.cpp $(HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

# vengeance: Monte-Carlo summary for Flip 7: With a Vengeance.
$(VENG): vengeance/main.cpp $(HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(TSTV): tests/test_vengeance.cpp $(HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

# PMU profiler: vendored third_party/perf.h (Apple Silicon kperf, dlopen'd).
$(PROF): perf/profile.cpp $(HDRS) third_party/perf.h | $(BIN)
	$(CXX) $(CXXFLAGS) -Ithird_party -o $@ $<

# Blocked all-94 DP re-profile: one source, two builds (PMU / logical counts).
$(PROFB): perf/profile_blocked.cpp $(HDRS) third_party/perf.h | $(BIN)
	$(CXX) $(CXXFLAGS) -Ithird_party -o $@ $<
$(PROFBI): perf/profile_blocked.cpp $(HDRS) third_party/perf.h | $(BIN)
	$(CXX) $(CXXFLAGS) -Ithird_party -DFLIP7_BLK_INSTR -o $@ $<

# Vengeance MC profiler: PMU build + logical-instrumentation build.
$(PROFV): perf/profile_vengeance.cpp $(HDRS) third_party/perf.h | $(BIN)
	$(CXX) $(CXXFLAGS) -Ithird_party -o $@ $<
$(PROFVI): perf/profile_vengeance.cpp $(HDRS) third_party/perf.h | $(BIN)
	$(CXX) $(CXXFLAGS) -Ithird_party -DFLIP7_VENG_INSTR -o $@ $<

$(TST1): tests/test_ch1.cpp $(HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(TST1B): tests/test_ch1b.cpp $(HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(TST2): tests/test_ch2.cpp $(HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(TST3): tests/test_ch3.cpp $(HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(TST4): tests/test_ch4.cpp $(HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(TST5): tests/test_ch5.cpp $(HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(TSTN): tests/test_neon.cpp $(HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(TSTA): tests/test_all_cards.cpp $(HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

help:
	@echo "Flip 7 -- exact & competitive analysis. Common targets:"
	@echo "  make            build everything into bin/"
	@echo "  make test       build and run the full test suite"
	@echo "  make run        print the Ch.1-3 headline numbers"
	@echo "  make competitive / nash / actions   Ch.4-5 (2-player)"
	@echo "  make competitive-3p / -4p, nash-3p, actions-3p   N-player"
	@echo "  make decide     build+run the decision oracle (see ./bin/decide --help)"
	@echo "  make vengeance  Monte-Carlo summary for Flip 7: With a Vengeance"
	@echo "  make all-cards  the full 94-card solitaire DP (heavy)"
	@echo "  make profile    PMU profiling of the hot kernels (sudo for counters)"
	@echo "  make clean"

# Fast headline numbers + tests (Ch.1 progression, Ch.2 strategy, Ch.3 tails,
# Ch.4 A-C win probabilities).
run: $(CH1) $(CH1B) $(CH2) $(CH3)
	./$(CH1)
	@echo
	./$(CH1B)
	@echo
	./$(CH2)
	@echo
	./$(CH3)

test: $(TST1) $(TST1B) $(TST2) $(TST3) $(TST4) $(TST5) $(TSTN) $(TSTO)
	./$(TST1)
	@echo
	./$(TST1B)
	@echo
	./$(TST2)
	@echo
	./$(TST3)
	@echo
	./$(TST4)
	@echo
	./$(TST5)
	@echo
	./$(TSTN)
	@echo
	./$(TSTO)
	@echo
	./$(TSTV)

# Decision oracle CLI. Build it, then run e.g.:
#   ./bin/decide --players 3 --my-hand 3,7,9 --my-total 110 \
#                --opp 95:5,8 --opp 130:2,3,4 --seen 12,12 --have-freeze
# (n<=3 builds + caches the exact win grid to data/ on first use.)
decide: $(DECIDE)
	./$(DECIDE) --players 2 --my-hand 5,9,12 --my-total 150 --opp 168 --seen 11,12

# Flip 7: With a Vengeance -- faithful-rules Monte-Carlo summary.
vengeance: $(VENG)
	./$(VENG)

# Opt-in heavy solves (out of the fast loop).
#   competitive: Ch.4 best-response grid (~9 s)
#   all-cards:   full 94-card solitaire DP (~1e9 states, ~30 GB, minutes)
competitive: $(CH4)
	./$(CH4)

# Symmetric Nash equilibrium via fictitious self-play (~3-4 min).
nash: $(NASH)
	./$(NASH)

# Chapter 5: adversarial action-card targeting (Freeze / Flip Three), ~seconds.
actions: $(CH5)
	./$(CH5)

# N-player generalizations (the 2-player chapters above are unchanged). The exact
# win-probability DP is parallelized across cores; it is exact for n<=3 and
# Monte-Carlo beyond. Run the binaries directly to pick n / target, e.g.:
#   ./bin/ch4_competitive N [target]     (greedy + best-response, exact for N<=3)
#   ./bin/ch4_nash players=N [target]    (symmetric value 1/N, MC-confirmed)
#   ./bin/ch5_actions players=N [target] (real-rules targeting vs an N-field)
# competitive-3p runs the full exact 3-player first-to-200 (greedy ~1 min,
# best-response several min, both threaded).
competitive-3p: $(CH4)
	./$(CH4) 3
competitive-4p: $(CH4)
	./$(CH4) 4
nash-3p: $(NASH)
	./$(NASH) players=3
actions-3p: $(CH5)
	./$(CH5) players=3

# PMU profiling of the hot kernels (needs root to program counters).
profile: $(PROF)
	@echo "run 'sudo ./$(PROF)' for PMU counters; without root only wall-time prints."
	./$(PROF)

# Re-profile the >60 s blocked all-94 DP. Build both, then:
#   sudo ./bin/profile_blocked cachewalk   (per-state cyc/IPC/misses/page-walks)
#   sudo ./bin/profile_blocked branch      (branch mispredicts)
#   ./bin/profile_blocked_instr            (logical memo fan-in / same-base split)
profile-blocked: $(PROFB) $(PROFBI)
	@echo "PMU (sudo): sudo ./$(PROFB) cachewalk | branch | exec"
	@echo "logical   : ./$(PROFBI)"

# Vengeance Monte-Carlo profiling: runs the logical build now, then prints the
# sudo command for the hardware counters.
profile-vengeance: $(PROFV) $(PROFVI)
	./$(PROFVI)
	@echo
	@echo "hardware counters (IPC / cache / branch) need root:"
	@echo "    sudo ./$(PROFV)"

all-cards: $(ALL)
	./$(ALL)

test-all-cards: $(TSTA)
	./$(TSTA)

clean:
	rm -rf $(BIN)

.PHONY: all help run test decide vengeance competitive competitive-3p competitive-4p nash nash-3p actions actions-3p profile profile-blocked profile-vengeance all-cards test-all-cards clean
