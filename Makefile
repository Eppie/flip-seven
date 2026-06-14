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

CXXFLAGS := $(STD) $(OPT) $(WARN) $(INC) $(MCPU)

BIN  := bin
HDRS := $(wildcard include/*.hpp)

CH1  := $(BIN)/ch1_solitaire_turn
CH1B := $(BIN)/ch1b_modifiers_sc
CH2  := $(BIN)/ch2_separability
CH3  := $(BIN)/ch3_tails
CH4  := $(BIN)/ch4_competitive
ALL  := $(BIN)/solitaire_all_cards
PROF := $(BIN)/profile
TST1 := $(BIN)/test_ch1
TST1B:= $(BIN)/test_ch1b
TST2 := $(BIN)/test_ch2
TST3 := $(BIN)/test_ch3
TST4 := $(BIN)/test_ch4
TSTA := $(BIN)/test_all_cards

all: $(CH1) $(CH1B) $(CH2) $(CH3) $(CH4) $(ALL) $(PROF) $(TST1) $(TST1B) $(TST2) $(TST3) $(TST4) $(TSTA)
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

$(ALL): solitaire_all_cards/main.cpp $(HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

# PMU profiler: vendored third_party/perf.h (Apple Silicon kperf, dlopen'd).
$(PROF): perf/profile.cpp $(HDRS) third_party/perf.h | $(BIN)
	$(CXX) $(CXXFLAGS) -Ithird_party -o $@ $<

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

$(TSTA): tests/test_all_cards.cpp $(HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

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

test: $(TST1) $(TST1B) $(TST2) $(TST3) $(TST4)
	./$(TST1)
	@echo
	./$(TST1B)
	@echo
	./$(TST2)
	@echo
	./$(TST3)
	@echo
	./$(TST4)

# Opt-in heavy solves (out of the fast loop).
#   competitive: Ch.4 best-response grid (~9 s)
#   all-cards:   full 94-card solitaire DP (~1e9 states, ~30 GB, minutes)
competitive: $(CH4)
	./$(CH4)

# PMU profiling of the hot kernels (needs root to program counters).
profile: $(PROF)
	@echo "run 'sudo ./$(PROF)' for PMU counters; without root only wall-time prints."
	./$(PROF)

all-cards: $(ALL)
	./$(ALL)

test-all-cards: $(TSTA)
	./$(TSTA)

clean:
	rm -rf $(BIN)

.PHONY: all run test competitive profile all-cards test-all-cards clean
