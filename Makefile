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
TST1 := $(BIN)/test_ch1
TST1B:= $(BIN)/test_ch1b

all: $(CH1) $(CH1B) $(TST1) $(TST1B)
	@echo "built with: $(CXX) $(MCPU)"

$(BIN):
	@mkdir -p $(BIN)

$(CH1): ch1_solitaire_turn/main.cpp $(HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(CH1B): ch1b_modifiers_sc/main.cpp $(HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(TST1): tests/test_ch1.cpp $(HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(TST1B): tests/test_ch1b.cpp $(HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

run: $(CH1) $(CH1B)
	./$(CH1)
	@echo
	./$(CH1B)

test: $(TST1) $(TST1B)
	./$(TST1)
	@echo
	./$(TST1B)

clean:
	rm -rf $(BIN)

.PHONY: all run test clean
