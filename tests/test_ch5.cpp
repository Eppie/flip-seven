// test_ch5 — assert the Chapter 5 action-card results (numbers-only exact layer).
//
// The exact Flip-Three / Freeze quantities must match an independent Monte-Carlo
// before they are trusted; the win-prob targeting DP must agree with its own MC.
// (Part C, the full 94-card duel, is a faithful-rules MC with no DP counterpart;
// its symmetric-matchup sanities are checked by ch5_actions itself.)
#include "flip7_actions.hpp"
#include "flip7_compete.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace flip7;

static int failures = 0;
static void check(bool ok, const char* what) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}
static bool close(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

int main() {
    printf("Chapter 5 tests\n");
    init_round_tables();
    constexpr int N = 1 << kNumValues;

    std::vector<double> U;
    std::vector<uint8_t> hit;
    numbers_opt_policy(U, hit);
    std::vector<double> reach;
    optimal_reach(hit.data(), reach);

    // --- exact pmfs --------------------------------------------------------
    std::vector<double> D(kRoundScoreMax + 1), De(kRoundScoreMax + 1), Dl(kRoundScoreMax + 1);
    const double pfD  = pmf_forced(0, 0, hit.data(), D.data());
    const double pfDe = pmf_forced(0, 3, hit.data(), De.data());
    const double pfDl = pmf_flip3_at_stop(hit.data(), Dl.data());
    const auto sD = pmf_stats(D.data(), pfD), sDe = pmf_stats(De.data(), pfDe), sDl = pmf_stats(Dl.data(), pfDl);

    double tot = 0; for (double p : D) tot += p;
    check(close(tot, 1.0, 1e-9), "baseline round pmf sums to 1");
    check(close(sD.mean, 18.5652176, 1e-4), "baseline mean == 18.5652 (matches Ch.4)");

    // Freeze never reduces below the stay value: EV(S) >= sum(S) everywhere.
    bool freeze_nonneg = true;
    for (int S = 0; S < N; ++S) if (U[S] + 1e-9 < (double)g_sum[S]) freeze_nonneg = false;
    check(freeze_nonneg, "Freeze damage EV(S)-sum(S) >= 0 for every hand");

    // Flip Three is never a gift; ~neutral when shallow, an attack mid-deep.
    std::vector<double> init(N, 0.0), out(kRoundScoreMax + 1);
    double dMean[7] = {0};
    for (int k = 0; k <= 6; ++k) {
        std::fill(init.begin(), init.end(), 0.0);
        double w = 0; for (int S = 0; S < N; ++S) if (g_pc[S] == k) { init[S] = reach[S]; w += reach[S]; }
        if (w <= 0) continue;
        double pf = pmf_from_dist(init.data(), 0, hit.data(), out.data());
        const double base = pmf_stats(out.data(), pf).mean;
        pf = pmf_from_dist(init.data(), 3, hit.data(), out.data());
        dMean[k] = pmf_stats(out.data(), pf).mean - base;
    }
    check(dMean[2] < 0 && dMean[3] < 0 && dMean[4] < 0, "Flip Three is an attack (dMean<0) when target is mid-deep");
    check(dMean[0] > -0.5, "Flip Three is ~neutral on a shallow hand (k=0)");
    check(std::fabs(dMean[6]) < std::fabs(dMean[4]),
          "Flip Three weakens near Flip 7 (forced draws complete the bonus)");

    // self-Flip3 ~ neutral (dominated by not using it); opp-Flip3@stop is a real attack.
    check(sDe.mean < sD.mean && sDe.mean > sD.mean - 0.5, "self Flip-Three is ~neutral (slightly negative)");
    check(sDl.mean < sD.mean - 8.0, "opp Flip-Three@stop slashes the target's mean");
    check(sDl.p_bust > sD.p_bust + 0.3, "opp Flip-Three@stop sharply raises the target's bust rate");

    // --- independent MC must reproduce the exact pmfs ----------------------
    const uint64_t n = 8'000'000;
    const auto m0 = mc_numbers_forced(0, hit.data(), n, 0xA1ULL);
    const auto m1 = mc_numbers_forced(1, hit.data(), n, 0xA2ULL);
    const auto m2 = mc_numbers_forced(2, hit.data(), n, 0xA3ULL);
    check(close(m0.mean, sD.mean, 0.03), "MC matches exact baseline mean");
    check(close(m1.mean, sDe.mean, 0.03), "MC matches exact self-Flip3 mean");
    check(close(m2.mean, sDl.mean, 0.03), "MC matches exact opp-attack mean");
    check(close(m2.p_bust, sDl.p_bust, 0.01), "MC matches exact opp-attack bust rate");

    // --- win-prob targeting DP (none / self / opp), with its own MC --------
    constexpr int kT = 200;
    auto support = [](const std::vector<double>& P) {
        std::vector<int> s; for (int i = 0; i < (int)P.size(); ++i) if (P[i] > 0) s.push_back(i); return s;
    };
    const auto supD = support(D), supDe = support(De), supDl = support(Dl);
    std::vector<double>  W((size_t)kT * kT, 0.0);
    std::vector<uint8_t> pol((size_t)kT * kT, 0);
    auto optval = [&](int a, int b, const std::vector<double>& pa, const std::vector<int>& sa,
                      const std::vector<double>& po, const std::vector<int>& so) {
        double num = 0.0; const double self = pa[0] * po[0];
        for (int x : sa) { const double pax = pa[x]; const int A = a + x;
            for (int y : so) { if (x == 0 && y == 0) continue; const double p = pax * po[y]; const int B = b + y;
                if (A >= kT || B >= kT) num += p * (A > B ? 1.0 : (A == B ? 0.5 : 0.0));
                else                    num += p * W[(size_t)A * kT + B]; } }
        return num / (1.0 - self);
    };
    for (int sumab = 2 * (kT - 1); sumab >= 0; --sumab) {
        const int alo = std::max(0, sumab - (kT - 1)), ahi = std::min(kT - 1, sumab);
        for (int a = alo; a <= ahi; ++a) { const int b = sumab - a;
            const double vn = optval(a, b, D, supD, D, supD);
            const double vs = optval(a, b, De, supDe, D, supD);
            const double vo = optval(a, b, D, supD, Dl, supDl);
            double best = vn; uint8_t bp = 0;
            if (vs > best) { best = vs; bp = 1; }
            if (vo > best) { best = vo; bp = 2; }
            W[(size_t)a * kT + b] = best; pol[(size_t)a * kT + b] = bp;
        }
    }
    check(W[0] > 0.5, "a free Flip Three each round is worth > 0.5 at the start");
    check(pol[(size_t)50 * kT + 150] == 2, "when behind, aim the Flip Three at the opponent");

    printf("%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
