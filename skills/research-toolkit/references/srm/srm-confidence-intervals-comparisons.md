---
name: srm-confidence-intervals-comparisons
description: Confidence intervals for comparisons between two groups - paired (before/after, matched) mean differences, two independent means with Welch vs pooled limits, odds ratios and differences of proportions from 2x2 counts (Woolf log-OR; Wald vs Wilson vs Agresti-Coull), plus reading error-bar charts - pure Python (numpy/scipy/statsmodels). Use when asked for a 95% CI for a paired mean difference, the difference of two group means, an odds ratio from four cell counts, whether to report Welch or pooled limits, whether overlapping error bars mean no difference, or whether the comparison interval is statistically valid.
---

# Confidence Intervals - Comparisons (Python)

## Purpose
CIs where the parameter is a comparison: paired mean difference `mu_d`, difference of two independent
means `mu_A - mu_B`, odds ratio, difference of proportions. Same skeleton as the single-sample sibling -
`estimate +/- multiplier x s.e.(estimate)`, multiplier 2 approximate, exact `t`/`z` for software (Dunn p358).

## Inputs
- Paired: one difference per unit of analysis, direction fixed up front (`saving = before - after`), or summary `d_bar, s_d, n`.
- Two means: per-group `mean, s.d., n`, or a software-supplied `s.e.(diff)` - it cannot be rebuilt from two s.d.s alone (Dunn p355, p359).
- 2x2 counts `a, b, c, d`: group A event/non-event then group B event/non-event; group A is the OR numerator.
- ALPHA = 0.05, APPROX_MULT = 2. Reorder inputs so Row1/Col1 is the numerator instead of mentally negating printed limits (Dunn p373-375).

## Algorithms (all pure numpy/scipy)
### Paired mean difference (Dunn Ch.23, p340-347)
```python
import math
from scipy import stats
before = (12.1, 11.0, 14.1, 13.8, 15.5, 12.2, 12.8, 9.9, 10.8, 12.7)  # Dunn Table 23.1 p341
after  = (12.0, 10.6, 13.4, 11.2, 15.3, 13.6, 12.6, 8.8, 9.6, 12.4)
d = [b - a for b, a in zip(before, after)]            # direction: saving = before - after
n = len(d); dbar = sum(d) / n
s_d = math.sqrt(sum((x - dbar) ** 2 for x in d) / (n - 1))    # s.d. OF the differences
se = s_d / math.sqrt(n)
exact = (dbar - stats.t.ppf(0.975, n - 1) * se, dbar + stats.t.ppf(0.975, n - 1) * se)
ci = stats.ttest_1samp(d, 0.0).confidence_interval(0.95)      # scipy cross-check, tol 1e-9
assert max(abs(float(ci.low) - exact[0]), abs(float(ci.high) - exact[1])) < 1e-9
print(f"RESULT d_bar={dbar:.4f} s_d={s_d:.6f} se={se:.6f} ci_approx=[{dbar - 2 * se:.4f},{dbar + 2 * se:.4f}] ci_exact_t=[{exact[0]:.4f},{exact[1]:.4f}]")
```
Approximate limits -0.10 to 1.18 MWh = the book's (Dunn p344); exact `t` -0.19 to 1.27 = jamovi/SPSS
(Dunn p346-347). `s_d` is the s.d. of the differences, not `s_1 - s_2`; n = 10 < 25 conditions validity on near-normal differences.
### Two independent means: Welch vs pooled (Dunn p355-367)
```python
import math
import numpy as np
from scipy import stats
from statsmodels.stats.weightstats import CompareMeans
def two_means_ci(mA, sA, nA, mB, sB, nB, equal_var=False, alpha=0.05):
    vA, vB = sA ** 2 / nA, sB ** 2 / nB
    if equal_var:                            # pooled/Student: needs equal variances
        se = math.sqrt(((nA - 1) * sA ** 2 + (nB - 1) * sB ** 2) / (nA + nB - 2) * (1 / nA + 1 / nB))
        df = nA + nB - 2
    else:                                    # Welch: default, fewer assumptions (Dunn p360)
        se = math.sqrt(vA + vB)
        df = (vA + vB) ** 2 / (vA ** 2 / (nA - 1) + vB ** 2 / (nB - 1))
    diff = mA - mB
    me = stats.t.ppf(1 - alpha / 2, df) * se
    return diff, se, df, (diff - me, diff + me)
cw = two_means_ci(585.19, 89.65, 32, 533.59, 65.36, 32); cp = two_means_ci(585.19, 89.65, 32, 533.59, 65.36, 32, equal_var=True)  # Dunn Table 24.3 p356
print(f"RESULT diff={cw[0]:.5f} se_welch={cw[1]:.4f} df={cw[2]:.2f} ci_welch=[{cw[3][0]:.2f},{cw[3][1]:.2f}] ci_pooled=[{cp[3][0]:.2f},{cp[3][1]:.2f}]")
fw = two_means_ci(30.7, 2.75, 10, 16.2, 4.44, 5); fp = two_means_ci(30.7, 2.75, 10, 16.2, 4.44, 5, equal_var=True)  # fall-recovery, Dunn p366
print(f"RESULT faceplant_welch=[{fw[3][0]:.2f},{fw[3][1]:.2f}] faceplant_pooled=[{fp[3][0]:.2f},{fp[3][1]:.2f}]")
rng = np.random.default_rng(42)                               # raw-data cross-check, tol 1e-9
a, b = rng.normal(585.19, 89.65, 32), rng.normal(533.59, 65.36, 32)
mine = two_means_ci(a.mean(), a.std(ddof=1), 32, b.mean(), b.std(ddof=1), 32)[3]
sm = CompareMeans.from_data(a, b).tconfint_diff(alpha=0.05, usevar="unequal")
sc = stats.ttest_ind(a, b, equal_var=False).confidence_interval(0.95)
assert max(abs(mine[0] - float(sm[0])), abs(mine[1] - float(sm[1])), abs(mine[0] - float(sc.low)), abs(mine[1] - float(sc.high))) < 1e-9
print("RESULT welch_crosscheck=statsmodels_and_scipy_ok")
```
Ship Welch by default - jamovi ("Welch's") and SPSS ("Equal variances not assumed") print it and it
makes fewer assumptions (Dunn p360); Welch 12.32-90.88 rounds to the book's 12.3 to 90.9. Pooled is
admissible only with defensible equal variances: at equal n the two nearly coincide, but with n = 10 vs 5
and s.d. 2.75 vs 4.44 pooled shrinks to 10.52-18.48 while the book reports Welch 9.10 to 19.90 (Dunn p366-367).
### Odds ratio from 2x2 counts, log/Woolf interval (Dunn p372-378)
```python
import math
from scipy import stats
from statsmodels.stats.contingency_tables import Table2x2
def or_ci(a, b, c, d, alpha=0.05):
    if min(a, b, c, d) == 0:                    # zero cell: add 0.5 to every cell and say so
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    z = stats.norm.ppf(1 - alpha / 2)
    or_hat = a * d / (b * c)
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)   # s.e. of ln(OR), not of the OR (Dunn p374)
    return or_hat, se, math.exp(math.log(or_hat) - z * se), math.exp(math.log(or_hat) + z * se)
or_hat, se, lo, hi = or_ci(52, 105, 2, 24)      # students; Dunn Table 25.1 p372
tab = Table2x2([[52, 105], [2, 24]])
sm_lo, sm_hi = tab.oddsratio_confint(alpha=0.05, method="normal")
assert max(abs(or_hat - tab.oddsratio), abs(lo - float(sm_lo)), abs(hi - float(sm_hi))) < 1e-9
exp_ct = [r * c / 183 for r in (157, 26) for c in (54, 129)]
assert min(exp_ct) >= 5                          # EXPECTED counts >= 5, not observed (Dunn p377)
print(f"RESULT OR={or_hat:.4f} se_lnOR={se:.6f} ci95=[{lo:.3f},{hi:.3f}] statsmodels_agrees=True expected_counts={[round(v, 2) for v in exp_ct]}")
```
Prints OR = 5.943, CI 1.35 to 26.1 - the book's software interval (Dunn p375); wide because one
*observed* cell is 2, yet valid since all expected counts clear 5 (Dunn p377-378). Build the interval
on the log scale - the OR itself is not normal, so never `+/- 2 s.e.` on the OR scale (Dunn p374,
p376). Report "times the odds", never "times as likely" (Dunn p372).
### Group percentages and difference of proportions: Wald vs Wilson vs Agresti-Coull (Dunn p373, p377)
```python
import math
from scipy import stats
from statsmodels.stats.proportion import proportion_confint, confint_proportions_2indep
z = stats.norm.ppf(0.975)
def wilson(x, n):
    p, den = x / n, 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z / den * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ctr - half, ctr + half
for x, n in ((52, 54), (105, 129), (2, 26)):    # group percentages beside the OR (Dunn p373)
    p = x / n
    wlo, whi = p - z * math.sqrt(p * (1 - p) / n), p + z * math.sqrt(p * (1 - p) / n)  # Wald
    wi = proportion_confint(x, n, 0.05, "wilson")
    ac = proportion_confint(x, n, 0.05, "agresti_coull")
    assert max(abs(wilson(x, n)[0] - float(wi[0])), abs(wilson(x, n)[1] - float(wi[1]))) < 1e-9
    warn = " WALD_OUT_OF_[0,1]_REPORT_WILSON" if wlo < 0 or whi > 1 else ""
    print(f"RESULT x={x} n={n} wald=[{wlo:.4f},{whi:.4f}] wilson=[{wi[0]:.4f},{wi[1]:.4f}] ac=[{ac[0]:.4f},{ac[1]:.4f}]{warn}")
for m in ("wald", "agresti-caffo", "newcomb"):  # p(with parents) - p(without), meals off-campus
    lo, hi = confint_proportions_2indep(52, 54, 105, 129, method=m)
    print(f"RESULT diff_prop_{m}=[{lo:.4f},{hi:.4f}]")
```
Wald falls below 0 at 2/26 - the small-cell failure that makes Wilson/Agresti-Coull the reportable
percentages next to an OR (Dunn p373, p377); avoid `method="score"`, whose inversion can fail on tiny cells.
### Reading error-bar charts: overlap does not mean "no difference" (Dunn p361-364)
```python
from scipy import stats
bars = {}                                            # Dunn Table 24.3 p356 (ms; s.e. per group)
for name, (m, se, n) in {"phone": (585.19, 15.847, 32), "control": (533.59, 11.554, 32)}.items():
    bars[name] = stats.t.interval(0.95, n - 1, loc=m, scale=se)
dc = (51.59375 - 2 * 19.61213, 51.59375 + 2 * 19.61213)   # difference CI, book s.e. (Dunn p358)
print(f"RESULT bar_phone=[{bars['phone'][0]:.2f},{bars['phone'][1]:.2f}] bar_control=[{bars['control'][0]:.2f},{bars['control'][1]:.2f}]")
print(f"RESULT bars_overlap={bars['phone'][0] < bars['control'][1]} ci_difference=[{dc[0]:.2f},{dc[1]:.2f}] excludes_zero={dc[0] > 0}")
```
The bars overlap while the difference CI (12.37 to 90.82, Dunn p358) excludes 0: overlap is no proof
of no difference, non-overlap is not a test, and bars show each mean's sampling variation, not the
difference - use a boxplot for individual observations and medians (Dunn p361-364).
## Output contract
One `RESULT name=value` line per quantity: `estimate` (`d_bar` | `mean_A - mean_B` | `OR` |
`p_A - p_B`), `se`, `df`, `ci_lo`/`ci_hi`, `method` (`approx|t|welch|pooled|woolf|wald|wilson|
agresti_coull|agresti_caffo|newcomb`), `statistically_valid`, `direction` (what a positive value
favours), `expected_counts` for a 2x2. Every closed form is asserted against scipy/statsmodels at tol 1e-9.

## Validity conditions
- Paired: n >= 25 differences, or a near-normal population of differences when n < 25 (Dunn p347).
- Two means: n >= 25 in BOTH groups, or both populations near-normal; otherwise state the caveat (Dunn p361, p367).
- Odds ratio: all four expected counts (row total x column total / N) >= 5 - expected, not observed (Dunn p377-378).
- Pairing is a design feature decided before seeing the numbers; the paired procedure on unpaired data (or vice versa) is the fatal error (Dunn p340-341).
- A CI containing 0 (or 1 for an OR) is inconclusive-with-direction, not "no effect" (Dunn p348, p379).

## Reporting conventions
One sentence carries estimate, interval, method, level, direction and validity: "an approximate 95%
CI for the population mean energy saving is from -0.10 to 1.18 MWh" (Dunn p344); say which group is
higher (Dunn p359) and name the OR's group order in parentheses (Dunn p375). Allowed: "we are 95%
confident the interval straddles the population value"; forbidden: that the fixed parameter has a 95%
probability of lying inside this interval (Dunn p383). Round only the reported answer (Dunn p344).

## Out of scope
Single-population proportion/mean CIs and sample-size planning (single-sample sibling skill),
hypothesis tests, p-values, chi-square, slope CIs, correlation. An interval is the dual of a test.

## Conventions
- ALPHA = 0.05; multiplier 2 for the approximate interval, exact t/z to reproduce software - say which you used (Dunn p344, p358).
- statsmodels is the cross-check oracle: `CompareMeans.tconfint_diff(usevar="unequal")` (keyword is `usevar`), `Table2x2.oddsratio_confint(method="normal")`, `proportion_confint`, `confint_proportions_2indep`. Fisher's exact conditional OR interval is not shipped; Woolf is the OR interval here.
- Raw demo data only via `numpy.random.default_rng(42)`; never invent a value - compute it and cite the source page.
- Source: Dunn, *Scientific Research and Methodology* (Springer, 2024) Ch.23-25, pp.339-386.
