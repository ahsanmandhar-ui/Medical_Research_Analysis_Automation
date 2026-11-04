---
name: srm-confidence-intervals-single
description: Confidence interval for ONE population proportion p or ONE population mean mu from a single sample, pure Python (math/scipy/statsmodels). Computes standard error, multiplier-of-2 approximate margin, exact Wald/Wilson/Agresti-Coull/Clopper-Pearson or Student-t interval, the statistical-validity check (both category counts > 5; n >= 25) and the sample size needed for a target margin. Use when asked for a 90/95/99% CI for a proportion or mean, "margin of error", standard error of p-hat or x-bar, or "how many respondents do I need".
---

# Confidence Intervals - Single Sample (Python)

## Purpose
Form and validate a confidence interval for one population parameter: a proportion `p` or a mean `mu`.
Every interval is `statistic ± (multiplier × standard error)`; `multiplier × s.e.` is the margin of
error, and the multiplier is ≈ 2 for an approximate 95% CI (Dunn p321-322).

## Inputs
- Proportion: `x` (count in the category) and `n`, or a 0/1 column. Use proportions (0.24), never
  percentages (24%), in every formula (Dunn p316).
- Mean: `xbar`, sample sd `s`, `n`, or one numeric column.
- `ALPHA = 0.05` (level 0.95); `MULT = 2` for the approximate interval; `CONF = 1 - ALPHA`.

## Algorithms (all pure numpy/scipy)

### One proportion (Dunn p308, p311, p314)
```python
import math
x, n, ALPHA = 365, 1516, 0.05          # energy-drink example, Dunn p311
p = x / n
se = math.sqrt(p * (1 - p) / n)        # s.e.(p_hat)
lo, hi = p - 2 * se, p + 2 * se        # approximate 95% CI
valid = x > 5 and (n - x) > 5          # statistical validity: BOTH counts exceed 5
print(f"RESULT p={p:.4f} se={se:.5f} approx95=[{lo:.4f},{hi:.4f}] valid={valid}")
```
Prints `p=0.2408 se=0.01098 approx95=[0.2188,0.2627] valid=True` (Dunn p311).

### Software intervals for the same count
```python
from statsmodels.stats.proportion import proportion_confint
x, n = 365, 1516
for m in ("normal", "wilson", "agresti_coull", "beta"):
    lo, hi = proportion_confint(x, n, alpha=0.05, method=m)
    print(f"RESULT {m}=[{lo:.4f},{hi:.4f}]")
```
`beta` is Clopper-Pearson exact - use it when an event count is 0 or tiny and the Wald interval collapses.

### One mean, sigma unknown -> Student-t, not z (Dunn p329-330, p332)
```python
import math
from scipy import stats
xbar, s, n = 2.8, 0.94, 586            # school-bag weights, Dunn p330
se = s / math.sqrt(n)                  # s.e.(x_bar)
approx = (xbar - 2 * se, xbar + 2 * se)
t_lo, t_hi = stats.t.interval(0.95, n - 1, loc=xbar, scale=se)
valid = n >= 25                        # or n < 25 with near-normal data
print(f"RESULT se={se:.5f} approx95=[{approx[0]:.4f},{approx[1]:.4f}] t95=[{t_lo:.4f},{t_hi:.4f}] valid={valid}")
```
Prints `approx95=[2.7223,2.8777]` - the interval is for the population mean, not for individuals or
the sample proportion (Dunn p325-326, p337).

### Sample size for a target margin of error (Dunn p317, p336)
```python
import math
margin_p, margin_m, sd = 0.07, 0.02, 0.39926
n_prop = math.ceil(1 / margin_p ** 2)          # 205
n_mean = math.ceil((2 * sd / margin_m) ** 2)   # 1595 (HDL example, Dunn p336)
print(f"RESULT n_for_proportion={n_prop} n_for_mean={n_mean}")
```
Always round **up** - a rounded-down n misses the target margin.

## Output contract
Return/report: `statistic`, `se`, `margin_of_error`, `ci_lo`, `ci_hi`, `confidence_level`,
`method` (`approx` | `wald` | `wilson` | `agresti_coull` | `beta` | `t`), `statistically_valid` (bool),
and, for planning, `n_required`. Print `RESULT name=value` per quantity.

## Validity conditions
- Proportion: statistical validity is about **both category counts > 5**, not total n (Dunn p314).
- Mean: the *sampling distribution of the mean* must be near-normal; `n >= 25` makes skewed data fine
  (right-skewed NHANES HDL is still valid, Dunn p334); with `n < 25` inspect a histogram - a clearly
  non-normal small sample (CT doses, n=17) makes the CI suspect (Dunn p333).
- Internal validity: the study must be well designed. External validity: a simple random sample, and
  `n < 10%` of the population, which the CI maths assumes (Dunn p323-324).
- Higher confidence means a wider interval; the parameter is fixed, the interval is the random quantity
  (Dunn p311, p325).

## Reporting conventions
State statistic, interval, method, level and validity in one sentence, using straddle/plausible-values
wording (Dunn p322-323): "Based on a simple random sample of n = 1516, the sample proportion was 0.241;
an approximate 95% CI for the population proportion is 0.219 to 0.263, which we are 95% confident
straddles the population proportion; the CI is statistically valid (365 and 1151 both exceed 5)."
Round only the reported answer, keep full precision in the working (Dunn p312-313).

## Out of scope
Differences between two proportions/means, paired data and odds ratios belong to the comparison-CI
skill (Dunn Ch.24-25); hypothesis tests are the dual of an interval, not the same tool.

## Conventions
- ALPHA = 0.05, MULT = 2 for the approximate interval, CONF = 0.95 unless stated otherwise.
- `math`/`scipy.stats` for the numerics; `statsmodels.proportion_confint` for exact methods; headless
  `matplotlib` (`Agg`) if a plot is asked for.
- Seeded demo data only via `numpy.random.default_rng(42)`; never hardcode a result - always compute it
  and print the `RESULT` lines above.
- Source: Dunn, *Scientific Research and Methodology* (Springer, 2024) Ch.19-22, pp.301-338.
