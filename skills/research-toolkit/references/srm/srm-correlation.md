---
name: srm-correlation
description: "Scatterplot reading (form, direction, variation, outliers, influential points), Pearson correlation r, R-squared as explained variation and the rho=0 significance test in pure Python (numpy.corrcoef, scipy.stats.pearsonr, scipy.stats.t), from Dunn, Scientific Research and Methodology (Springer, 2024) Ch.33-34, pp.481-502. Use when asked about the association between two QUANTITATIVE variables (\"is X correlated with Y\"), how strong a correlation is, an R-squared value, a p-value for H0: rho = 0 (two- or one-tailed), whether a correlation is statistically valid, or how to recover r from a reported R-squared. Not for fitting a prediction line, slope or residuals (use srm-linear-regression); not for comparing categories (use srm-hypothesis-tests-categorical)."
---

# Correlation: Scatterplot, Pearson r, R-squared (Python)

## Purpose
Describe and test the *linear* association between two quantitative variables: read the scatterplot,
compute Pearson `r` and R-squared, test `H0: rho = 0`, check validity. Correlation measures strength
only and never establishes cause (Dunn p489-490, p492).

## Inputs
- `x`, `y`: paired numeric vectors of equal length, one row per unit of analysis; both must be
  quantitative, else a scatterplot and `r` are the wrong device (Dunn p482).
- Reported-statistic mode: `r` and `n`, or `R2` alone (then `|r| = sqrt(R2)`, sign from the plot).
- `ALPHA = 0.05`; `alt` in `two` | `greater` | `less`; `MIN_N = 25`; `SEED = 42`.

## Algorithms (all pure numpy/scipy)

### Scatterplot reading: form, direction, variation, outliers, influential point
```python
import numpy as np
rng = np.random.default_rng(42)
x = rng.uniform(5, 25, 60); y = 2 + 0.9 * x + rng.normal(0, 3.5, 60)
b1, b2 = np.polyfit(x, y, 1), np.polyfit(x, y, 2); ss = float(np.sum((y - y.mean()) ** 2))
gain = (np.sum((y - np.polyval(b1, x)) ** 2) - np.sum((y - np.polyval(b2, x)) ** 2)) / ss
r = float(np.corrcoef(x, y)[0, 1]); vr = float((y - np.polyval(b1, x)).std() / y.std())
form = "curved" if gain > 0.05 else ("linear" if abs(r) > 0.2 else "none")
variation = "small" if vr < 0.35 else ("moderate" if vr < 0.60 else "large")
print(f"RESULT r={r:.4f} form={form} direction={'positive' if r > 0 else 'negative'} variation={variation} quad_r2_gain={gain:.4f} resid_sd_over_y_sd={vr:.3f}")
print(f"RESULT r_with_one_added_influential_point={float(np.corrcoef(np.append(x, 30.0), np.append(y, 2.0))[0, 1]):.4f}")
```
Describe form (linear/curved/none), direction (positive/negative; not relevant when curved) and variation
(little scatter = strong, much = weak), then anything unusual (Dunn p482-483). An extreme outlier sits
beyond 3 x IQR from Q1/Q3 (Dunn p227) and one point far along `x` can swing `r`, so report `r` with and
without it; `r` is allowed only once the form reads as linear (Dunn p490).

### Pearson r, the estimate of population rho
```python
import numpy as np
from scipy import stats
rng = np.random.default_rng(42)
x, y = rng.multivariate_normal([0.0, 0.0], [[1.0, -0.584], [-0.584, 1.0]], size=78).T
r_cc = float(np.corrcoef(x, y)[0, 1]); r_sp, p_two = stats.pearsonr(x, y)
r_def = float(np.mean((x - x.mean()) / x.std(ddof=0) * (y - y.mean()) / y.std(ddof=0)))
assert abs(r_cc - r_sp) < 1e-9 and abs(r_cc - r_def) < 1e-9
print(f"RESULT r_corrcoef={r_cc:.4f} r_pearsonr={r_sp:.4f} r_zscore_definition={r_def:.4f} p_two_sided={p_two:.3g}")
```
`r` is unitless, lies in [-1, +1], and equals the mean product of standardised values: sign gives
direction, magnitude gives strength; `r = 0` means no *linear* relationship, not none (Dunn p489-490).

### R-squared, the calculator trap, and recovering r from a reported R-squared
```python
import math
r = -0.584                                       # red deer molar weight vs age (Dunn p490)
R2 = r * r                                       # square the WHOLE signed value
print(f"RESULT r={r:+.3f} r_squared={R2:.4f} explained_pct={R2 * 100:.1f} extraneous_pct={100 - R2 * 100:.1f}")
print(f"RESULT calculator_trap_correct={(-0.5) ** 2:.2f} wrong={-0.5 ** 2:.2f}")
for R2_rep in (0.9929, 0.6707): print(f"RESULT reported_r_squared={R2_rep:.4f} magnitude_of_r={math.sqrt(R2_rep):.3f} sign=read_off_scatterplot")
```
R-squared is never negative and is quoted as a percentage: the variation in `y` explained by the linear
relationship with `x` rather than by the mean of `y`; the rest is attributed to extraneous variables
(weight, diet, exercise, genetics) (Dunn p493-494). Squaring discards the sign, so a published
R-squared yields `|r|` only - the "be careful" hint in Exercises 34.4-34.5 (Dunn p501-502).

### Inference: t-test on r, reproducing the Ch.34 worked numbers
```python
import math
from scipy import stats
def corr_test(r, n, alt="two"):
    df = n - 2; t = r * math.sqrt(df / (1.0 - r * r))
    p = 2 * stats.t.sf(abs(t), df) if alt == "two" else (stats.t.sf(t, df) if alt == "greater" else stats.t.cdf(t, df))
    return t, df, p
for label, r, n, alt, cite in [("red deer molar weight vs age", -0.584, 78, "two", "Dunn p490,p494-495"),
        ("NHANES systolic BP vs age", 0.532, 10000, "two", "Dunn p491,p494"),
        ("removal efficiency vs inlet temp", 0.891, 32, "two", "Dunn p498-499"),
        ("Phu Quoc dogs length vs height", 0.837, 30, "greater", "Dunn p496-497"),
        ("6MWD vs VO2max (paediatric)", 0.457, 29, "two", "Dunn p500"),
        ("work experience vs STAI stress", 0.346, 30, "two", "Dunn p497-498")]:
    t, df, p = corr_test(r, n, alt)
    pl = "P<0.001" if p < 0.001 else f"P={p:.3f}"
    print(f"{label}: r={r:+.3f} R2={r * r * 100:.1f}% n={n} t={t:.2f} df={df} {pl} ({alt}) ({cite})")
print(f"RESULT deer_explained_pct={(-0.584) ** 2 * 100:.1f} p_two_6mwd={corr_test(0.457, 29)[2]:.4f}")
```
`H0: rho = 0`; `H1: rho != 0` (two-tailed) or `> 0`/`< 0` (one-tailed, half the two-tailed P when the
sign agrees); `t = r*sqrt((n-2)/(1-r*r))` on `n-2` df (Dunn p495, p497). Deer print 34.1% explained and
P < 0.001, systolic BP 28.3%, removal efficiency 79.4%, 6MWD P = 0.013 - all as published.

### Correlation versus the regression slope
```python
import numpy as np
from scipy import stats
rng = np.random.default_rng(7)
x = rng.normal(50, 10, 200); y = 6.0 + 0.9 * (x - 50) + rng.normal(0, 3.0, 200)
r, _ = stats.pearsonr(x, y); r_mm, _ = stats.pearsonr(x, y * 1000)
slope = r * y.std(ddof=1) / x.std(ddof=1); slope_mm = r_mm * (y * 1000).std(ddof=1) / x.std(ddof=1)
print(f"RESULT r={r:.4f} r_after_unit_change={r_mm:.4f} slope={slope:.4f} slope_after_unit_change={slope_mm:.2f}")
```
`r` is bounded and scale-free - rescaling the response leaves `r` identical but multiplies the slope by
1000. The slope answers "how much y per unit x" and carries units; `r` answers "how tightly do they
co-vary"; only their signs always agree (Dunn p507).

### The three statistical-validity conditions
```python
import numpy as np
from scipy import stats
rng = np.random.default_rng(42)
def screens(x, y):
    n = len(x); r = float(np.corrcoef(x, y)[0, 1]); ss = float(np.sum((y - y.mean()) ** 2))
    b1, b2 = np.polyfit(x, y, 1), np.polyfit(x, y, 2)
    gain = (np.sum((y - np.polyval(b1, x)) ** 2) - np.sum((y - np.polyval(b2, x)) ** 2)) / ss
    linear = bool(gain < 0.05 and abs(r - float(stats.spearmanr(x, y).statistic)) < 0.15)
    res = (y - np.polyval(b1, x))[np.argsort(x)]; k = max(1, n // 3); sds = [s for s in (res[:k].std(), res[k:2 * k].std(), res[2 * k:].std()) if s > 0]
    ratio = float(max(sds) / min(sds)); ext = sum(int(np.sum((v < q1 - 3 * (q3 - q1)) | (v > q3 + 3 * (q3 - q1)))) for v in (x, y) for q1, q3 in [np.percentile(v, [25, 75])])
    print(f"RESULT n={n} r={r:.4f} linear={linear} sd_ratio={ratio:.2f} extreme_outliers={ext} statistically_valid={bool(linear and ratio < 2.0 and n >= 25 and ext == 0)}")
x_lin, y_lin = rng.multivariate_normal([0, 0], [[1, -0.584], [-0.584, 1]], size=78).T; d = rng.uniform(5, 35, 40)
screens(x_lin, y_lin)                                             # deer-like: all three conditions hold
screens(d, 0.15 * (d - 20) ** 2 + rng.normal(0, 1, 40) * d / 5)   # lime-like: curve plus fanning spread
s24 = rng.uniform(5, 25, 24)
screens(s24, 6 + 0.9 * s24 + rng.normal(0, 3.0, 24))               # punting/CBR-like: n below the 25 rule
```

## Output contract
Report `n`, `r`, `r_squared` (with `explained_pct` / `extraneous_pct`), `t_statistic`, `df`, `p_value`,
`alt`, `form`, `direction`, `variation`, `extreme_outliers`, `statistically_valid` (naming the failing
condition). Each block prints `RESULT name=value` lines to stdout and writes no files.

## Validity conditions
- Both variables quantitative (Dunn p482); approximately linear form before quoting `r`, since a curved cloud can sit near `r = 0` (Dunn p490-491).
- Variation in `y` roughly constant across `x`, and no extreme outlier driving the result (Dunn p496, p227).
- `n >= 25`, a rough figure: punting (n = 13) and CBR (n = 16) fail it despite clear trends (Dunn p500-502).
- External validity needs a simple random sample, internal validity a well-designed study (Dunn p496).
- Small P means `rho != 0`, never "strong": `r = 0.457` returned P = 0.013 (Dunn p495, p500).
- Observational data invite confounding - also check `x` and `y` against extraneous variables (Dunn p492).

## Reporting conventions
Give the answer to the RQ, the evidence (P with its tail; no test statistic), the sample summary (`r`;
`n`) and the validity verdict (Dunn p495): "very strong evidence (two-tailed P < 0.001) of a correlation
between molar weight and age in male red deer (r = -0.584; n = 78); R-squared = 34.1%, so about 34.1% of
the variation is explained by age and the rest by extraneous variables; the test is statistically
valid". For `r = 0.346; n = 30` the paper prints P = 0.18 and the plain t-test 0.061 - the "no evidence"
call is unchanged (Dunn p497-498).

## Out of scope
Fitting the line of best fit, slope `b1` and its CI, predicted values, residual diagnostics and
extrapolation belong to the sibling skill `srm-linear-regression`; category comparisons belong to
`srm-hypothesis-tests-categorical`. No CI is built for `r`: it is bounded, so its sampling distribution
is not normal and Dunn omits the CI deliberately (Dunn p494-495).

## Conventions
- `ALPHA = 0.05`; P < 0.001 is worded "very strong evidence", matching software printing 0.000 (Dunn p495).
- Seeded synthetic data via `numpy.random.default_rng(42)`; never invent a value - compute it and cite the source page.
- Cites use the book's PDF pagination: Ch.33 pp.481-488, Ch.34 pp.489-502; reports round only the final answer.
- `numpy.corrcoef` / `scipy.stats.pearsonr` for `r`, `scipy.stats.t` for the P; Spearman is a linearity
  cross-check only, never a reported coefficient (Dunn p490).
- `pingouin` is not installed and is never imported; its one-table output (r, t, df, one- and two-tailed P)
  and the book's jamovi/SPSS figures are reproduced by the blocks above.
- Source: Dunn, *Scientific Research and Methodology* (Springer, 2024) Ch.33-34, pp.481-502.
