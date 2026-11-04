---
name: srm-linear-regression
description: Fit, interpret and test a SIMPLE linear regression y_hat = b0 + b1*x (one explanatory x, one response y) in pure Python - least-squares OLS via statsmodels and sklearn, intercept/slope meaning and units, predicted values with an extrapolation guard, the slope hypothesis test t=(b1-beta1)/s.e.(b1) with two- or one-tailed P, approximate (multiplier 2) and exact (t, df=n-2) CI for the population slope, and the three statistical-validity checks (linearity, constant spread, n>=25) plus leverage and Cook's-distance diagnostics. Use when the user wants a regression equation, asks how to interpret the slope/intercept, wants to "predict y for a given x", asks "is the slope significantly different from zero", wants a "confidence interval for the slope", asks to "check the residuals / assumptions / influential points", asks "is this extrapolation", or supplies paired (x, y) numeric data. Based on Dunn, Scientific Research and Methodology (Springer 2024) ch.35.
---

# Simple Linear Regression (Python)

## Purpose
Describe *what* the straight-line relationship between one explanatory `x` and one response `y` is, via
`y_hat = b0 + b1*x`, for prediction or explanation, and make inference on the population slope `beta1`
(Dunn p503-504). Correlation measures the *strength* of the line; regression states its *equation*.

## Inputs
- `x`, `y`: paired numeric columns/arrays, **or** summary stats `b1`, `s.e.(b1)`, `n` (Dunn p511-512).
- `ALPHA = 0.05`, `CONF = 0.95`, `MULT = 2` for the approximate CI, `MIN_N = 25` (Dunn p515).
- `y_hat` ("why-hat") is the *predicted* response, distinct from the observed `y` (Dunn p504).

## Algorithms (all pure numpy/scipy)

### Fit the least-squares line (Dunn p506-507)
```python
import numpy as np, statsmodels.api as sm
from sklearn.linear_model import LinearRegression
rng = np.random.default_rng(42)
n = 78
x = rng.uniform(4.4, 14.4, n)                          # red-deer age range, Dunn p507
y = 4.398 - 0.181 * x + rng.normal(0, 0.2, n)          # true b0=4.398, b1=-0.181
X = sm.add_constant(x); res = sm.OLS(y, X).fit()       # minimises sum of squared vertical distances
b0, b1 = res.params; se1 = res.bse[1]; t1 = res.tvalues[1]; p1 = res.pvalues[1]
lo, hi = res.conf_int(0.05)[1]
m = LinearRegression().fit(x.reshape(-1, 1), y)        # cross-check the same two coefficients
print(f"OLS y_hat={b0:.3f}+({b1:.3f})*x  sklearn b0={m.intercept_:.3f} b1={m.coef_[0]:.3f}  R^2={res.rsquared:.4f}")
print(f"RESULT n={n} intercept={b0:.4f} slope={b1:.4f} se_slope={se1:.5f} t_slope={t1:.3f} p={p1:.4g} r_squared={res.rsquared:.4f} exact_ci=({lo:.4f},{hi:.4f})")
```
The seed-42 sample recovers the deer slope near `-0.181` and intercept near `4.398` (here `-0.180`/`4.351`,
Dunn p507). `b0` has the units of `y`; `b1` has units of `y` per unit of `x` (Dunn p505).

### Slope test and CI for beta1 from summary statistics (Dunn p511-515)
```python
from scipy import stats
def infer(b1, se, n, alternative="two-sided", null=0.0):
    df = n - 2; t = (b1 - null) / se                   # t = (b1 - beta1)/s.e.(b1)
    two_p = 2 * stats.t.sf(abs(t), df)
    p = two_p if alternative == "two-sided" else (stats.t.sf(t, df) if alternative == "greater" else stats.t.cdf(t, df))
    tcrit = stats.t.ppf(0.975, df)                     # exact multiplier, df = n - 2
    return t, p, (b1 - 2 * se, b1 + 2 * se), (b1 - tcrit * se, b1 + tcrit * se)
examples = {"red-deer": (-0.181, 0.0289, 78, "two-sided"), "ED": (-0.34790, 0.046672, 30, "two-sided"),
            "sleep-apnoea": (1.3663, 0.65441, 60, "two-sided"), "digestibility": (0.047, 0.001, 36, "greater"),
            "rowers": (1.2, 0.112, 37, "two-sided")}
for label, (b1, se, n, alt) in examples.items():
    t, p, approx, exact = infer(b1, se, n, alt)
    print(f"RESULT {label}: t={t:.3f} {alt}_p={p:.4g} approx95=[{approx[0]:.4f},{approx[1]:.4f}] exact95=[{exact[0]:.4f},{exact[1]:.4f}]")
```
Red deer: `t = -6.26` (Dunn reports `-6.27`) and approximate 95% CI `-0.239` to `-0.123` (Dunn p512-514).
Sleep apnoea: `t = 2.09`, `P = 0.041` -> *moderate* evidence, CI `0.057` to `2.68` (Dunn p518).
One-tailed `P = two-tailed P / 2` only when the RQ pre-specifies direction (Dunn p519).

### Prediction, extrapolation guard and hand checks (Dunn p508-509, p521)
```python
b0, b1, x_range = 4.398, -0.181, (4.4, 14.4)          # deer molar-weight (g) vs age (yr)
for xv in (10, 11, 20):
    yhat = b0 + b1 * xv
    extrap = not (x_range[0] <= xv <= x_range[1])      # Definition 35.1: outside observed x range
    print(f"x={xv}: y_hat={yhat:.3f}{'  [EXTRAPOLATION - unreliable]' if extrap else ''}")
print(f"RESULT deer10={b0 + b1 * 10:.3f} deer20_extrap={b0 + b1 * 20:.3f} rower180={-138 + 1.2 * 180:.1f}")
print(f"RESULT rise_over_run_deer_guess={(17 - 5) / (5 - 1):.1f} cyclone_guess={(8 - 17) / (2 - (-2)):.2f} sunscreen_r={(0.64 ** 0.5):.2f}")
```
Deer age 10 -> `2.588 g`, age 11 -> `2.407 g` (slope subtracted), age 20 -> `0.778 g` but the oldest deer is
`14.4`, so `0.778` is extrapolation and "may be rubbish" (Dunn p508-509). Rowers `y_hat=-138+1.2x`, `x=180`
-> `78.0 kg`; a `|r| = sqrt(R^2)` and the sign of `b1` always equals the sign of `r` (Dunn p507, p521).

### Statistical validity and influential-point diagnostics (Dunn p515-516)
```python
import os, tempfile
import numpy as np, statsmodels.api as sm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import het_breuschpagan
rng = np.random.default_rng(42); n = 40
x = rng.uniform(2, 12, n); y = 1.5 + 0.7 * x + rng.normal(0, 0.8, n)
X = sm.add_constant(x); res = sm.OLS(y, X).fit(); fitted, resid = res.fittedvalues, res.resid
res2 = sm.OLS(y, sm.add_constant(np.column_stack([x, x ** 2]))).fit()
linear_ok = res2.compare_f_test(res)[1] > 0.05          # curvature F-test p > 0.05 -> approximately linear
spread_ok = het_breuschpagan(resid, X)[1] > 0.05        # Breusch-Pagan -> constant variance
size_ok = n >= 25                                       # Dunn rough cut-off (Dunn p515)
infl = res.get_influence(); lev = infl.hat_matrix_diag; cook = infl.cooks_distance[0]
big_cook = int(np.sum(cook > 4 / n)); high_lev = int(np.sum(lev > 2 * 2 / n))
verdict = linear_ok and spread_ok and size_ok
out = tempfile.mkdtemp(); fig, ax = plt.subplots(1, 2, figsize=(8, 3.6))
ax[0].scatter(fitted, resid, s=14); ax[0].axhline(0, color="grey", lw=1); ax[0].set_title("resid vs fitted")
ax[1].scatter(lev, np.abs(infl.resid_studentized_external), s=14); ax[1].set_title("leverage vs |stud|")
fig.tight_layout(); fig.savefig(os.path.join(out, "regression_diagnostics.png"), dpi=110); plt.close(fig)
print(f"linear={linear_ok} spread={spread_ok} size={size_ok} cook>4/n={big_cook} leverage>4/n={high_lev}")
print(f"RESULT verdict={'valid' if verdict else 'caution'}")
```
The seed-42 sample passes all three conditions -> `verdict=valid`; the punting exercise (`n = 13`, Dunn p522)
fails condition 3, so its inference is unreliable despite an impressive line (Dunn p515-516).

## Output contract
Return/report `b0`, `b1`, `se_b1`, `t_stat`, `p_value` (`alternative` = `two-sided`|`greater`|`less`),
`approx_ci_lo`/`approx_ci_hi` (multiplier 2), `exact_ci_lo`/`exact_ci_hi` (t, `df = n-2`), `r_squared`,
`y_hat` per predicted `x` with `is_extrapolation` flag, and the diagnostics verdict
(`linear_ok`, `spread_ok`, `size_ok`, `influential_count`). Print one `RESULT name=value` line per quantity.

## Validity conditions
- Statistically valid only if all three hold: (1) relationship approximately linear, (2) variation in `y`
  approximately constant across `x`, (3) `n >= 25` (Dunn p515). Internal validity = well-designed study;
  external validity = `x` is a simple random sample (Dunn p515).
- Never trust a prediction with `x` outside `[min(x), max(x)]` - label it extrapolation (Dunn p509).
- `b1 = 0` means no linear relationship and `r = 0`; but a near-zero slope can hide a strong curved
  relationship, so always inspect the residual plot (Dunn p510).
- Rounding trap: recompute `t` from un-rounded software `b1`/`s.e.`, not the printed values (Dunn p519).

## Reporting conventions
One sentence for the test plus the CI, e.g. deer: "The sample presents very strong evidence (t = -6.27;
P < 0.001) of a relationship between age and molar weight (slope: -0.181; n = 78; 95% CI from -0.239 to
-0.124) in the population." (Dunn p514-518). Evidence ladder by P: `< 0.001` very strong, `0.001-0.01`
strong, `0.01-0.05` moderate, `> 0.05` little/no (Dunn p513-518). Reframe a small slope per 10 units:
ED slope `-0.348` patients/day reads as `-3.5` patients per 10 days (Dunn p513). Interpret `b0` only when
`x = 0` is inside the observed range (Dunn p509).

## Out of scope
Strength of the association (a single `r`) is the sibling srm-correlation skill (Dunn ch.33-34). Multiple
regression, regularisation or cross-validation, and CI for one mean/proportion or two-group comparisons are
owned by other skills; this file is one-variable slope inference, not an ML pipeline.

## Conventions
- ALPHA = 0.05, MULT = 2, MIN_N = 25; synthetic data only via `numpy.random.default_rng(42)`.
- `statsmodels` OLS gives `s.e.(b1)`, t, P and exact CI; `sklearn` cross-checks the coefficients; headless
  `matplotlib` (`Agg`) writes diagnostics only into a `tempfile` dir.
- The source reads jamovi `Estimate` / SPSS `B` columns; `res.params` is the identical quantity. No extra or
  missing package is required - every number here is computed on the installed stack, never invented.
- Source: Dunn, *Scientific Research and Methodology* (Springer, 2024) ch.35, pp.503-526.
