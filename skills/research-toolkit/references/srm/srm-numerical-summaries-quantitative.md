---
name: srm-numerical-summaries-quantitative
description: Numerically describe ONE quantitative variable, or many BY GROUP, in pure Python (numpy/pandas/scipy) - mean vs median, range, sample standard deviation and variance, quartiles/IQR/percentiles, distribution shape in words, the 68-95-99.7, standard-deviation and IQR outlier rules, plus a pandas groupby summary table and a weighted (pooled) mean. Use when asked to "summarise/ describe quantitative data numerically", compute mean/median/sd/variance/IQR/percentiles, standardise to a z-score, flag outliers, or build a summary or "baseline characteristics" table.
---

# Numerical Summaries - Quantitative Data (Python)

## Purpose
Describe a quantitative variable along four features - average, variation, shape, unusual values - using
statistics computed from a sample. A number describing a sample is a statistic; the population value it
estimates (mean mu, sd sigma) is a parameter and is almost always unknown (Dunn p210, p214).

## Inputs
- `x`: one numeric column/array (single group), or a DataFrame with a numeric value column + a grouping
  column (many groups). Quantitative data only; do not average labels or categories (Dunn p234).
- Optional summary-stat inputs for the outlier rules: `Q1`, `Q3`, `mean`, `sd`.
- `DDOF = 1` (sample variance/sd divide by n-1; s estimates the unknown sigma, Dunn p218).

## Algorithms (all pure numpy/scipy)

### One group - average and spread (Dunn p212-218)
```python
import numpy as np
x = np.array([4.8, 5.2, 5.2, 5.4, 5.2, 6.5, 4.5, 5.7, 4.8, 5.2])   # Jersey cow butterfat %, Dunn p212
mean = x.mean(); median = np.median(x); rng = x.max() - x.min()
variance = x.var(ddof=1); sd = x.std(ddof=1)
print(f"RESULT mean={mean:.2f} median={median:.2f} range={rng:.2f} variance={variance:.4f} sd={sd:.3f}")
```
Prints mean=5.25 (Dunn p213), median=5.20 (Dunn p215), range=2.00 (Dunn p217), variance=0.3072 and
sd=0.554% (Dunn p218). Mean = sum(x)/n (the "balance point"); range = max-min uses only the two extremes,
so it is heavily swayed by outliers and is rarely quoted alone (Dunn p217).

### Median position rule, even vs odd n (Dunn p214-215)
```python
import numpy as np
x = np.array([484, 492, 436, 464, 478, 444, 398, 476])   # corneal thickness microns, Dunn p214
n = len(x); pos = (n + 1) / 2                            # median sits at ordered observation (n+1)/2
median = np.median(x)                                    # even n: midway between the 4th and 5th (464,476)
print(f"RESULT n={n} median_position={pos} median={median:.0f}")
```
Prints median_position=4.5 and median=470 microns (Dunn p215). Odd n -> the single middle value; even n ->
halfway between the two middle values (some software differ, Dunn p215).

### Quartiles, percentiles and z-scores (Dunn p220-222)
```python
import numpy as np
x = np.array([4.8, 5.2, 5.2, 5.4, 5.2, 6.5, 4.5, 5.7, 4.8, 5.2])   # butterfat %, Dunn p212
q1 = np.percentile(x, 25); q2 = np.percentile(x, 50); q3 = np.percentile(x, 75)
iqr = q3 - q1                         # IQR = Q3 - Q1 = the spread of the middle 50%
p90 = np.percentile(x, 90)            # 90th percentile separates the smallest 90% from the rest
z = (x - x.mean()) / x.std(ddof=1)    # standardised value (no. of sample sd from the mean)
print(f"RESULT p50_median={q2:.2f} Q1={q1:.2f} Q3={q3:.2f} IQR={iqr:.2f} p90={p90:.2f} z_at_mean={z[3]:.1f}")
```
Q1, Q2, Q3 are the 25th/50th/75th percentiles, so p50_median=5.20 reproduces the sample median (Dunn p215,
p222); IQR is the range of the middle 50% and is not influenced by outliers (Dunn p220). A z-score counts
standard deviations from the mean; the 68-95-99.7 thresholds below use integer z multipliers (Dunn p226).

### IQR rule - mild and extreme outlier fences (Dunn p228)
```python
import numpy as np
Q1, Q3 = 24.7, 50.6                              # permeability (s), n=81 sheets, Dunn p228
iqr = np.round(Q3 - Q1, 1)                        # = 25.9
mild_lo, mild_hi = np.round(Q1 - 1.5*iqr, 1), np.round(Q3 + 1.5*iqr, 1)
ext_lo,  ext_hi  = np.round(Q1 - 3.0*iqr, 1), np.round(Q3 + 3.0*iqr, 1)
print(f"RESULT IQR={iqr} mild_fence=[{mild_lo},{mild_hi}] extreme_fence=[{ext_lo},{ext_hi}]")
```
Prints IQR=25.9, mild fence [-14.2, 89.5], extreme fence [-53.0, 128.3] (Dunn p228): a mild outlier lies
>1.5xIQR beyond Q1/Q3, an extreme one >3xIQR (Dunn p227).

### Standard-deviation rule and the 68-95-99.7 rule (Dunn p226)
```python
import numpy as np
from scipy import stats
mean, sd = 173.2, 6.88                            # UK husbands' heights (cm), Dunn p226
for k in (1, 2, 3):                               # k = the z multiplier of the empirical rule
    lo, hi = round(mean - k*sd, 1), round(mean + k*sd, 1)
    exact = (2*stats.norm.cdf(k) - 1) * 100       # exact area within k sd of the mean
    print(f"RESULT k={k} bounds=[{lo},{hi}] exact_within={exact:.2f}")
```
k=1 -> [166.3, 180.1] (68% of data), k=2 -> [159.4, 187.0] (95%), k=3 -> [152.6, 193.8] (99.7%) - the
empirical rule; the exact areas 68.27/95.45/99.73 match Dunn p226. For approx-symmetric data any value
beyond 3 sd from the mean may be flagged an outlier; some texts use 2.5 sd - all such rules are arbitrary
(Dunn p226-227).

### Many groups - summary table and weighted mean (Dunn p232)
```python
import numpy as np, pandas as pd
rows = {"All participants": (8474, 1.36, 1.29, 0.399, 0.49),
        "Smokers":          (1388, 1.31, 1.24, 0.424, 0.52),
        "Non-smokers":      (1668, 1.39, 1.32, 0.428, 0.54)}   # direct HDL mmol/L, Dunn Table 13.8 p232
cols = ["n", "mean", "median", "sd", "iqr"]
df = pd.DataFrame.from_dict(rows, orient="index", columns=cols)   # with raw data use df.groupby(g).describe()
wmean = np.average([1.31, 1.39], weights=[1388, 1668])            # pooled mean of the two smoking groups
print(f"RESULT overall_mean={df.loc['All participants','mean']:.2f} overall_sd={df.loc['All participants','sd']:.3f} "
      f"overall_iqr={df.loc['All participants','iqr']:.2f} weighted_mean_2groups={wmean:.3f}")
print(df.to_string())
```
Echoes the table overall mean=1.36, sd=0.399, IQR=0.49 and group means 1.31/1.39 (Dunn p232). The weighted
mean weights each group mean by its n; it gives 1.354, which differs from the 1.36 all-participants value
because smoking status was unavailable for the rest - missing data can shift a summary (Dunn p232).

## Output contract
Return, per group: `n`, `mean`, `median`, `range`, `variance`, `sd`, `Q1`, `Q3`, `iqr`, requested
`percentiles`, and outlier fences (`mild_lo/mild_hi`, `ext_lo/ext_hi`) with flagged values. For many groups
return a tidy DataFrame (index = group, columns = n/mean/median/sd/iqr) plus a `weighted_mean`. Print one
`RESULT name=value` line per quantity; round only the reported answer, not the working (Dunn p214).

## Validity conditions
- Mean and sd are sensible only for symmetric data; both are pulled by skewness and outliers, so for
  skewed data report median and IQR instead (Dunn p216, p234). The streamflow example shows this: mean
  1123.2 ML vs median 146.1 ML (Dunn p211).
- The SD rule and 68-95-99.7 assume an approximately bell-shaped distribution; use the IQR rule otherwise
  (Dunn p227-228). Every outlier rule is arbitrary (Dunn p227).
- A statistic estimates an unknown parameter; the population mu/sigma are never known from the sample, and
  sampling variation alone can make two sample means differ (Dunn p210, p232-233).

## Reporting conventions
- Say "mean" or "median", not "average" (both can be meant, Dunn p211). Quote mean +/- sd or median +/-
  IQR and label which (Dunn p230). Round to one or two decimals beyond the raw data (Dunn p214).
- Describe shape in words only (symmetric, right/left-skewed, bimodal), not a skewness number (Dunn p223,
  Table 13.9 p234). In tables put the caption above, align on the decimal point, use few lines (Dunn p229).

## Out of scope
Graphs (histograms, boxplots, dot plots) belong to the graphical-summaries skill; full normal-distribution
modelling (CDF tail areas, inverse percentile from a z) to the normal-model skill; CIs and hypothesis tests
about means to the inference skills; correlations/regressions to their own skills; categorical and
qualitative summaries to the qualitative-numerical-summaries skill. No extra package is needed - numpy,
pandas and scipy.stats cover every rule above (the text's jamovi/SPSS output is reproduced in code).

## Conventions
- `DDOF=1` for sample variance/sd; `numpy.percentile` default linear interpolation (software may differ,
  Dunn p215); mean/median and sd/IQR chosen by symmetry (Dunn p234).
- Never hardcode a summary - compute it and print the `RESULT` line, citing the source page for the method.
- Demo data are the actual worked sets from the source (butterfat, corneal, permeability, heights); use
  `numpy.random.default_rng(42)` only for any synthetic filler, and `matplotlib.use("Agg")` if a plot is
  forced despite being out of scope.
- Source: Dunn, *Scientific Research and Methodology* (Springer, 2024) Ch.13, pp.209-236.
