---
name: srm-distributions-normal-model
description: Normal-distribution modelling in pure Python (numpy/scipy). Standardises an observation to a z-score with z=(x-mu)/sigma, reads exact left/right/between-tail areas with scipy.stats.norm.cdf and compares them to the 68-95-99.7 approximation, then works backwards from a percentile with the unstandardising formula x=mu+z*sigma. Use when asked for a probability/proportion under a normal model, a z-score or standardised value, an area under the normal curve, or the value cutting off the smallest/largest k%.
---

# Distributions and the Normal Model (Python)

## Purpose
Describe a population with a normal model (mean `mu`, sd `sigma`) and answer probability questions:
standardise to `z`, find the area under the curve, or invert an area back to a value. The bell-shaped
normal model is a theoretical description of the population, not any single sample (Dunn p274-275).

## Inputs
- `MU`, `SIGMA`: the model's theoretical mean and standard deviation (population parameters, cm/inches/mm).
- `x`: an observed value to standardise (front-to-back direction), OR
- a target left-tail area/percentile to work backwards from (back-to-front direction).
- z-scores are read to two decimal places, as the printed tables do (Dunn p279).

## Workflow (Dunn p281)
Draw the curve and mark `x`; shade the region of interest; compute `z` with Eq. (17.1); look the area up
in the normal table (area is always to the LEFT of `z`); combine left-areas to get the answer. The total
area under a normal curve is one (Dunn p279, p282).

## Algorithms (all pure numpy/scipy)

### Standardise and read the left-tail area
```python
from scipy import stats
MU, SIGMA = 175, 7                       # model: heights of Australian adult males (Dunn p274-275)
for x in (182, 161, 160):
    z = round((x - MU) / SIGMA, 2)       # Eq. (17.1) z=(x-mu)/sigma, to 2 dp like the tables
    print(f"RESULT x={x} z={z:.2f} area_left={stats.norm.cdf(z):.4f}")
```
Prints `z=1.00 area_left=0.8413` (so P(taller than 182)=0.1587, "about 16%", Dunn p276), `z=-2.00
area_left=0.0228` (exact table value, Dunn p279-280), and `z=-2.14 area_left=0.0162` (the 160 cm case,
"about 1.6%", Example 17.6, Dunn p281). A negative `z` means the value is below the mean; `z` has no units (Dunn p277-278).

### Approximate (68-95-99.7) vs exact area
```python
import numpy as np
from scipy import stats
rng = np.random.default_rng(42)          # seeded Monte-Carlo check of the empirical rule
h = rng.normal(175, 7, size=200000)      # the model, sampled large (Dunn p275-276)
within = [round(float(((h > 175 - k*7) & (h < 175 + k*7)).mean()), 3) for k in (1, 2, 3)]
exact_tail = round(float(stats.norm.cdf(-2.0)), 4)   # exact, not the rule's 2.5% (Dunn p279-280)
print(f"RESULT within_123sd={within} rule=0.68/0.95/0.997 approx_below2sd=0.025 exact_below2sd={exact_tail}")
```
The 68-95-99.7 rule is an approximation only: it says ~2.5% lie below 2 sd (Dunn p277), but the exact
table area is `0.0228` (2.28%, Dunn p279-280). Sample histograms vary around this model (Dunn p275).

### Right-tail and between-tail areas (tree diameters)
```python
from scipy import stats
MU, SIGMA = 8.8, 2.7                     # model: tree diameters, Aedo-Ortiz (Dunn p281)
z_lo = round((6 - MU) / SIGMA, 2)        # = -1.04
z_hi = round((11 - MU) / SIGMA, 2)       # = 0.81
left6 = stats.norm.cdf(z_lo)             # 0.1492 (Dunn p282)
print(f"RESULT z_lo={z_lo:.2f} z_hi={z_hi:.2f} P(diam>6)={1-left6:.4f} P(6<diam<11)={stats.norm.cdf(z_hi)-left6:.4f}")
```
`P(diam>6)=0.8508` ("about 85%", Example 17.7, Dunn p282) comes from 1 - (left area), since tables give
only the left area. `P(6<diam<11)=0.6418` ("about 64%", Example 17.8) is the difference of two left areas,
0.7910 - 0.1492 (Dunn p283).

### Unstandardising: working backwards from a percentile
```python
from scipy import stats
MU, SIGMA = 8.8, 2.7                     # tree diameters (Dunn p281)
z10 = round(stats.norm.ppf(0.10), 2)     # smallest 10% -> table z=-1.28 (Dunn p285)
x10 = MU + z10 * SIGMA                   # Eq. (17.2) x=mu+z*sigma -> 5.344 (Dunn p286)
z75 = round(stats.norm.ppf(0.75), 3)     # largest 25% = left area 75% -> z=0.674 (Dunn p287)
x75 = MU + z75 * SIGMA                   # -> about 10.6 inches (Dunn p287)
print(f"RESULT z_p10={z10:.2f} x_smallest10={x10:.3f} z_p75={z75:.3f} x_largest25={x75:.3f}")
```
Here the probability is known and the value is sought, so the table is read backwards: closest area 0.1003
gives `z=-1.28` and `x=5.344` inches for the smallest 10% of trees (Example 17.9, Dunn p285-286); the
largest 25% uses the left area 0.75, `z=0.674` (Example 17.10, Dunn p287).

### Backwards check on another model (ball bearings)
```python
from scipy import stats
MU, SIGMA = 50.0, 0.1                    # "50mm" bearings, sd 0.1mm (Dunn p286)
z = round(stats.norm.ppf(0.15), 2)       # smallest 15% -> table z=-1.04 (Dunn p286)
print(f"RESULT z={z:.2f} min_sellable_diameter={MU + z*SIGMA:.3f}")
```
Prints `z=-1.04 min_sellable_diameter=49.896`: bearings below about 49.9 mm cannot be sold (Think 17.2,
Dunn p286) - the same left-area 0.1492 seen for the tree tail.

## Output contract
Return/report per question: `z` (2-dp standardised value), `area_left`, `area_right` (=1-left),
`area_between` (difference of two left areas); or, working backwards, `z_percentile` and `x = mu + z*sigma`.
Print one `RESULT name=value` line per quantity; state proportions as decimals and as "about N%".

## Validity conditions
- The normal distribution is a MODEL: no real variable is exactly normal, but the idealised shape is a
  useful description of the population (Dunn p275, p277).
- If it fits poorly, adjust `mu`/`sigma` until the model matches many samples (Dunn p276).
- Percentages depend only on the number of standard deviations from the mean, not on the values of `mu`, `sigma` (Dunn p276-277).

## Conventions
- `scipy.stats.norm.cdf` / `.ppf` replace the printed R/online z-tables; no extra package is needed.
- Standardise by rounding `z` to 2 dp before the lookup (tables use two decimals, Dunn p279); for backwards
  work round `ppf` to the 2-3 dp table value (e.g. `z=-1.28`, `z=0.674`) before unstandardising.
- Simulated demo data seeded only with `numpy.random.default_rng(42)`; never type a result - compute it and
  print the `RESULT` lines above. Draw any curve headless: `matplotlib.use("Agg")`.
- Source: Dunn, *Scientific Research and Methodology* (Springer, 2024) Ch.17 "Distributions and models", pp.273-290.

## Out of scope
Confidence intervals for a mean/proportion (sibling CI skill), sampling distributions of the statistic
(CLT), hypothesis tests, and non-normal models (binomial etc.) are not handled here - this skill is single-
observation normal modelling and z/area arithmetic only (Dunn p273-290).
