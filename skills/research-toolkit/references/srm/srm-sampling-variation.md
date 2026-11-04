---
name: srm-sampling-variation
description: Sampling distributions and standard errors of sample proportions and sample means, pure Python (numpy/scipy). Simulates repeated seeded samples from a roulette wheel to show how p_hat and x_bar vary from sample to sample, checks the simulated standard error against sqrt(p(1-p)/n) and sigma/sqrt(n), contrasts standard deviation with standard error, and bootstrap-estimates the SE of any statistic (proportion, odds, mean, median, sd). Use when asked what a sampling distribution or sampling variation is, why a larger n gives a more precise estimate, whether to report SD or SE, or how to simulate repeated sampling.
---

# Sampling Variation and Standard Errors (Python)

## Purpose
Show how a sample statistic varies from sample to sample (sampling variation), measure that variation with a
standard error, and keep the standard error apart from the standard deviation of individual observations
(Dunn p291-297).

## Inputs
- `WHEEL = np.arange(37)`: the population being sampled - a European roulette wheel, numbers 0-36, where the
  classical probability of an odd-spin is p = 18/37 = 0.486 (Dunn p292). Any 1-D numeric array substitutes.
- `n`: sample size(s) to compare. The book's figures use n = 15, 30, 70, 100 for the proportion and
  n = 15, 50, 100, 250 for the mean (Dunn p293, p295).
- `REPS`/`B` = 20000 simulated samples / resamples; `SEED = 42`. No ALPHA here - forming intervals is a
  sibling skill.

## Algorithms (all pure numpy/scipy)

### Sampling distribution of a sample proportion (Dunn p292-294)
```python
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, tempfile
rng = np.random.default_rng(42)
WHEEL = np.arange(37)                              # roulette numbers 0-36 (Dunn p292)
P = (WHEEL % 2 == 1).mean()                        # = 18/37 odd spins
REPS, NS = 20000, (15, 30, 70, 100)               # panel sizes of Fig. 18.2 (Dunn p293)
fig, axes = plt.subplots(2, 2, figsize=(8, 5))
for ax, n in zip(axes.ravel(), NS):
    spins = rng.choice(WHEEL, size=(REPS, n))      # REPS samples of n spins each
    phat = (spins % 2 == 1).mean(axis=1)
    ax.hist(phat, bins=25, color="#4C72B0")
    ax.set_title(f"n = {n}")
    print(f"RESULT n={n} mean_phat={phat.mean():.4f} se_sim={phat.std(ddof=1):.4f} "
          f"se_formula={np.sqrt(P*(1-P)/n):.4f}")
png = os.path.join(tempfile.mkdtemp(prefix="sv_"), "phat-sampling-distributions.png")
fig.tight_layout(); fig.savefig(png, dpi=110); plt.close(fig)
print(f"RESULT p_odd={P:.3f} png_bytes={os.path.getsize(png)}")
```
Prints `p_odd=0.486`; at n = 15 the simulated SE (0.1283) matches sqrt(p(1-p)/n) (0.1291) and at n = 100 both
are 0.050 - the spread of `p_hat` shrinks and its histogram becomes bell-shaped as n grows (Dunn p292-293).

### Sampling distribution of a sample mean (Dunn p294-296)
```python
import numpy as np
from scipy import stats
rng = np.random.default_rng(42)
WHEEL = np.arange(37)
MU, SIGMA = WHEEL.mean(), WHEEL.std(ddof=0)        # population mean 18.0, population sd 10.677
for n in (15, 50, 100, 250):                       # panel sizes of Fig. 18.3 (Dunn p295)
    xbar = rng.choice(WHEEL, size=(20000, n)).mean(axis=1)
    print(f"RESULT n={n} mean_xbar={xbar.mean():.3f} se_sim={xbar.std(ddof=1):.3f} "
          f"se_formula={SIGMA/np.sqrt(n):.3f} sd_recovered={xbar.std(ddof=1)*np.sqrt(n):.2f} "
          f"skew={stats.skew(xbar):.3f}")
print(f"RESULT pop_mean={MU:.1f} pop_sd={SIGMA:.3f} pop_skew={stats.skew(WHEEL):.3f}")
```
The population is flat (skew 0.000) and its sd is 10.677, yet `x_bar` is near-normal even at n = 15 and
`sd_recovered` = se_sim x sqrt(n) is 10.66-10.74 at every n: the SE shrinks as `sigma/sqrt(n)` while the
population sd is unchanged (Dunn p294-296).

### Standard deviation vs standard error (Dunn p296-297)
```python
import numpy as np
from scipy import stats
rng = np.random.default_rng(42)
n = 200
minutes = rng.normal(47, 12, n)
minutes = minutes - minutes.mean() + 47            # demo sample with mean 47 min (Dunn p298 Ex18.1)
sd, se = minutes.std(ddof=1), stats.sem(minutes)   # s.e. of the mean = sd/sqrt(n)
print(f"RESULT mean={minutes.mean():.2f} n={n} sd={sd:.3f} se={se:.4f} sd_over_se={sd/se:.2f} "
      f"sqrt_n={np.sqrt(n):.2f}")
phat = 0.51                                        # Example 18.1 (Dunn p296)
for se_p in (0.01, 0.2):
    print(f"RESULT phat={phat} se={se_p} n_implied={phat*(1-phat)/se_p**2:.1f}")
```
`sd_over_se=14.14` equals `sqrt_n=14.14`, the whole difference between the two measures. Example 18.1's two
standard errors for the same `p_hat = 0.51`, 0.01 and 0.2, correspond to about 2499 and 6 spins: the small SE
means the estimate barely moves from sample to sample, the large one means any single estimate may be far from
p (Dunn p296).

### Every statistic has a standard error - bootstrap (Dunn p296-297)
```python
import numpy as np
rng = np.random.default_rng(42)
WHEEL = np.arange(37)
n, B = 100, 20000                                  # one sample of n = 100 spins (Dunn p292)
spins = rng.choice(WHEEL, size=n)
phat = (spins % 2 == 1).mean()
boot = spins[rng.integers(0, n, size=(B, n))]      # B resamples with replacement, each of size n
p = (boot % 2 == 1).mean(axis=1)
se_of = {"p_hat": p, "odds": p / (1 - p), "x_bar": boot.mean(axis=1),
         "median": np.median(boot, axis=1), "sd": boot.std(axis=1, ddof=1)}
for name, v in se_of.items():
    print(f"RESULT se_{name}={v.std(ddof=1):.4f}")
print(f"RESULT n={n} phat={phat:.3f} se_formula_phat={np.sqrt(phat*(1-phat)/n):.4f}")
```
The sample gives `phat=0.500`; the bootstrapped SE (0.0498) reproduces the formula value 0.0500, and the same
simulation yields SEs for the odds (0.2089), the median (2.2021) and s (0.4627), which have no simple formula
- the mean, proportion, odds ratio, median and sd are all statistics, so all have a standard error (Dunn p297).

## Output contract
Per statistic report `statistic`, `estimate`, `se`, `se_method` (`simulation` | `formula` | `bootstrap`), `n`,
`reps`; for an n-comparison report the paired list `(n, se)` showing SE falling as n rises. One `RESULT
name=value` line per quantity. Any figure is a 2x2 panel PNG (one panel per n) written into a `tempfile`
directory; return its path, never a working file.

## Validity conditions
- A sampling distribution is the distribution of the *statistic over all possible samples*, not of the
  observations (Dunn p292, p294).
- The normal description holds only under certain conditions, and for a proportion the approximation gets
  better as n gets larger (Dunn p292, p294).
- The SE is defined for samples "found a certain way, and of a given size": a different selection method or a
  different n is a different sampling distribution, so recompute it (Dunn p296).
- Larger n gives a smaller SE and a more precise estimate, all else being equal (Dunn p295-296).
- Population parameters (p, mu, population odds) have no standard error; "standard error of the population
  proportion" is inappropriate (Dunn p297, p298).

## Reporting conventions
- Abbreviate to SE or s.e. and always name the statistic: `s.e.(x_bar)`, `s.e.(p_hat)` (Dunn p297).
- SD quantifies how individuals vary; SE quantifies how the estimate varies; they give two different types of
  information, so never use the SE to describe the variability of the sample (Dunn p296-298).
- "Error" here is not a mistake, and nothing is non-standard about it (Dunn p296).
- Choosing between them (Dunn p298 Ex 18.1): variation of spending per transaction or of children's ages in a
  sample = SD; precision of a reported mean of 47 minutes or of a reported percentage = SE.

## Out of scope
Turning an SE into a margin of error or a confidence interval, and the count/size validity checks, belong to
the single-sample and comparison CI skills; how the sample was chosen (simple random, systematic, cluster,
convenience) and non-sampling error belong to the sampling-methods skill; describing individuals with an SD
belongs to the numerical-summaries skill; tests that consume the SE belong to the hypothesis-testing skills.

## Conventions
- `numpy.random.default_rng(42)`; `REPS = B = 20000`; `ddof=1` for simulated/resampled SDs, `ddof=0` for the
  population SD.
- Proportions entered as 0-1, never percentages; `s.e.(p_hat) = sqrt(p(1-p)/n)` and
  `s.e.(x_bar) = sigma/sqrt(n)` (Dunn p293-296).
- Headless matplotlib (`Agg`) plus `tempfile` for output; `scipy.stats.sem` for the closed-form `s.e.(x_bar)`.
- The book simulates repeated sampling in general-purpose software; nothing beyond numpy/scipy is needed - the
  repeated sampling and the bootstrap are hand-rolled here with `rng.integers` indices.
- Never quote an SE without its statistic, its n and its method; compute it, do not assume it.
- Source: Dunn, *Scientific Research and Methodology* (Springer, 2024) Ch.18, pp.291-298.
