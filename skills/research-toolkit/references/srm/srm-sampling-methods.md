---
name: srm-sampling-methods
description: Draw and audit research samples in pure Python (numpy.random, pandas.sample) following Dunn ch.5 - simple random, systematic, stratified, cluster and multistage draws from a sampling frame, the hidden-periodicity bias check for systematic sampling, equal vs proportional stratified allocation, representativeness audits of a sample profile against a population profile, and the precision-vs-accuracy distinction. Use when asked how to select a sample, which sampling scheme a study used, whether a sample can represent its population, or whether a big non-random sample beats a small random one.
---

# Sampling Methods - Random, Systematic, Stratified, Cluster, Multistage (Python)

## Purpose
Pick a sample so the estimate is **accurate** (set by the type of sampling) and **precise** (set by the sample
size), and diagnose when it is neither: accuracy = closeness of estimates to the population value on average;
precision = closeness of the estimates to each other (Dunn p86).

## Inputs
- `frame`: the sampling frame = list of all members of the population, length `N` (Dunn p89); `n`: target
  sample size; `k = N // n` the systematic step; `strata` / `clusters` integer labels.
- `SEED = 42` via `numpy.random.default_rng(42)`; `random` means determined completely by chance (Dunn p87).

## Algorithms (all pure numpy/scipy)
### Simple random sample (SRS) - every size-`n` subset equally likely (Dunn p89)
```python
import numpy as np, pandas as pd
N, n = 441, 40                                   # 441 enrolled, sample 40 (Dunn p89-90)
frame = pd.DataFrame({"student_id": np.arange(1, N + 1)})
srs = frame.sample(n=n, random_state=42)
print(f"RESULT srs_n={len(srs)} srs_unique={int(srs.student_id.nunique())} "
      f"srs_lo={int(srs.student_id.min())} srs_hi={int(srs.student_id.max())}")
```
Prints `srs_n=40 srs_unique=40`. SRS needs a usable frame, often impossible: a list of all wombats is absurd.

### Systematic sample - random start, then every k-th (Dunn p90)
```python
import numpy as np
N, n, start = 441, 40, 9                         # book's random start is 9 (Dunn p90)
k = N // n                                       # 441/40 ~ 11
sel = np.arange(start, N + 1, k)[:n]
print(f"RESULT k={k} start={start} first4={sel[:4].tolist()} n_selected={len(sel)} last={int(sel[-1])}")
```
Prints `k=11 first4=[9, 20, 31, 42] n_selected=40 last=438` - start at 9, then every 11th person (Dunn p90),
but systematising buys nothing when taking every individual is just as easy.

### Hidden periodicity turns systematic sampling into bias (Dunn p91)
```python
import numpy as np
rng = np.random.default_rng(42)
floors, per_floor, k = 8, 10, 10                 # eight levels, every 10th apartment (Dunn p91)
start = int(rng.integers(1, k + 1))
units = np.arange(start, floors * per_floor + 1, k)
labels = [f"{int((u - 1) // per_floor) + 1}-{int((u - 1) % per_floor) + 1}" for u in units]
positions = sorted({int((u - 1) % per_floor) + 1 for u in units})
print(f"RESULT start={start} picks={labels} distinct_apartment_positions={positions}")
print(f"RESULT book_case_start10={[f'{f}-10' for f in range(1, floors + 1)]}")
for step in (20, 10, 4, 2):                      # pixel-grid fractions, Dunn p100
    print(f"RESULT systematic_every_{step}_gives={100 / step:g}_percent_sample")
```
One distinct position whatever the start: the book's run picks 1-10, 2-10 ... 8-10, all the larger (likely
wealthier) apartments (Dunn p91); on an image, every 20th/10th/4th/2nd pixel = 5/10/25/50 percent (Dunn p100).

### Stratified sample - SRS inside each stratum (Dunn p92-93)
```python
import numpy as np
rng = np.random.default_rng(42)
F, M, n = 294, 147, 40                           # females and males on the frame (Dunn p92-93)
share = F / (F + M)                              # about two-thirds female
equal = (n // 2, n - n // 2)                     # 20 F + 20 M
prop = (int(round(n * share)), int(round(n * (1 - share))))   # 27 F + 13 M
draw = np.concatenate([rng.choice(F, prop[0], replace=False), rng.choice(M, prop[1], replace=False)])
print(f"RESULT female_share_frame={share:.4f} equal={equal} proportional={prop} total_drawn={draw.size}")
```
Prints `equal=(20, 20) proportional=(27, 13) total_drawn=40`: both stratify by sex, the second matches the
frame (Dunn p92-93). Strata must be unrelated to the variables of interest.

### Cluster and multistage samples (Dunn p93-94)
```python
import numpy as np
rng = np.random.default_rng(42)
n_tut, sizes = 20, rng.integers(15, 26, size=20)   # 20 tutorials, sizes 15-25 (Dunn p93-94)
clust = np.sort(rng.choice(n_tut, size=3, replace=False)) + 1
cluster_n = int(sizes[clust - 1].sum())             # everyone in the chosen clusters
s1 = np.sort(rng.choice(n_tut, size=10, replace=False)) + 1
s2 = 4 * len(s1)                                    # 4 students from each chosen tutorial
print(f"RESULT cluster_tuts={clust.tolist()} cluster_n={cluster_n}")
print(f"RESULT multistage_stage1={s1.tolist()} stage2_per_cluster=4 multistage_n={s2}")
```
Cluster sampling fixes the number of clusters, not the number of people (here `cluster_n` overshoots the
target 40); multistage takes an SRS of clusters then an SRS within each, which is why the ABS samples
cities, then suburbs, then streets, then houses (Dunn p94).

### Accuracy comes from the method, precision from the size (Dunn p86)
```python
import numpy as np
rng = np.random.default_rng(42)
true_mean = 41.0                                   # population mean age, set for this demo
child = rng.normal(12.0, 1.5, size=180_000)        # frame of school children only (Dunn p86)
whole = rng.normal(true_mean, 20.0, size=180_000)  # frame = everyone
def replicate(frame, size, reps):
    est = np.array([frame[rng.choice(len(frame), size, replace=False)].mean() for _ in range(reps)])
    return est.mean() - true_mean, est.std()
big_bias, big_spread = replicate(child, 9000, 20)
rnd_bias, rnd_spread = replicate(whole, 100, 200)
print(f"RESULT n9000_wrong_frame bias={big_bias:.2f} spread={big_spread:.3f}")
print(f"RESULT n100_random_frame bias={rnd_bias:.2f} spread={rnd_spread:.3f}")
```
The 9000-per-sample children frame is precise (tiny spread) and badly inaccurate - it answers "average age
of school children", not of all Queenslanders; the n=100 frame is accurate and imprecise (Dunn p86).

### Representativeness audit - sample profile vs population profile (Dunn p96-97)
```python
import numpy as np, pandas as pd
ev_n = 121                                         # EV study sample, Dunn p97 Table 5.3
tbl = pd.DataFrame({"trait": ["Male", "Female", "Under 18", "18-44", "45-64", "65+",
                              "Income under 75k", "Income 75k+", "Prefer not to say"],
                   "sample_pct": [77.68, 22.32, 0.00, 55.36, 31.25, 13.39, 28.56, 51.78, 19.64],
                   "pop_pct": [49.20, 50.80, 24.00, 36.50, 26.40, 13.00, 67.49, 22.51, 0.00]})
tbl["n_in_sample"] = (tbl.sample_pct * ev_n / 100).round().astype(int)
tbl["ratio"] = (tbl.sample_pct / tbl.pop_pct.replace(0.0, np.nan)).round(2)
print(tbl.to_string(index=False))
print(f"RESULT ev_n={ev_n} male_n={tbl.n_in_sample.iloc[0]} male_ratio={tbl.ratio.iloc[0]} over_rep_count={int((tbl.ratio > 1).sum())}")
fb_pct, fb_n, sm_pct, sm_n = 0.79, 53000, 0.837, 1809    # voluntary poll vs paid panel (Dunn p88)
vol_n, rand_n = 10_000_000, 50_000                       # 1936 voluntary vs random survey (Dunn p85)
print(f"RESULT facebook_oppose_n={fb_pct * fb_n:.0f} panel_agree_n={round(sm_pct * sm_n)}")
print(f"RESULT voluntary_1936_n={vol_n} random_1936_n={rand_n} ratio={vol_n / rand_n:.0f}")
```
Prints `male_n=94 male_ratio=1.58`: the non-random social-media sample over-represents males, the 18-44
group and wealthier respondents, so its results apply mainly to people with a technical background.

## Output contract
Per draw: `method` (`srs` | `systematic` | `stratified` | `cluster` | `multistage`), `frame_size`,
`n_target`, `selected_ids`, `k`, `allocation` (stratified counts per stratum), `cluster_ids`, `n_selected`
(cluster: only `cluster_ids` is fixed in advance). Per audit: `sample_pct`, `pop_pct`, `ratio`,
`over_or_under`, `externally_valid` (bool). Print one `RESULT name=value` line per quantity; write any
selected-ids CSV only inside a `tempfile` dir.

## Validity conditions
- A frame must exist and match the target population; using QLD/NT residents to estimate air-conditioner
  use for all Australians is selection bias from the wrong frame (Dunn p95, p98).
- Non-random samples (judgement, convenience, voluntary) may not generalise; they generalise only to the
  population they do represent (Dunn p88).
- A large sample is not always better than a random one: the 1936 voluntary survey of 10,000,000 vs the
  random survey of 50,000, and a 53,000-response voluntary poll, both show size cannot fix method (Dunn p85, p88, p101).
- Stratifying by the explanatory variable is not stratified sampling for a RQ comparing its levels (Dunn p92);
  systematic needs no hidden cycle (Dunn p91); random is preferred, but representative is often the best
  achievable (Dunn p95-96).
- Selecting at several times/places (Cafe A Monday, Cafe B Tuesday, Library Thursday) improves coverage but
  is still not random and can double-count (Dunn p96).

## Conventions
- SEED = 42 everywhere (`numpy.random.default_rng(42)`, `random_state=42` for pandas); demo frames only.
- `k = N // n` (floor); stratified counts via `round()` and re-checked to sum to `n`; figures headless with
  `matplotlib.use("Agg")`; nothing written outside a `tempfile` dir.
- Only `numpy`/`pandas` are needed; a `sklearn` resampler is replaced inline by `rng.choice(..., replace=False)`.
- Never invent a population value - compute it from a seeded frame and print the `RESULT` lines above;
  report the sampling scheme, the frame, `n` and the population the results can generalise to (Dunn p96-97).
- Selection bias is audited here; non-response and response bias are defined in the source but handled by
  the collection-protocol skill (Dunn p98).
- Source: Dunn, Scientific Research and Methodology (Springer, 2024) Ch.5, p83-102.

## Out of scope
Sampling variation, sampling distributions and standard errors (sibling sampling-variation skill), CIs and
tests that consume a sample, and survey instrument/collection protocols (sibling data-collection skill).
