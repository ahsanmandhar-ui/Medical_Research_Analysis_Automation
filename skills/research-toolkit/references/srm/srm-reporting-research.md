---
name: srm-reporting-research
description: Build and QC the written/scientific report following Dunn's structure, checklists and critical-reading rules - assemble an AIMRaD/STARD-compliant results section, produce caption-correct tables (GFM markdown via pandas, no vertical lines, caption above) and headless figures (caption below, units, no 3-D), verify reported numbers are internally consistent (CI centred on the mean, +/- is SE not SD, unit conversions add up), and read a paper with the six-step/POCI guide, in pure Python (numpy/pandas/matplotlib). Use when asked to write or critique a results section, structure a paper, build a compliant table or figure, or "read this abstract critically".
---

# Writing & Reading Research Reports (Python)

## Purpose
Assemble and quality-check the reporting layer of a study: the article structure, compliant tables and
figures, the wording of results, and a critical read of someone else's paper. It does not compute the
statistics themselves - it structures and audits them (Dunn Ch.37 p537-552, Ch.36 p529-536).

## Inputs
- `sections`/`abstract_parts`: lists of article & structured-abstract headings to check (Dunn p539, p541).
- summary table rows as `(level, n, mean, sd, [ci_lo, ci_hi])` - already-computed statistics.
- reported constants to audit: means, +/- spread, CI endpoints, unit conversions from an abstract.
- figure output directory: any `tempfile` dir; no path outside it is written.

## Algorithms (all pure numpy/scipy)

### Article structure + STARD abstract completeness
Check the AIMRaD skeleton and the STARD items a structured abstract should carry, then reproduce a
corpus figure: Singh & Prasad followed N = 105 amputees, 35 died in 3 years = 33% mortality (Dunn p541-542).
```python
import numpy as np
rng = np.random.default_rng(42)              # seeded per convention; no randomness used
N, died = 105, 35                             # Singh & Prasad 2016, Dunn p541-542
mortality = died / N                          # reported as 33%
SECTIONS = ["Title","Abstract","Introduction","Method","Results","Discussion","References"]
REQUIRED_ABSTRACT = ["Background","Objective","Study design","Methods","Results","Conclusion"]
draft = list(REQUIRED_ABSTRACT)               # structured abstract present
struct_ok = all(s in SECTIONS for s in ["Abstract","Introduction","Method","Results","Discussion"])
abstract_ok = all(k in draft for k in REQUIRED_ABSTRACT)   # smoke assert of required headings
assert struct_ok and abstract_ok
print(f"RESULT AIMRaD_ok={struct_ok} STARD_abstract_ok={abstract_ok} mortality_pct={round(100*mortality)} N={N} died={died}")
```
Prints `mortality_pct=33 N=105 died=35` (Dunn p541-542).

### Build a compliant table (caption above, decimals aligned, no vertical lines)
`pandas.to_markdown` needs the `tabulate` package which is not installed, so the GitHub-markdown builder
is reimplemented inline. Audit the reported spread: in the paper-plane example the CI 829.29-934.39 is
centred on the mean 881.84, so the displayed `881.84 +/- 140.73 ms` is the SD, not the CI half-width (Dunn p551).
```python
import pandas as pd, numpy as np
lo, hi, mean_rep, pm = 829.29, 934.39, 881.84, 140.73   # Exercise 37.5, Dunn p551
midpoint, halfwidth = (lo+hi)/2, (hi-lo)/2              # CI centre vs +/- value
def to_markdown_gfm(df):                               # inline GFM builder (tabulate missing)
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    rule = "|" + "|".join(["---"]*len(cols)) + "|"     # no vertical lines (checklist E.2)
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, rule] + rows)
tbl = pd.DataFrame({"Design":["Basic Dart","Hunting Flight"],
                    "Mean (ms)":[mean_rep, 1504.19],
                    "95% CI":[f"{lo}-{hi}", "1242.86-1765.52"]})
md = to_markdown_gfm(tbl)                              # caption goes ABOVE the table (Dunn p545)
assert md.splitlines()[0].startswith("| Design")
print(f"RESULT ci_centred_on_mean={bool(np.isclose(midpoint, mean_rep))} midpoint={midpoint:.2f} spread_sd={pm:.2f} vs ci_halfwidth={halfwidth:.2f} rows={len(tbl)}")
```
Prints `ci_centred_on_mean=True midpoint=881.84` (Dunn p551).

### Export a checklist-compliant figure (caption below, units, linear scale, no 3-D)
Bar chart of the Hearing Loss Questionnaire means by earphone use from Table 36.1 (Yes n=745 mean 19.8;
No n=100 mean 19.0), saved headless into a temp dir (Dunn p534, graphics checklist E.1 p591).
```python
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, tempfile, os
rng = np.random.default_rng(42)
levels = ["Yes","No"]; means = np.array([19.8, 19.0]); ns = np.array([745, 100]); sds = np.array([3.08, 1.71])
fig, ax = plt.subplots(figsize=(4,3))                 # E.1: simple, one message, faint grid
ax.bar(levels, means, yerr=sds, capsize=4, color=["#4C72B0","#DD8452"])
ax.set_ylabel("Hearing loss score (units: 17-34 scale)"); ax.set_xlabel("Earphone use")
ax.set_title("Mean hearing loss by earphone use")     # caption is written BELOW in the report
ax.grid(axis="y", alpha=0.25)
d = tempfile.mkdtemp(); png = os.path.join(d, "fig.png")
fig.tight_layout(); fig.savefig(png); plt.close(fig)  # no chart junk, no artificial 3-D (Dunn p545)
print(f"RESULT fig_saved={os.path.exists(png)} yes_mean={means[0]} yes_n={int(ns[0])}")
```
Prints `yes_mean=19.8 yes_n=745` (Table 36.1, Dunn p534).

### Read a paper critically: verify the abstract's own numbers
Six-step / POCI reading; check the unit conversions reported in the Groves bicycle abstract before
trusting it (27 mi = 43.5 km; steel 809 mi = 1302 km; carbon 711 mi = 1144 km; 20.9 lb = 9.5 kg;
29.75 lb = 13.5 kg) (Dunn p531-532).
```python
import numpy as np
rng = np.random.default_rng(42)
MI_KM, LB_KG = 1.609344, 0.45359237
checks = [("27 mi", 27*MI_KM, 43.5), ("809 mi", 809*MI_KM, 1302), ("711 mi", 711*MI_KM, 1144),
          ("20.9 lb", 20.9*LB_KG, 9.5), ("29.75 lb", 29.75*LB_KG, 13.5)]
consistent = [abs(round(calc)-rep) <= 1 for _, calc, rep in checks]
print("RESULT conversions_consistent=" + str(all(consistent)) + " " + " ".join(f"{n}={c:.1f}~{r}" for n, c, r in checks))
```
Prints `conversions_consistent=True` with 43.5, 1302, 1144 reproduced (Dunn p531-532).

## Output contract
Return/print, per quantity, a `RESULT name=value` line. Produce: `AIMRaD_ok`, `STARD_abstract_ok`,
`mortality_pct`; a markdown table string (caption above, decimal-aligned, no vertical lines) plus
`ci_centred_on_mean`, `spread_sd`, `ci_halfwidth`; a saved figure path with `fig_saved`; and
`conversions_consistent`. A results paragraph should report number of cases, missing values, a verbal
description of variables, and estimates with their precision, with no interpretation (Dunn p543).

## Reading checklist
Evaluate any paper against the six steps: ask the RQ (POCI: Population, Outcome, Comparison, Intervention),
design, collect, describe/summarise, analyse, report; flag whether the analysis and the conclusion match
(Dunn p530).

## Validity conditions
- Every table/figure must be understandable without the surrounding text and is discussed, not just
  dropped in; tables caption above, figures caption below, units always shown (Dunn p545, p591-592).
- A reported interval is centred on the statistic; if `mean +/- s` does not match `(lo+hi)/2` and
  `(hi-lo)/2`, the author has mixed SD with SE (Dunn p551).
- Conclusions must follow from the shown evidence and answer the stated RQ (Dunn p543).

## Reporting conventions
- Prefer passive voice and past tense; short unambiguous sentences; do not overstate (Dunn p546-547).
- Replace lexically ambiguous words: say mean/median not "average", substantial not "significant",
  study not "experiment", usual not "normal" (Dunn p544).
- Never plagiarise: paraphrase and cite; no quote over ~25 words (Dunn p547-548).
- `tabulate` is not installed, so the GitHub-markdown table builder is reimplemented inline above;
  `pandas.to_html` works directly if HTML output is wanted. Seeded `default_rng(42)`; headless `Agg`
  figures; write only into `tempfile`.
- Source: Dunn, *Scientific Research and Methodology* (Springer, 2024) Ch.36-37 pp.529-552, App. E pp.591-592.

## Out of scope
Forming confidence intervals, running hypothesis tests or regression, the end-to-end analysis pipeline,
and the reproducibility/provenance gate are sibling skills; this skill only structures, words and audits
the report, tables and figures, and reads others' papers critically (Dunn p530, p543).

## Conventions
- ALPHA is not used here; this skill consumes already-computed statistics. Cite the page next to every
  structure rule, threshold and checklist item. Never invent a number - compute and print it, then cite
  the source page for the method (Dunn Ch.36-37 p529-552, checklists p591-592).
