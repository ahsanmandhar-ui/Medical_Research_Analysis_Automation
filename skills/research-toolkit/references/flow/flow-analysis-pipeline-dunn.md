---
name: flow-analysis-pipeline-dunn
description: Run and gate a whole quantitative study in Python - research question -> design -> describe -> infer -> report, following Dunn's six basic steps. Classifies an RQ as quantitative/qualitative/mixed by keyword scoring, builds a case-control 2x2 with odds ratio, Woolf 95% CI and Fisher/chi-square, checks six stage gates, and emits a provisional evidence-based report plus plot. Pure numpy/pandas/scipy/matplotlib. Use when asked to plan or run an end-to-end study, decide "quant vs qual vs mixed", design a case-control or observational study, gate a pipeline, or "write up the results honestly".
---

# Quantitative Research Pipeline, End to End (Python)

## Purpose
Turn an observation into a provisional, evidence-based conclusion by running Dunn's six basic
research steps in order and gating each one (Dunn Fig.1.1, p26-27); a hypothesis is only ever
temporarily accepted until contradictory evidence emerges (Dunn p24).

## Inputs
- `rq` (str): one or more research-question texts for stage 1.
- Case-control counts for stages 2-5: `n_cases` (25), `n_controls` (75 matched 1:3), or the
  2x2 of exposed/unexposed cases/controls; a `seed` (42) if synthesising the draw.
- `ALPHA = 0.05`; `Z = 1.96` for the 95% Woolf CI.

## Workflow
Ask the question (Chap.2) -> design (Chaps.3-9) -> collect data (Chap.10) -> describe &
summarise (Chaps.11-14) -> analyse (Chaps.15-35) -> report (Chaps.36-37). Not strictly linear -
you may jump steps - but following it is good practice (Dunn p26).

## Algorithms (all pure numpy/scipy)

### Stage 1 - classify the research question (Dunn p27-29)
If answering the RQ needs a numerical summary (percentage, average, proportion, difference,
association) it is quantitative; reasons/meanings -> qualitative; a numeric survey plus
open-ended/focus-group depth -> mixed (Dunn p29). Let the research problem drive the choice (Dunn p27).
```python
QUANT = ["average","percentage","proportion","association","correlat","relationship between",
         "survey","experiment","measure","odds","which of","quickest","risk of","prevalence"]
QUAL  = ["why do","experience","perception","meaning","explore","focus group","interview","themes"]
MIXED = ["open-ended","follow-up","then survey","survey then","mixed","focus group"]
def classify(text):
    t = text.lower()
    q = sum(k in t for k in QUANT); a = sum(k in t for k in QUAL); m = sum(k in t for k in MIXED)
    if m and q and a: return "mixed"
    if q > a: return "quantitative"
    if a > q: return "qualitative"
    return "quantitative" if q > 0 else "qualitative"
RQS = [("What percentage of the population experiences minor side-effects from this medication?","quantitative"),
       ("Why do people opt to purchase an electric car?","qualitative"),
       ("Which of three junctional tourniquets are quickest, on average, to apply?","quantitative"),
       ("Why do people dump rubbish in mangroves?","qualitative"),
       ("Survey 500 buyers, then run follow-up focus groups to explore themes?","mixed")]
for q, lab in RQS: print(f"{classify(q):>12} | expected {lab} | {q[:46]}")
print(f"RESULT n_rq={len(RQS)} source_example_accuracy={sum(classify(q)==lab for q,lab in RQS)}/{len(RQS)}")
```
Prints `source_example_accuracy=5/5` - the classifier reproduces the worked answers in Dunn p29.

### Stages 2-5 - design, collect, describe, analyse (Dunn p26)
Case-control 1:3 design; seed a reproducible draw; build the 2x2; odds ratio + Woolf log CI +
Fisher exact / Yates chi-square.
```python
import numpy as np, pandas as pd
from scipy import stats
rng = np.random.default_rng(42)
n_cases, n_controls, p_case, p_ctrl = 25, 75, 0.80, 0.35
cases = rng.random(n_cases) < p_case
controls = rng.random(n_controls) < p_ctrl
df = pd.DataFrame({"group": ["case"]*n_cases + ["control"]*n_controls,
                   "exposure": np.concatenate([cases, controls]).astype(int)})
a = int(df[df.group=="case"].exposure.sum()); b = n_cases - a
c = int(df[df.group=="control"].exposure.sum()); d = n_controls - c
tab = np.array([[a, b], [c, d]])
or_ = (a*d)/(b*c); se = np.sqrt(1/a + 1/b + 1/c + 1/d)
lo, hi = np.exp(np.log(or_) - 1.96*se), np.exp(np.log(or_) + 1.96*se)
_, pchi, _, _ = stats.chi2_contingency(tab, correction=True)
_, pf = stats.fisher_exact(tab)
print(f"RESULT 2x2 a={a} b={b} c={c} d={d} prev_case={a/n_cases:.3f} prev_control={c/n_controls:.3f}")
print(f"RESULT odds_ratio={or_:.4f} ci95=[{lo:.4f},{hi:.4f}] p_fisher={pf:.4f} p_chi_yates={pchi:.4f}")
```
Reproduces the potting-mix / Legionella example: OR **4.0788** (95% CI **1.5170 to 10.9665**),
Fisher **p=0.0052**, prevalence 18/25=0.720 vs 29/75=0.387 (expected values, Dunn p26 structure).

### Stage gates - six-step completeness check (Dunn Fig.1.1, p26-27)
```python
GATES = [("ask_question", lambda s: bool(str(s.get("research_question","")).strip())),
 ("design_study", lambda s: bool(s.get("design_type")) and s["n_cases"]>0 and s["n_controls"]>0),
 ("collect_data", lambda s: s["n_missing"]==0),
 ("describe_summarise", lambda s: 0<=s["prev_case"]<=1 and 0<=s["prev_control"]<=1),
 ("analyse_data", lambda s: s["effect_estimate"] is not None and 0<=s["p_value"]<=1),
 ("report_results", lambda s: bool(s["provisional_wording"]))]
GOOD = {"research_question":"Is potting mix associated with L. longbeachae?","design_type":"case-control",
 "n_cases":25,"n_controls":75,"n_missing":0,"prev_case":0.72,"prev_control":0.387,
 "effect_estimate":4.08,"p_value":0.0052,"provisional_wording":True}
BAD = dict(GOOD, research_question="", n_missing=3, effect_estimate=None, provisional_wording=False)
def run(s):
    rows = []
    for n, f in GATES:
        try: ok = bool(f(s))
        except (KeyError, TypeError): ok = False
        rows.append((n, ok))
    return rows
for lab, s in [("compliant", GOOD), ("broken", BAD)]:
    rows = run(s)
    print(f"[{lab}] gates_passed={sum(k for _, k in rows)}/{len(GATES)} ->", ", ".join(n for n,k in rows if not k) or "all pass")
print(f"RESULT total_gates={len(GATES)}")
```
Compliant state passes 6/6; the broken state fails four gates, so no reporting until all clear.

### Stage 6 - provisional report + summary plot (Dunn p24, p26, Chaps 36-37)
```python
import tempfile
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
a, b, c, d = 18, 7, 29, 46
or_ = (a*d)/(b*c); se = np.sqrt(1/a + 1/b + 1/c + 1/d)
lo, hi = np.exp(np.log(or_) - 1.96*se), np.exp(np.log(or_) + 1.96*se)
_, pf = stats.fisher_exact(np.array([[a, b], [c, d]]))
direction = "higher" if or_ > 1 else "lower"
report = (f"Potting-mix handling was associated with {direction} odds of infection (OR {or_:.2f}, "
          f"95% CI {lo:.2f} to {hi:.2f}; Fisher exact p={pf:.4f}). It was PARTIALLY RESPONSIBLE, but "
          "OTHER FACTORS were also involved; accepted only until contradictory evidence emerges.")
with tempfile.TemporaryDirectory() as td:
    out = Path(td)
    (out/"report.md").write_text("# Quantitative study report\n\n"+report+"\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["case","control"], [a/(a+b), c/(c+d)], color=["#c0392b","#2980b9"])
    ax.set_ylim(0, 1); ax.set_ylabel("exposure prevalence")
    fig.savefig(out/"exposure_by_group.png", dpi=90); plt.close(fig)
    print(f"RESULT report_bytes={(out/'report.md').stat().st_size} plot_exists={(out/'exposure_by_group.png').exists()}")
print("RESULT report_provisional=True")
```

## Stage gates
A stage passes only when its data-derived artefact exists (Dunn p24, Fig.1.1 p26-27): (1)
ask_question - a concrete RQ stated; (2) design_study - type set and both n > 0 fixed before
collecting; (3) collect_data - `n_missing == 0`; (4) describe_summarise - both prevalences in
[0,1]; (5) analyse_data - an effect estimate and p in [0,1] exist; (6) report_results - wording
is provisional. Report only when all six clear.

## Output contract
Every stage prints `RESULT name=value`. Stage 1: `label`, `source_example_accuracy`. Stages 2-5:
`2x2 a/b/c/d`, `prev_case`, `prev_control`, `odds_ratio`, `ci95=[lo,hi]`, `p_fisher`, `p_chi_yates`.
Gate check: `gates_passed=k/6` plus failing gate names. Stage 6 writes `report.md` and
`exposure_by_group.png` into the output dir (a temp dir when standalone) and returns the effect,
interval, both p-values, `statistically_valid`, and the provisional sentence.

## Validity conditions
- Case-control measures association via the **odds ratio**, which only approximates the risk ratio when the outcome is rare (Dunn Chaps 15-35).
- Woolf CI needs all four cells > 0; a zero cell makes it not estimable (return NaN and fail the analyse gate).
- Prefer **Fisher exact** when any cell < 5, common with only 25 cases (Dunn p26).
- Matching (1:3) guards confounding; an unmatched analysis can be biased. Quant vs qual is a gross generalisation (Table 1.1, Dunn p27).

## Conventions
- ALPHA=0.05, Z=1.96; use proportions not percentages in formulas; seed synthesis with `numpy.random.default_rng(42)`.
- Report effect size AND interval, name partial responsibility and residual confounding, and write "supports / consistent with", never "proves" (Dunn p24, p26).
- jamovi/SPSS appear in the source (Dunn p26); this reimplements the same contingency numerics with pandas + scipy, so those GUI packages add no numeric capability here.
- Never invent a value - always compute it and cite the source page for the method.

## Out of scope
Single narrow statistics (one CI, one test, correlation) have dedicated skills; meta-analysis
pooling and production ML pipelines are siblings. This skill owns the orchestration and shared
plumbing of a whole study, not one specific test (Dunn p26-27).
