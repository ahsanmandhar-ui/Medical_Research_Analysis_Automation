---
name: maa-validation-vs-r-stata
description: Cross-validate a Python meta-analysis result table against R (meta/metafor) or Stata reference numbers in pure pandas/numpy - write_reference(path, py_table) scaffolds the reference CSV before R output exists, crosscheck(py_table, reference_csv_path, atol, rtol, cols) compares cell by cell with numpy.allclose semantics, printing PASS/DIFF per estimate and returning a diff_keys list. Carries the tolerance tiers and worked gaps of the 17-trial haloperidol-versus-placebo Python/STATA/R comparison table, the display-rounding versus algorithmic-agreement split, and the log-scale and reciprocal traps that make an OR column fail against a log-OR column. Use when asked to verify Python meta-analytic numbers against R or Stata, choose how tight a cross-tool tolerance should be, explain a small discrepancy between packages, or guard a pooled-estimate pipeline against a scale or sign error. NOT for computing the estimates - pooling, heterogeneity, effect sizes and bias diagnostics are separate skills.
---

# Cross-Validating Python Meta-Analysis Output Against R and Stata (Python)

## Purpose
Compare a Python meta-analytic result table with a caller-supplied CSV of R (`meta`/`metafor`) or Stata numbers and decide
cell by cell whether they agree within a stated tolerance - the audit the source runs on its own Python/STATA/R table (Masoumi & Shahraz 2022 p5-7). Nothing is re-pooled here; only compared.

## Inputs
- `py_table`: pandas DataFrame, one row per study (optionally plus `Total` rows); id column `study`; value columns `rr`, `ci_lo`, `ci_hi`, `w_fe`, `w_re` (ratios, percent weights) or `est`, `lo`, `hi` on the log scale.
- `reference_csv_path`: **caller-supplied** path to the CSV holding the R/Stata numbers. No data file ships with this skill; each block below writes its own reference CSV inside a `tempfile` directory it creates.
- `atol`, `rtol`: numpy semantics, agreement iff `|py - ref| <= atol + rtol*|ref|`; defaults `ATOL = 0.005`, `RTOL = 0.001`.
- `cols`: subset of shared numeric columns; `None` = every shared column except `study`.

## Algorithms (all pure numpy/scipy)

### `write_reference` + `crosscheck` on the article's own Table 2 rows
```python
import numpy as np, pandas as pd, os, tempfile
ATOL, RTOL = 0.005, 0.001   # algorithmic tier: half a printed 2-dp digit plus 0.1% of the reference
COLS = ["rr", "ci_lo", "ci_hi", "w_fe", "w_re"]
# Python column of Table 2 (Masoumi & Shahraz 2022 p7): RR, CI low/high, fixed-/random-effects weight %.
ROWS = [("Arvanitis1997",1.42,0.89,2.25,18.86,14.66),("Beasley1996",1.05,0.73,1.50,31.22,16.46),
 ("Bechelli1983",6.21,1.52,25.35,2.05,4.48),("Borison1992",7.00,0.40,122.45,0.49,1.30),
 ("Chouinard1993",3.49,1.11,10.95,3.10,6.10),("Durost1964",8.68,1.26,59.95,1.09,2.65),
 ("Garry1962",1.75,0.58,5.24,3.37,6.46),("Howard1974",2.04,0.67,6.21,3.27,6.33),
 ("Marder1994",1.36,0.75,2.47,11.37,12.40),("Nishikawa1982",3.00,0.14,65.91,0.42,1.13),
 ("Nishikawa1984",9.20,0.58,145.77,0.53,1.39),("Reschke1974",3.79,1.06,13.60,2.48,5.20),
 ("Selman1976",1.48,0.94,2.35,19.11,14.71),("Serafetinides1972",8.40,0.50,142.28,0.51,1.33),
 ("Simpson1967",2.35,0.13,43.53,0.48,1.26),("Spencer1992",11.00,1.67,72.40,1.14,2.77),
 ("Vichaiya1971",19.00,1.16,311.97,0.52,1.36)]
py_table = pd.DataFrame(ROWS, columns=["study"] + COLS)
def write_reference(path, py_table):
    """--generate-reference equivalent: scaffold the CSV a caller later pastes R/Stata numbers into."""
    py_table.to_csv(path, index=False); return path

def crosscheck(py_table, reference_csv_path, atol=ATOL, rtol=RTOL, cols=None, verbose=True):
    """Cell-by-cell PASS/DIFF of a Python table against a caller-supplied R/Stata CSV, numpy tolerances."""
    ref = pd.read_csv(reference_csv_path).set_index("study"); py = py_table.set_index("study")
    cols = cols if cols else [c for c in py_table.columns if c in ref.columns and c != "study"]
    hits = []
    for st in py.index:
        for c in cols:
            a, b = float(py.at[st, c]), float(ref.at[st, c])
            ok = bool(np.allclose([a], [b], atol=atol, rtol=rtol))
            hits.append((f"{st}.{c}", abs(a - b), ok))
            if verbose: print(f"{'PASS' if ok else 'DIFF'} {st}.{c} py={a:g} ref={b:g} absdiff={abs(a - b):g}")
    bad = [h[0] for h in hits if not h[2]]
    print(f"RESULT compared={len(hits)} pass={len(hits) - len(bad)} diff={len(bad)} atol={atol} rtol={rtol}")
    return {"compared": len(hits), "n_pass": len(hits) - len(bad), "n_diff": len(bad),
            "diff_keys": bad, "max_absdiff": max([h[1] for h in hits], default=0.0)}
# The only cells where the printed R / STATA columns differ from Python's (Table 2, p7); every other cell is
# identical in the source. Marder1994.rr is then perturbed on purpose - a planted bug the checker must catch.
R_DIFF = {"rr": {"Nishikawa1984": 9.00, "Serafetinides1972": 8.38, "Simpson1967": 2.27},
          "ci_lo": {"Nishikawa1984": 0.57, "Simpson1967": 0.12},
          "ci_hi": {"Borison1992": 121.94, "Nishikawa1982": 65.55, "Nishikawa1984": 142.29,
                    "Serafetinides1972": 141.44, "Simpson1967": 41.77, "Vichaiya1971": 311.71},
          "w_fe": {"Borison1992": 0.50, "Nishikawa1982": 0.43},
          "w_re": {"Arvanitis1997": 14.67, "Beasley1996": 16.48, "Chouinard1993": 6.09, "Garry1962": 6.45,
                   "Howard1974": 6.32, "Marder1994": 12.41, "Reschke1974": 5.19, "Selman1976": 14.72}}
S_DIFF = {"rr": {"Chouinard1993": 3.50}, "ci_lo": {"Garry1962": 0.59},
          "ci_hi": {"Borison1992": 122.44, "Chouinard1993": 10.96, "Nishikawa1982": 65.90,
                    "Nishikawa1984": 145.76, "Vichaiya1971": 311.96}}
def filled(cells):
    t = py_table.set_index("study")
    for col, d in cells.items(): t.loc[list(d), col] = list(d.values())
    return t.reset_index()
with tempfile.TemporaryDirectory() as td:
    rpath = os.path.join(td, "r_reference.csv"); spath = os.path.join(td, "stata_reference.csv")
    write_reference(rpath, py_table)
    print(f"RESULT scaffold_rows={len(pd.read_csv(rpath))} w_sum_fe={py_table['w_fe'].sum():.2f} w_sum_re={py_table['w_re'].sum():.2f}")
    assert abs(py_table["w_fe"].sum() - 100) < 0.05 and abs(py_table["w_re"].sum() - 100) < 0.05
    r_ref = filled(R_DIFF); r_ref.loc[r_ref["study"] == "Marder1994", "rr"] = 1.58
    r_ref.to_csv(rpath, index=False); filled(S_DIFF).to_csv(spath, index=False)
    res_r = crosscheck(py_table, rpath)
    print(f"RESULT r_diff_keys={'|'.join(res_r['diff_keys'])}")
    assert res_r["n_diff"] == 13 and res_r["n_pass"] == 72, res_r["diff_keys"]      # DIFF outcome
    assert {"Nishikawa1984.rr", "Serafetinides1972.rr", "Simpson1967.rr", "Marder1994.rr"} <= set(res_r["diff_keys"])
    assert not {"Arvanitis1997.rr", "Spencer1992.w_re", "Vichaiya1971.ci_hi"} & set(res_r["diff_keys"])   # PASS side
    strict = crosscheck(py_table, spath, cols=["rr"], verbose=False)
    display = crosscheck(py_table, spath, cols=["rr"], rtol=0.003, verbose=False)
    ci_disp = crosscheck(py_table, spath, cols=["rr", "ci_lo", "ci_hi"], atol=0.011, verbose=False)
    assert strict["diff_keys"] == ["Chouinard1993.rr"] and display["n_diff"] == 0 and ci_disp["n_diff"] == 0
    print("RESULT assert_table2 OK")
```
85 comparisons, 72 PASS, 13 DIFF. The flagged risk ratios are exactly the three the article reports as discrepant
against R - "three of seventeen rows" (Masoumi & Shahraz 2022 p5-6) - plus the planted Marder1994 row, so a real bug
is caught. `Vichaiya1971.ci_hi` differs by 0.26 yet passes, as do R's eight sub-0.02 `w_re` cells, because rtol
scales with magnitude; `Borison1992.w_fe` (0.49 vs 0.50) does not. Against STATA the strict tier flags only
`Chouinard1993.rr` (Stata prints `3.5` for `3.49`) and the display tier clears it, matching the article's claim of
agreement with STATA at the second decimal (p6); weight sums compute to 100.01 / 99.99.

### Log-scale and reciprocal traps on the summary rows
```python
import numpy as np, pandas as pd
ATOL, RTOL = 0.005, 0.001
# Table 2 "Total" block, Python vs R (Masoumi & Shahraz 2022 p7).
sum_tbl = pd.DataFrame([("rr_random", 2.09, 2.08), ("ci_lo_random", 1.49, 1.49), ("ci_hi_random", 2.92, 2.92),
                        ("rr_fixed", 1.57, 1.57), ("tau2", 0.146, 0.146), ("i2", 41.37, 41.27),
                        ("Q", 27.29, 27.24)], columns=["stat", "py", "ref"])
for tol, tier in [(0.005, "algorithmic"), (0.02, "last_digit"), (0.10, "summary")]:
    ok = np.isclose(sum_tbl["py"], sum_tbl["ref"], atol=tol, rtol=RTOL)
    print(f"RESULT tier={tier} atol={tol} pass={int(ok.sum())}/{len(sum_tbl)} fails="
          f"{'-'.join(sum_tbl.loc[~ok, 'stat'].tolist()) or 'none'}")
assert np.isclose(sum_tbl["py"], sum_tbl["ref"], atol=0.10, rtol=RTOL).all()
# Trap 1 - scale: R (metafor) returns the pooled LOG-RR while Python and Stata print the RATIO.
py_rr, r_log = 2.09, float(np.log(2.08))
print(f"RESULT log_vs_ratio_absdiff={abs(py_rr - r_log):.3f} pass={bool(np.allclose(py_rr, r_log, atol=ATOL, rtol=RTOL))}")
print(f"RESULT same_pair_on_log_scale_gap={abs(float(np.log(py_rr)) - r_log):.4f}")
assert not np.allclose(py_rr, r_log, atol=ATOL, rtol=RTOL)              # different scales, not different methods
assert np.allclose(float(np.log(py_rr)), r_log, atol=ATOL, rtol=RTOL)   # ...and they agree once put on one scale
# Trap 2 - direction: a flipped comparator turns 1.49 into its reciprocal, a bare sign change on the log scale.
lo_py, lo_flip = 1.49, 1.0 / 1.49
print(f"RESULT reciprocal={lo_flip:.4f} log_signed_pass={bool(np.allclose(np.log(lo_py), np.log(lo_flip), atol=ATOL, rtol=RTOL))}"
      f" log_ABS_pass={bool(np.allclose(abs(np.log(lo_py)), abs(np.log(lo_flip)), atol=ATOL, rtol=RTOL))}")
assert not np.allclose(np.log(lo_py), np.log(lo_flip), atol=ATOL, rtol=RTOL)        # signs must be kept
assert np.allclose(abs(np.log(lo_py)), abs(np.log(lo_flip)), atol=ATOL, rtol=RTOL)  # a |log| check would PASS
print("RESULT scale_and_sign_traps OK")
```
One printed gap, three verdicts: at `0.005` the summary rows fail on `rr_random`, `i2` and `Q`; at `0.02` only `i2` and `Q`; at
the summary tier `0.10` all seven agree - the source's "same level of absolute agreement" for fixed, random and summary rows.

## Output contract
`write_reference(path, py_table)` writes a header-plus-values CSV at `path` - the `--generate-reference` step, so the
workflow runs before any R output exists - and returns it. `crosscheck(py_table, reference_csv_path, atol, rtol, cols,
verbose)` prints one line per compared cell, `PASS` or `DIFF <study>.<column> py=<a> ref=<b> absdiff=<d>`, then
`RESULT compared=n pass=n diff=n atol=.. rtol=..`, and returns `{"compared", "n_pass", "n_diff", "diff_keys",
"max_absdiff"}`; `diff_keys` holds `"<study>.<column>"` strings in table order, empty iff everything agrees. Workflow:
`write_reference` -> caller pastes the R/Stata column in -> `crosscheck` -> assert `n_diff == 0`.

## Validity conditions
- Both tables must share scale, direction and effect measure; a log-OR against an OR, or `RR` against `1/RR`, fails
  as if the analysis were wrong. Convert before calling `crosscheck`, which never guesses a scale (p2, p7).
- Every `study` key of `py_table` must exist in the reference CSV (the loop indexes the Python rows, so a renamed
  study raises instead of silently skipping), and 2-dp references cap the claim at `|diff| <= 0.005`; a gap of exactly
  one last digit is display rounding, not a method disagreement (Masoumi & Shahraz 2022 p6-7).
- Point estimates, CI bounds, weights and summary statistics need separate tiers; one global tolerance either drowns
  in weight noise or hides a real pooled-effect gap (p6-7).
- The source's Fig-1 narrative (`Q=35.18`, `I2=54.51%`, prediction interval 0.73-7.17) is not rebuildable from its own
  Table 2 rows, where the same formulas give about [0.854, 5.09]; assert the computed value and report the gap (p3, p7).

## Out of scope
Computing anything: per-study effect sizes and weights, fixed/random pooling (MH, Peto, IV, DL), `Q`/`I2`/`tau2` and prediction
intervals, subgroup and meta-regression tests, Egger and funnel diagnostics, forest-plot drawing - each is a sibling skill.
Parsing an R or Stata export (CSV/Excel/JSON) belongs to the pandas I/O skill.

## Conventions
- Default `ATOL = 0.005`, `RTOL = 0.001` (algorithmic tier, ratio-scale columns); the display tier `atol = 0.011` / `rtol = 0.003` reproduces
  "compatible with STATA 100% of the time at the second decimal" (Masoumi & Shahraz 2022 p6); weights `atol = 0.02`, summary stats `atol = 0.10`.
- Tolerances follow numpy exactly: `|py - ref| <= atol + rtol*|ref|`, so the reference value owns the relative part.
- Log-scale columns are exponentiated to the ratio scale, or the ratio pair is logged, before comparing; signs stay.
- No R, no Stata, no external process and no install at build time; `pythonmeta`/`zepid` are absent, so their printed numbers are only literal reference values.
- Reference CSVs are written only inside `tempfile.TemporaryDirectory()`; every check prints `RESULT name=value`.
- Source: Masoumi & Shahraz, *Meta-analysis using Python: a hands-on tutorial*, BMC Med Res Methodol 2022;22:193, comparison text p5-6, Table 1 p6, Table 2 p7.
