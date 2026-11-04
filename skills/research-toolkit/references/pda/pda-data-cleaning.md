---
name: pda-data-cleaning
description: Clean a pandas table in pure Python (pandas/numpy/re) - NA policy with dropna/how/thresh and fillna or mean imputation, duplicate detection with duplicated/drop_duplicates subset and keep, sentinel replacement via replace, element-wise map transforms, axis renaming with index.map and rename, outlier detection with abs().gt and capping, seeded sampling and permutation, cut/qcut discretization with bin counts, NA-safe str accessor methods and regex extract, categorical dtype with codes/set_categories and get_dummies. Use when asked to handle missing data, remove duplicate rows, rename column or index labels, turn -999 codes into NA, filter or cap outliers, bin continuous values into groups, trim/split/extract strings, or convert text columns to categorical.
---

# Data Cleaning and Preparation (pandas)

## Purpose
Turn a loaded table into an analysis-ready one: resolve missing values, drop duplicates, replace sentinel
codes, rename labels, filter and cap outliers, bin continuous data, normalise strings, encode
categoricals. Preparation is reported to consume the large majority of an analyst's time (McKinney p221).

## Inputs
- A `pandas.DataFrame` (or `Series`) already loaded, plus the cleaning policy per column.
- NA sentinels are `np.nan` for `float64` and `None` for object data; descriptive statistics exclude NA by
  default, and these methods return new objects rather than mutating the input (McKinney p221-224).
- Outlier threshold: absolute value 3. Random artifacts use `numpy.random.default_rng(42)` (McKinney p235-238).

## Algorithms (all pure numpy/scipy)

### NA policy: filter versus fill (McKinney p223-227)
```python
import numpy as np, pandas as pd
s = pd.Series([1.0, np.nan, 3.5, np.nan, 7.0])
n_missing = int(s.isna().sum())
filled = s.fillna(s.mean())                       # mean-impute the observed values
df = pd.DataFrame([[1.0, 6.5, 3.0], [1.0, np.nan, np.nan],
                   [np.nan, np.nan, np.nan], [np.nan, 6.5, 3.0]])
any_na = len(df.dropna())                         # default drops rows with ANY NA
all_na = len(df.dropna(how="all"))                # drop only fully-NA rows
t2 = len(df.dropna(thresh=2))                     # keep rows with >= 2 non-NA values
ffill = df.ffill().iloc[1, 2]                     # propagate last observed value
print(f"RESULT n_missing={n_missing} fillna_mean={filled.iloc[1]:.4f} dropna_any={any_na} "
      f"dropna_all={all_na} dropna_thresh2={t2} ffill={ffill}")
```
Prints `fillna_mean=3.8333` (mean of the observed 1, 3.5, 7), `dropna_any=1`, `dropna_all=3`; `thresh` tolerates a chosen amount of missingness instead of all-or-nothing (McKinney p224-227).

### Removing duplicates (McKinney p227-229)
```python
import pandas as pd
d = pd.DataFrame({"k1": ["one", "two"] * 3 + ["two"], "k2": [1, 1, 2, 3, 3, 4, 4]})
d2 = d.copy(); d2["v1"] = range(len(d2))
keep_last = d2.drop_duplicates(["k1", "k2"], keep="last")
print(f"RESULT dup_rows={int(d.duplicated().sum())} n_unique_k1={len(d2.drop_duplicates(subset=['k1']))} "
      f"after_drop_dup={len(d.drop_duplicates())} keep_last_v1={int(keep_last['v1'].iloc[-1])}")
```
`duplicated` marks a row whose values equal an earlier row; both methods consider all columns unless a `subset` is given, and `keep` picks the survivor - first by default (McKinney p228-229).

### Sentinel replacement, value mapping, axis renaming (McKinney p229-233)
```python
import numpy as np, pandas as pd
meat = pd.DataFrame({"food": ["bacon", "pulled pork", "bacon", "pastrami", "corned beef",
                              "bacon", "pastrami", "honey ham", "nova lox"],
                     "ounces": [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
m = {"bacon": "pig", "pulled pork": "pig", "pastrami": "cow",
     "corned beef": "cow", "honey ham": "pig", "nova lox": "salmon"}
meat["animal"] = meat["food"].map(m)                       # Series.map takes a dict or callable
pig_oz = float(meat.loc[meat["animal"] == "pig", "ounces"].sum())
sent = pd.Series([1.0, -999.0, 2.0, -999.0, -1000.0, 3.0])
clean = sent.replace([-999, -1000], np.nan)                # Series.replace != Series.str.replace
ax = pd.DataFrame(np.arange(12).reshape((3, 4)), index=["Ohio", "Colorado", "New York"],
                  columns=["one", "two", "three", "four"])
ax.index = ax.index.map(lambda x: x[:4].upper())           # relabel by assigning index.map
ren = ax.rename(columns={"three": "peekaboo"})             # rename returns a new object
print(f"RESULT pig_ounces={pig_oz} sentinels_nulled={int(clean.isna().sum())} "
      f"index0={ax.index[0]} renamed_col={'peekaboo' in list(ren.columns)}")
```
The pig rows (bacon, pulled pork, honey ham) total `32.0` ounces and `replace` nulls all `3` sentinel cells in one call - fillna is just its missing-data special case (McKinney p229-231).

### Outlier detection, capping and random sampling (McKinney p235-238)
```python
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
data = pd.DataFrame(rng.standard_normal((1000, 4)))
n_out = int(data.abs().gt(3).any(axis="columns").sum())     # rows with any |x| > 3
capped = data.mask(data.abs() > 3, np.sign(data) * 3)       # cap outside [-3, 3]
rows = pd.DataFrame(rng.standard_normal((200, 3)))          # table to draw samples from
perm = rows.take(rng.permutation(len(rows)))                # reorder every row once
sub = rows.sample(n=5, random_state=42)                     # subset without replacement
rep = rows.sample(n=500, replace=True, random_state=42)     # repeats allowed
print(f"RESULT outlier_rows={n_out} cap_min={capped.values.min():.1f} cap_max={capped.values.max():.1f} "
      f"perm_rows={len(perm)} sample_rows={len(sub)} sample_replace_rows={len(rep)}")
```
The parentheses around the comparison matter because `any` runs on its result; the `np.sign` form is the book's capping statement and forces `cap_min=-3.0` / `cap_max=3.0` (McKinney p236-237).

### Discretization and binning (McKinney p233-235)
```python
import numpy as np, pandas as pd
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
cats = pd.cut(ages, [18, 25, 35, 60, 100])                  # right-closed bins by default
vc = cats.value_counts().sort_index()
rng = np.random.default_rng(42)
q = pd.qcut(rng.standard_normal(1000), 4).value_counts().sort_index()
print(f"RESULT youth_bin={int(vc.iloc[0])} cut_counts={list(map(int, vc.values))} "
      f"qcut_counts={list(map(int, q.values))} codes={list(map(int, cats.codes))[:4]}")
```
`cut` uses fixed edges, so `(18, 25]` holds `5` people and the bins are `[5, 3, 3, 1]` exactly as printed in the book; `qcut` bins on sample quantiles instead, giving equal-size bins (McKinney p233-235).

### String operations and regex (McKinney p245-253)
```python
import re, numpy as np, pandas as pd
data = pd.Series({"Dave": "dave@google.com", "Steve": "steve@gmail.com",
                  "Rob": "rob@gmail.com", "Wes": np.nan})
has_gmail = data.str.contains("gmail")                      # str.* skips over and propagates NA
pattern = r"([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})"
ext = data.str.extract(pattern, flags=re.IGNORECASE)        # one column per capture group
pieces = [p.strip() for p in "a,b,  guido".split(",")]       # split, then trim whitespace
print(f"RESULT n_gmail={int(has_gmail.sum())} wes_is_na={bool(pd.isna(data['Wes']))} "
      f"domains={list(ext[1].dropna())} rebuilt={'::'.join(pieces)}")
```
`data.map` fails on the NA row, which is why the vectorized `str` accessor exists: it reports `n_gmail=2` and leaves `Wes` NA, while `extract` returns username, domain and suffix as separate columns (McKinney p251-252).

### Categorical dtype, memory and dummy variables (McKinney p239-241, p253-263)
```python
import numpy as np, pandas as pd
N = 10_000_000
labels = pd.Series(["foo", "bar", "baz", "qux"] * (N // 4))
cats = labels.astype("category")                            # dictionary-encoded representation
obj_mem = int(labels.memory_usage(deep=True))
cat_mem = int(cats.memory_usage(deep=True))
df = pd.DataFrame({"key": ["b", "b", "a", "c", "a", "b"], "data1": range(6)})
dm = pd.get_dummies(df["key"], prefix="key")                # k distinct values -> k 0/1 columns
ordered = pd.Categorical.from_codes([0, 1, 2, 0, 0, 1], ["foo", "bar", "baz"], ordered=True)
wider = cats.cat.set_categories(["foo", "bar", "baz", "qux", "zoo"])
print(f"RESULT object_mem={obj_mem} cat_mem={cat_mem} ratio={obj_mem / cat_mem:.1f} "
      f"dummy_cols={dm.shape[1]} key_b_count={int(dm['key_b'].sum())} "
      f"n_levels={len(wider.cat.categories)} ordered={ordered.ordered}")
```
The encoding keeps integer codes plus one copy of each string, so the categorical column of this ten-million-row example is `52.0` times smaller than the object one (the ratio is build-dependent) (McKinney p260-261); `get_dummies` on `3` distinct keys yields `3` columns with `key_b` counting `3`, and `set_categories` declares unobserved levels so `value_counts` reports them.

## Output contract
Return a cleaned `DataFrame`/`Series`; carry these exact diagnostic names forward: `n_missing`, `fillna_mean`, `dropna_any`, `dropna_all`, `dropna_thresh2`, `dup_rows`, `after_drop_dup`, `keep_last_v1`, `sentinels_nulled`, `pig_ounces`, `outlier_rows`, `cap_min`, `cap_max`, `perm_rows`, `sample_rows`, `sample_replace_rows`, `cut_counts`, `qcut_counts`, `n_gmail`, `domains`, `object_mem`, `cat_mem`, `ratio`, `dummy_cols`, `n_levels`. Each block prints them on its `RESULT name=value` line.

## Validity conditions
- Imputation reuses observed values silently - record which statistic filled which column, and analyse the missingness itself first to catch collection problems or bias (McKinney p222).
- Prefer nullable extension dtypes (`Int64`, `string`, `boolean`) so a column does not become `float64` merely because one value is missing; their sentinel is `pandas.NA` (McKinney p242-245).
- Capping at absolute `3` shrinks the tails: mean and std move, the quartiles should not; if they do, the threshold is too tight. `qcut` needs distinct values - heavy ties collapse the quantile edges.

## Out of scope
Loading and IO (`read_csv`/`read_table`), reshaping and pivoting (melt, stack, unstack, crosstab) and `groupby` aggregation are sibling skills; only a cleaned categorical column is handed to them. Splitting one delimited string into dummy columns is a cleaning step, not a join (McKinney p240).

## Conventions
- Seeds: `numpy.random.default_rng(42)` for generated arrays, `random_state=42` for `sample`.
- Absolute-value cutoff of 3 for outliers; bins are right-closed (`pd.cut` default) unless `right=False`.
- Use `df.ffill()` / `df.bfill()` - the `fillna(method=...)` keyword is gone on modern pandas - and name the axis for a column-wise drop (`dropna(axis="columns", how="all")`).
- `pyarrow`, which backs the fast `string` extension dtype, is not installed, so the categorical encoding above carries the memory saving. No figure is produced; if one is asked for, set `matplotlib.use("Agg")` first.
- Never invent a value - compute it in a block and print the `RESULT` line.
- Source: McKinney, *Python for Data Analysis*, 3rd ed. (O'Reilly, 2022) ch.7, pp.221-264.
