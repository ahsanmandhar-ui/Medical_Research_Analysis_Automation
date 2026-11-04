---
name: pda-groupby-aggregation
description: Split-apply-combine in pure pandas/numpy - group by array, column, dictionary/Series, callable or MultiIndex level keys, run optimized or custom/named aggregations, broadcast group statistics back with transform (normalization, ranks, unwrapped z-scores), use apply for per-group subset selection, group-specific missing-value fills, weighted averages and correlations, bucket a column with cut/qcut then aggregate per bucket, and build boardwork with pivot_table (margins, fill_value, aggfunc=len) and crosstab. Use when asked to group or aggregate by one or more keys, compute per-group stats, add a per-group mean or z-score column, take top-n per group, or produce a pivot table or cross-tabulation with totals.
---

# GroupBy: Split-Apply-Combine (Python)

## Purpose
Split a pandas object on one or more keys, apply a function to each piece, combine the pieces - the pattern Wickham named split-apply-combine
(McKinney p338). `pivot_table` and `crosstab` are spreadsheet boardwork built on this same groupby plus hierarchical-index reshaping (p369-370).

## Inputs
- Group key(s), freely mixable since all become arrays internally (McKinney p338-339, p345-346): an array/list as long as the grouped axis; a column name
  or list; a dict/Series mapping axis labels to group names, unused entries harmless; a callable invoked once per index label. Data is a Series or
  DataFrame grouped on its index; a non-numeric key column is silently dropped from numeric aggregations as a nuisance column (p341).
- Optimized method names (Table 10-1, p347-348): `any, all, count, cummin, cummax, cumsum, cumprod, first, last, mean, median, min, max, nth, ohlc, prod,
  quantile, rank, size, sum, std, var`. `SEED = 42`, `N_BINS = 4`, `TOL = 1e-6`; frames are synthesised inline, the tipping-style one adding `tip_pct =
  tip / total_bill` as the source does (p350).

## Algorithms (all pure numpy/scipy)

### Split on keys, then aggregate: size, count, custom and named aggregations
```python
import numpy as np, pandas as pd
df = pd.DataFrame({"key1": ["a", "a", None, "b", "b", "a", None], "key2": pd.Series([1, 2, 1, 2, 1, None, 1], dtype="Int64"),
                   "data1": [-0.204708, 0.478943, -0.519439, -0.555730, 1.965781, 1.393406, 0.092908],
                   "data2": [0.281746, 0.769023, 1.246435, 1.007189, -1.296221, 0.274992, 0.228913]})
m = df["data1"].groupby([df["key1"], df["key2"]]).mean()      # two keys -> hierarchical index
print("RESULT pair_means=", {str(k): round(v, 6) for k, v in m.items()}, "unstack=\n", m.unstack().round(6))
print("RESULT sizes_dropna_false=", df.groupby("key1", dropna=False).size().to_dict(), "nonnull_count_a=", df.groupby("key1").count().loc["a"].tolist())
rng = np.random.default_rng(42)
tips = pd.DataFrame({"day": rng.choice(["Thur", "Fri", "Sat", "Sun"], 240), "smoker": rng.choice(["No", "Yes"], 240, p=[0.64, 0.36]), "total_bill": np.round(rng.gamma(4, 5, 240) + 3, 2)})
tips["tip"] = np.round(tips.total_bill * rng.uniform(0.08, 0.30, 240), 2); tips["tip_pct"] = tips.tip / tips.total_bill
g = tips.groupby(["day", "smoker"]); print("RESULT named_agg=\n", g.agg(n_orders=("tip_pct", "count"), average=("tip_pct", "mean"), spread=("tip_pct", np.std), ptp=("tip_pct", lambda s: s.max() - s.min())).round(4).head(2))
print("RESULT as_index_false_cols=", list(tips.groupby(["day", "smoker"], as_index=False)["tip_pct"].mean().columns), "nuisance_dropped=", "key1" not in df.groupby("key1").count())
```

Pair means, the unstacked 2x2 table, sizes a=3/b=2/NaN=2 and non-null counts [2,3,3] are the printed source values exactly (McKinney p340-342).
`(output_name, func)` triples name result columns in one `agg` call; `as_index=False` (or `reset_index()`) returns keys as columns, skipping index
construction (p350-353). Functions outside Table 10-1 (`ptp`, `nsmallest`) run but are slower as each chunk is sliced separately (p348-349).

### Dictionary, callable and index-level keys
```python
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
people = pd.DataFrame(rng.standard_normal((5, 5)), columns=list("abcde"), index=["Joe", "Steve", "Wanda", "Jill", "Trey"])
people.iloc[2, 1:3] = np.nan                                 # NA cells exactly as the source (p345)
colors = pd.Series({"a": "red", "b": "red", "c": "blue", "d": "blue", "e": "red", "f": "orange"})
print("RESULT nonnull_by_color=\n", people.T.groupby(colors).count().T)
print("RESULT callable_key_len3_is_Joe_row=", people.groupby(len).sum().loc[3].equals(people.loc["Joe"]))
print("RESULT mixed_callable_and_list_min=", people.groupby([len, ["one", "one", "one", "two", "two"]]).min().loc[(5, "one")].round(6).tolist())
cols = pd.MultiIndex.from_arrays([["US", "US", "US", "JP", "JP"], [1, 3, 5, 1, 3]], names=["cty", "tenor"])
print("RESULT count_by_level_cty=", pd.DataFrame(rng.standard_normal((4, 5)), columns=cols).T.groupby(level="cty").count().T.iloc[0].to_dict())
```

A dict/Series key acts as a fixed-size mapping, a callable is applied to each label, and key types can be mixed (p345-346). The
non-null-by-color counts (Joe 2/3, Steve 2/3, Wanda 1/2, Jill 2/3, Trey 2/3) and the `level="cty"` count of 2 JP against 3 US per
row are the printed tables (p346-347).

### transform: broadcast a group statistic to the input shape
```python
import numpy as np, pandas as pd
df = pd.DataFrame({"key": ["a", "b", "c"] * 4, "value": np.arange(12.0)})     # McKinney p366
g = df.groupby("key")["value"]
def normalize(x): return (x - x.mean()) / x.std()
t = g.transform(normalize)
print("RESULT group_means=", g.mean().to_dict(), "z_first4=", [round(v, 6) for v in t.head(4)], "descending_ranks=", g.transform(lambda s: s.rank(ascending=False)).tolist())
print("RESULT apply_gk_false_equals_transform=", t.equals(df.groupby("key", group_keys=False)["value"].apply(normalize)), "unwrapped_z_equals=", np.allclose((df["value"] - g.transform("mean")) / g.transform("std"), t))
```

Means 4.5/5.5/6.5, z-scores -1.161895 / -0.387298 and ranks 4,4,4 / 3,3,3 / 2,2,2 / 1,1,1 are the printed values (p366-368). `transform` accepts only
a function returning a scalar to broadcast or an object of the group's shape and must not mutate its input (p365). Vectorised "unwrapped" arithmetic
on `transform('mean')` and `transform('std')` needs no written function and takes the fast path (p368-369).

### apply: per-group subset selection and group-specific fills
```python
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
tips = pd.DataFrame({"smoker": rng.choice(["No", "Yes"], 60), "day": rng.choice(["Sat", "Sun"], 60), "total_bill": np.round(rng.gamma(4, 5, 60) + 3, 2), "tip_pct": rng.uniform(0.08, 0.30, 60)})
def top(d, k=2, col="tip_pct"): return d.sort_values(col, ascending=False)[:k]   # subset selection (p354)
res = tips.groupby(["smoker", "day"]).apply(top, k=1, col="total_bill")
print("RESULT top1_rows_per_smoker=", res.groupby(level="smoker").size().to_dict(), "index_levels=", res.index.nlevels)
data = pd.Series([0.329939, 0.981994, np.nan, -1.613716, 1.561587, np.nan, 0.359244, np.nan], index=["Ohio", "New York", "Vermont", "Florida", "Oregon", "Nevada", "California", "Idaho"])
region = ["East"] * 4 + ["West"] * 4
print("RESULT group_means=", {k: round(v, 6) for k, v in data.groupby(region).mean().items()}, "filled_by_group_mean=", data.groupby(region, group_keys=False).apply(lambda gp: gp.fillna(gp.mean())).round(6).tolist())
print("RESULT fill_by_group_name=", data.groupby(region, group_keys=False).apply(lambda gp: gp.fillna({"East": 0.5, "West": -1.0}[gp.name])).round(6).tolist())
```

`apply` splits, calls the function on each chunk and concatenates, so it must return a scalar or a pandas object; extra arguments follow the function
and `gp.name` gives the group label (McKinney p353-360). East -0.100594 / West 0.960416, the filled Vermont/Nevada/Idaho cells and the 0.5/-1.0 fills
equal the printed tables (p359-360). The same pattern carries `np.average(gp["data"], weights=gp["weights"])` (p362), `corrwith` (p364) and a per-chunk OLS (p365).

### Bucket and quantile analysis by group
```python
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
frame = pd.DataFrame({"data1": rng.standard_normal(1000), "data2": rng.standard_normal(1000)})
bins, qidx = pd.cut(frame["data1"], 4), pd.qcut(frame["data1"], 4, labels=False)
print("RESULT cut_equal_width_counts=", frame.groupby(bins, observed=True).size().tolist(), "qcut_equal_size_counts=", frame.groupby(qidx, observed=True).size().tolist())
print("RESULT qcut_min_max=\n", frame.groupby(qidx, observed=True)["data1"].agg(["min", "max"]).round(4))
def get_stats(gp): return pd.DataFrame({"min": gp.min(), "max": gp.max(), "count": gp.count(), "mean": gp.mean()})
apply_path = frame.groupby(bins, observed=True).apply(get_stats).xs("data1", level=1)[["min", "max", "mean"]]
agg_path = frame.groupby(bins, observed=True).agg(["min", "max", "count", "mean"])["data1"][["min", "max", "mean"]]
print("RESULT apply_equals_agg_list=", np.allclose(apply_path.to_numpy(), agg_path.to_numpy(), atol=1e-12), "quantile_25_75_by_bucket=", frame.groupby(qidx, observed=True)["data1"].quantile([0.25, 0.75]).round(4).tolist())
```

The `Categorical` returned by `cut` is itself a group key, so bucketing plus groupby is one step, and a function returning a DataFrame generalises the
per-bucket table (McKinney p356-357). Equal-width `cut` buckets take very unequal counts while `qcut` puts exactly 250 of the 1000 rows in each bucket
as in the source's 250-per-bucket table (p358); the `apply` path equals `agg(["min","max","count","mean"])` and `quantile([0.25,0.75])` per bucket.

### pivot_table and crosstab
```python
import numpy as np, pandas as pd
rng = np.random.default_rng(42); n = 240
tips = pd.DataFrame({"day": rng.choice(["Thur", "Fri", "Sat", "Sun"], n), "time": np.where(rng.random(n) < 0.7, "Dinner", "Lunch"), "smoker": rng.choice(["No", "Yes"], n, p=[0.64, 0.36]), "size": rng.integers(1, 7, n), "tip_pct": rng.uniform(0.05, 0.35, n)})
pv = tips.pivot_table(index=["day", "smoker"], values="tip_pct")
print("RESULT pivot_equals_groupby=", np.allclose(pv["tip_pct"], tips.groupby(["day", "smoker"])["tip_pct"].mean().reindex(pv.index)))
print("RESULT aggfunc_len_margins=\n", tips.pivot_table(index="time", columns="day", values="tip_pct", aggfunc=len, margins=True))
pm = tips.pivot_table(index=["time", "day"], columns="smoker", values="tip_pct", margins=True)
print("RESULT margins_All_row=", pm.tail(1).round(6)["All"].tolist(), "overall_mean=", round(float(tips["tip_pct"].mean()), 6), "unweighted_cell_mean=", round(float(pm.iloc[:-1, :-1].to_numpy().mean()), 6))
print("RESULT fill_value_row=", tips.pivot_table(index=["time", "size"], columns="day", values="tip_pct", fill_value=0).iloc[0].round(4).to_dict())
survey = pd.DataFrame({"Nationality": ["USA", "Japan", "USA", "Japan", "Japan", "Japan", "USA", "USA", "Japan", "USA"],
                       "Handedness": ["Right-handed", "Left-handed", "Right-handed", "Right-handed", "Left-handed", "Right-handed", "Right-handed", "Left-handed", "Right-handed", "Right-handed"]})
print("RESULT crosstab=\n", pd.crosstab(survey["Nationality"], survey["Handedness"], margins=True))
```

`pivot_table` defaults to `aggfunc="mean"`, `index` keys down the rows and `columns` keys across, `margins=True` adds `All` subtotals, `fill_value` patches
empty cells (McKinney p369-372, options Table 10-2 p372); the same rectangle comes from `groupby(...).mean()` (p370-371) and `aggfunc=len` counts every row
of a cell where `"count"` drops NA inside it (p371). crosstab is the frequency case - printed survey table Japan 2/3/5, USA 1/4/5, All 3/7/10 (p372-373).

## Output contract
- `groupby()` returns a lazy GroupBy object; nothing is computed until an aggregation runs (p340). One key gives a Series indexed by the key (index name
  taken from it); k keys give a hierarchical index of observed combinations, widenable with `unstack()`; `grouped[["col"]]` stays a DataFrame and
  `grouped["col"]` returns a Series (p340, p344-345).
- `agg` with a list or `(name, func)` triples gives one column per function, hierarchical columns only where a column got more than one function (p353);
  `transform` returns the input's shape and index; `apply` concatenates pieces under a group-key index level unless `group_keys=False`; `pivot_table` and
  `crosstab` add `All` margin rows/columns when `margins=True`. Print one `RESULT name=value` line per quantity, counts as integers, statistics at 6 dp.

## Validity conditions
- Missing values in a group key are excluded by default, so group sizes will not sum to `len(df)`; pass `dropna=False` when the NA group matters
  (McKinney p342). Table 10-1 statistics ignore NA inside a group; `size` counts rows while `count` counts non-NA values (p342, p347-348).
- A nuisance non-numeric column is dropped from `mean` without warning - select columns explicitly (p341). With a Categorical key `observed=True` lists
  only present categories, `observed=False` also emits every empty one (Table 10-2, p372).
- An `All` margin cell is the statistic of all underlying rows, not the average of the cell values, so comparing cells across unbalanced groups misleads
  (p371). `transform` raises if it breaks its shape rule or mutates its input (p365); `apply` requires a scalar or pandas object back.

## Conventions
- Build frames from `numpy.random.default_rng(42)` or the source's printed literals; never hardcode a group statistic - compute it and print the
  `RESULT` line. `TOL = 1e-6`, statistics at 6 dp; prefer Table 10-1 aliases over custom functions (p349).
- `groupby(axis="columns")` no longer exists in the installed pandas: transpose, group on the index, transpose back (`people.T.groupby(colors).count().T`);
  the reproduced tables are unchanged. Pass `group_keys=False` when an `apply` result should keep the original index, else that index is one level deeper
  than the book's output and grouping columns are excluded from the frame handed to the function.
- Use `as_index=False` for report tables and `observed=True` for `cut` keys; `mean`/`std` aliases are sample statistics while `np.std` is population sd -
  say which in the column name. No figure here; render headless with `matplotlib.use("Agg")`. `statsmodels` is only for the per-chunk OLS and optional.

## Out of scope
Reshaping for its own sake (`stack/unstack/pivot/melt/merge/concat`) belongs to the reshape-and-join skill - this file only unstacks a groupby result; NA
policy, duplicates, string tidying and outlier treatment belong to the cleaning skill - only a fill value *varying by group* is treated here, as a
groupby-plus-apply pattern (McKinney p359-360). Time aggregation (`resample`) is chapter 11 (p338); fitting or interpreting a regression is the modelling skill.
