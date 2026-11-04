---
name: pda-data-wrangling-reshape
description: Reshape and combine labelled pandas tables in pure Python - hierarchical (MultiIndex) indexing with partial selection and swaplevel, stack/unstack, merge and join on keys or indexes with inner/left/right/outer joins and cardinality validation, concat along either axis, combine_first to patch overlapping data, and pivot/melt between long and wide format. Use when asked to join, merge, concatenate or align frames, convert long to wide or wide to long, pivot or melt a table, unstack a MultiIndex, or fill missing cells of one dataset from another.
---

# Data Wrangling - Join, Combine, Reshape (Python)

## Purpose
Rearrange and combine labelled pandas objects: a reshaping toolkit (MultiIndex, stack/unstack, pivot/melt)
plus a combination toolkit (merge, join, concat, combine_first). Every count and shape quoted below is
printed by the block that computes it.

## Inputs
- `left`, `right`: `DataFrame`/`Series` to combine, keyed by shared columns or by (multi)level index labels.
- `long_data`: tidy frame, one row per observation, with row-label, column-label and value columns.
- `how` in `inner` (default), `left`, `right`, `outer` (McKinney p274); `axis` in `index`, `columns`.

## Algorithms (all pure numpy/scipy)

### Hierarchical index, then stack / unstack (McKinney p265-268, p289-291)
```python
import numpy as np, pandas as pd
frame = pd.DataFrame(np.arange(12).reshape((4, 3)), index=[["a", "a", "b", "b"], [1, 2, 1, 2]],
                     columns=[["Ohio", "Ohio", "Colorado"], ["Green", "Red", "Green"]])
frame.index.names, frame.columns.names = ["key1", "key2"], ["state", "color"]
wide = pd.DataFrame(np.arange(6).reshape((2, 3)), index=pd.Index(["Ohio", "Colorado"], name="state"),
                    columns=pd.Index(["one", "two", "three"], name="number"))
ragged = pd.concat([pd.Series([0, 1, 2, 3], list("abcd"), dtype="Int64"),
                    pd.Series([4, 5, 6], list("cde"), dtype="Int64")], keys=["one", "two"]).unstack()
long_s = wide.stack()                                       # columns rotate down into the rows
print(f"RESULT nlevels={frame.index.nlevels} ohio={frame['Ohio'].shape} "
      f"swap_head={frame.swaplevel('key1', 'key2').index[:2].tolist()}")
print(f"RESULT stack_rows={len(long_s)} roundtrip={long_s.unstack().equals(wide)} "
      f"stack_keeps_na={len(ragged.stack())} stack_dropped={len(ragged.stack().dropna())}")
```
Prints `nlevels=2 ohio=(4, 2) stack_rows=6 roundtrip=True stack_keeps_na=10 stack_dropped=7` - the 7-row roundtrip is the book's (McKinney p290).

### merge on keys: join type and cardinality (McKinney p272-277)
```python
import pandas as pd
df1 = pd.DataFrame({"key": ["b", "b", "a", "c", "a", "a", "b"], "data1": pd.Series(range(7), dtype="Int64")})
df2 = pd.DataFrame({"key": ["a", "b", "d"], "data2": pd.Series(range(3), dtype="Int64")})
mm1 = pd.DataFrame({"key": ["b", "b", "a", "c", "a", "b"], "data1": pd.Series(range(6), dtype="Int64")})
mm2 = pd.DataFrame({"key": ["a", "b", "a", "b", "d"], "data2": pd.Series(range(5), dtype="Int64")})
left = pd.DataFrame({"key1": ["foo", "foo", "bar"], "key2": ["one", "two", "one"], "lval": [1, 2, 3]})
right = pd.DataFrame({"key1": ["foo", "foo", "bar", "bar"], "key2": ["one", "one", "one", "two"], "rval": [4, 5, 6, 7]})
inner, outer = pd.merge(df1, df2), pd.merge(df1, df2, on="key", how="outer")  # shared name = key; outer = union
many = pd.merge(mm1, mm2, on="key", how="left", validate="many_to_many")      # Cartesian product of keys
print(f"RESULT cols={list(inner.columns)} inner={len(inner)} outer={len(outer)} "
      f"outer_na={int(outer['data1'].isna().sum())} many_left={len(many)} many_b={(many['key'] == 'b').sum()}")
print(f"RESULT two_key_outer={len(pd.merge(left, right, on=['key1', 'key2'], how='outer'))} "
      f"overlap={list(pd.merge(left, right, on='key1', suffixes=('_L', '_R')).columns)}")
```
`inner=6 outer=8` (McKinney p272-273); 3 left `b` rows x 2 right `b` rows give `many_b=6` of `many_left=11` (p275-276).

### Merging on the index, and DataFrame.join (McKinney p277-281)
```python
import pandas as pd
left1 = pd.DataFrame({"key": ["a", "b", "a", "a", "b", "c"], "value": pd.Series(range(6), dtype="Int64")})
right1 = pd.DataFrame({"group_val": [3.5, 7]}, index=["a", "b"])
left2 = pd.DataFrame([[1., 2.], [3., 4.], [5., 6.]], index=list("ace"), columns=["Ohio", "Nevada"]).astype("Int64")
right2 = pd.DataFrame([[7., 8.], [9., 10.], [11., 12.], [13., 14.]], index=list("bcde"),
                      columns=["Missouri", "Alabama"]).astype("Int64")
another = pd.DataFrame([[7., 8.], [9., 10.], [11., 12.], [16., 17.]], index=list("acef"),
                       columns=["New York", "Oregon"])
on_index = pd.merge(left1, right1, left_on="key", right_index=True)  # unique right index keeps left labels
both_idx = pd.merge(left2, right2, how="outer", left_index=True, right_index=True)
print(f"RESULT on_index_rows={len(on_index)} kept_index={on_index.index.tolist()} "
      f"outer_rows={len(pd.merge(left1, right1, left_on='key', right_index=True, how='outer'))} "
      f"group_val={left1.join(right1, on='key')['group_val'].tolist()}")  # join() is LEFT by default
print(f"RESULT index_merge_equals_join={both_idx.equals(left2.join(right2, how='outer'))} "
      f"join_list={left2.join([right2, another], how='outer').shape}")
```
`on_index_rows=5 kept_index=[0, 1, 2, 3, 4]` shows preserved left labels on a many-to-one index join (McKinney
p278); `group_val` ends in `nan` for the unmatched `c` (p280), `join_list=(6, 6)` for a list of frames (p281).

### concat along an axis (McKinney p281-286)
```python
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
s1, s2, s3 = (pd.Series([0, 1], list("ab"), dtype="Int64"), pd.Series([2, 3, 4], list("cde"), dtype="Int64"),
              pd.Series([5, 6], list("fg"), dtype="Int64"))
arr = np.arange(12).reshape((3, 4))
a = pd.DataFrame(rng.standard_normal((3, 4)), columns=list("abcd"))
b = pd.DataFrame(rng.standard_normal((2, 3)), columns=list("bda"))
keyed = pd.concat([s1, s1, s3], keys=["one", "two", "three"])   # keys make the chunks identifiable
print(f"RESULT stacked={len(pd.concat([s1, s2, s3]))} widened={pd.concat([s1, s2, s3], axis='columns').shape} "
      f"inner={pd.concat([s1, pd.concat([s1, s3])], axis='columns', join='inner').shape}")
print(f"RESULT keyed_levels={keyed.index.nlevels} keyed_unstack={keyed.unstack().shape} "
      f"np_concat={np.concatenate([arr, arr], axis=1).shape} glued={pd.concat([a, b], ignore_index=True).shape}")
```
`stacked=7 widened=(7, 3) inner=(2, 2) keyed_levels=2 glued=(5, 4)` (McKinney p282-286): the other axis is unioned unless `join="inner"`.

### combine_first on overlapping data (McKinney p287-288)
```python
import numpy as np, pandas as pd
a = pd.Series([np.nan, 2.5, 0.0, 3.5, 4.5, np.nan], index=list("fedcba"))
b = pd.Series([0.0, np.nan, 2.0, np.nan, np.nan, 5.0], index=list("abcdef"))
df1 = pd.DataFrame({"a": [1., np.nan, 5., np.nan], "b": [np.nan, 2., np.nan, 6.], "c": range(2, 18, 4)})
df2 = pd.DataFrame({"a": [5., 4., np.nan, 3., 7.], "b": [np.nan, 3., 4., 6., 8.]})
positional = np.where(pd.isna(a), b, a)      # label-blind: matches cells, not rows
patched = df1.combine_first(df2)             # caller's non-NA values win, column set is the union
print(f"RESULT where={np.round(positional, 3).tolist()} aligned_sum={float(a.combine_first(b).sum())}")
print(f"RESULT patched={patched.shape} cols={list(patched.columns)} a={patched['a'].tolist()} "
      f"c_na={int(patched['c'].isna().sum())}")
```
`where=[0.0, 2.5, 0.0, 3.5, 4.5, 5.0]`, `aligned_sum=15.5` reproduce the book (McKinney p287); `patched=(5, 3)` with `a=[1.0, 4.0, 5.0, 3.0, 7.0]`, `c_na=1` is the frame example (p288).

### pivot long to wide, melt wide to long (McKinney p291-297)
```python
import numpy as np, pandas as pd
dates = pd.period_range("1959Q1", periods=5, freq="Q-DEC", name="date").to_timestamp("D")
wide = pd.DataFrame({"realgdp": [2710.349, 2778.801, 2775.488, 2785.204, 2847.699],
                     "infl": [0.00, 2.34, 2.74, 0.27, 2.31], "unemp": [5.8, 5.1, 5.3, 5.6, 5.2]}, index=dates)
wide.columns.name = "item"
long_data = wide.stack().reset_index().rename(columns={0: "value"})   # wide -> long, one row per value
back = long_data.pivot(index="date", columns="item", values="value")  # long -> wide
keyed = pd.DataFrame({"key": ["foo", "bar", "baz"], "A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
melted = pd.melt(keyed, id_vars="key")                               # 3 value columns -> 9 rows
print(f"RESULT long_rows={len(long_data)} long_cols={list(long_data.columns)} wide={back.shape} "
      f"gdp_1959Q1={float(back.iloc[0]['realgdp'])} unemp_1959Q1={float(back.iloc[0]['unemp'])}")
print(f"RESULT pivot_equals_set_index_unstack={long_data.set_index(['date', 'item']).unstack(level='item')['value'].equals(back)} "
      f"melt={melted.shape} foo_row={melted.pivot(index='key', columns='variable', values='value').loc['foo'].tolist()}")
import matplotlib; matplotlib.use("Agg"); import pathlib, tempfile     # headless shape check only
from matplotlib import pyplot as plt
fig, ax = plt.subplots(figsize=(2.5, 1.6)); ax.imshow(back.to_numpy(float), aspect="auto"); ax.axis("off")
png = pathlib.Path(tempfile.mkdtemp()) / "pivot-check.png"; fig.savefig(png, dpi=50); plt.close(fig)
print(f"RESULT check_png_written={png.is_file()}")
```
`gdp_1959Q1=2710.349 unemp_1959Q1=5.8` is the book's first pivoted row (McKinney p293-294), `long_rows=15` is
5 dates x 3 items, `pivot` = `set_index` + `unstack` (p295), `melt` its inverse with `melt=(9, 3)` (p296-297).

## Output contract
Return the reshaped or combined object; print one `RESULT name=value` line per quantity.
- `merge`/`join` -> `DataFrame` of key + left + right columns, `NA`/`nan` in unmatched cells, a `_merge` column only with `indicator=True` (McKinney p277).
- `concat` -> `Series` on `axis="index"`, else `DataFrame`; `keys=` stacks the chunks into a `MultiIndex`, `names=` labels it (p284-285).
- `combine_first` -> caller's values plus the other object's labels; it patches, never overwrites (p288).
- `stack` -> the moved axis becomes the innermost level; `unstack`/`pivot` -> wide frame; `melt` -> `variable`/`value` columns (p296-297).

## Validity conditions
- Column-on-column merges discard both row indexes; `reset_index` first when labels carry data (p276).
- `validate="one_to_one"`/`"many_to_one"` turns surprise row multiplication into an error; a many-to-many merge is the Cartesian product (p275-277).
- `unstack` silently creates missing data when a level's values are absent from a subgroup (p290).
- An index join preserves the left index only while the right index is unique (p278).
- `pivot` needs one value per (row, column) pair; repeated pairs are an aggregation problem, not a reshape.
- `np.where` matches by position - only `combine_first` checks that the labels line up (p287).

## Out of scope
Reading files and building frames, NA detection/filtering/deduplication and type coercion, and groupby/agg
semantics (only `level=` selection is used here) are sibling skills, as are resampling, lag and shifting.

## Conventions
- Synthetic examples only, seeded with `numpy.random.default_rng(42)`; `dtype="Int64"` so outer joins show `<NA>` instead of coercing to float.
- Name axes and levels as strings (`axis="columns"`, `level="state"`); keep the outer level sorted with `sort_index(level=0)` before heavy slicing (McKinney p269).
- pandas >= 3: `stack()` keeps the NA combinations the book dropped and `stack(dropna=False)` raises, so get the book's shorter frame with `.stack().dropna()` (McKinney p290); `PeriodIndex(year=, quarter=)` and freq `"Q"` are gone - use `period_range(..., freq="Q-DEC")` and `"QE"` (p292).
- Any figure is headless (`matplotlib.use("Agg")`) and written only inside a `tempfile.mkdtemp()` directory.
- Never invent a value: compute it, print it, and cite the page for the rule. Source: McKinney, *Python for
  Data Analysis*, 3rd ed. (O'Reilly, 2022) ch.8, pp.265-298.
