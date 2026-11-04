---
name: pda-pandas-foundations
description: Build and inspect the core pandas objects (Series, DataFrame, Index) in pure Python - creation from arrays/lists/dicts, index and name attributes, head/tail, column and row retrieval, label-vs-position selection with []/loc/iloc, reindex/drop/dropna, isna/notna and boolean/isin filtering, index-label data alignment and arithmetic (union alignment, add(fill_value), DataFrame-Series broadcasting). Use when asked what a Series/DataFrame/Index is, how to make one, to reindex or drop labels, select rows/columns with loc/iloc, filter with a mask, or add/subtract two differently-indexed pandas objects.
---

# pandas Foundations - Series, DataFrame, Index (Python)

## Purpose
Create, inspect, select from and combine the three workhorse pandas objects - Series, DataFrame, Index -
so a table can be built and interrogated before any cleaning, joining, grouping or IO (McKinney p141-192).

## Inputs
- An array/list for a Series, or a dict of equal-length sequences / nested dict for a DataFrame (McKinney p147).
- `labels` for `reindex`/`loc`, `columns=`/`index=` for construction, a Boolean `mask` for filtering, a second
  pandas object for alignment/arithmetic. `pd` = `pandas`, `np` = `numpy` are the only imports (McKinney p142).

## Algorithms (all pure numpy/scipy)

### Series objects: index and name attributes
```python
import pandas as pd
import numpy as np
obj = pd.Series([4, 7, -5, 3])                       # no index given -> RangeIndex 0..N-1 (McKinney p142)
obj.index = ["Bob", "Steve", "Jeff", "Ryan"]         # index can be reassigned in place (McKinney p146)
obj.name = "values"; obj.index.name = "person"       # both the object and index carry a name (McKinney p146)
print(f"RESULT dtype={obj.dtype} idx_type={type(pd.Series([4,7]).index).__name__} name={obj.name}")
```
A `RangeIndex` is the default; NumPy-style ops (`obj2[obj2>0]`, `obj2*2`) preserve the index-value link (McKinney p143-144).

### DataFrame: columns order and a missing column
```python
import pandas as pd
import numpy as np
data = {"state": ["Ohio","Ohio","Ohio","Nevada","Nevada","Nevada"],
        "year": [2000,2001,2002,2001,2002,2003],
        "pop": [1.5,1.7,3.6,2.4,2.9,3.2]}
frame = pd.DataFrame(data)
f2 = pd.DataFrame(data, columns=["year","state","pop","debt"])   # debt absent -> all-NaN column (McKinney p149)
print(f"RESULT shape={frame.shape} cols={list(f2.columns)} debt_isna={int(f2['debt'].isna().sum())}")
```
Columns follow dict insertion order; a name not in the dict appears filled with missing values (McKinney p148-149).

### head / tail
```python
import pandas as pd
data = {"pop": [1.5,1.7,3.6,2.4,2.9,3.2]}
frame = pd.DataFrame(data)
print(f"RESULT head={list(frame.head(2)['pop'])} tail={list(frame.tail(2)['pop'])} nrows={len(frame)}")
```
`head` takes the first rows (5 by default), `tail` the last - expected `[1.5,1.7]` and `[2.9,3.2]` (McKinney p148).

### Index objects: immutable, set-like, duplicates allowed
```python
import pandas as pd
import numpy as np
obj = pd.Series(np.arange(5), index=["a","a","b","b","c"])       # duplicate labels are legal (McKinney p155)
labels = pd.Index(["foo","bar","bar","baz"])                     # Index is immutable (McKinney p154)
print(f"RESULT is_unique={obj.index.is_unique} dup_a={len(obj['a'])} mono={labels.is_monotonic_increasing} uniq={list(pd.Index(['foo','foo','bar']).unique())}")
```
`Index` is immutable (assignment raises) and set-like; a repeated label selects all occurrences, returning a Series not a scalar (McKinney p154-155, p182).

### Reindexing and dropping labels
```python
import pandas as pd
import numpy as np
obj = pd.Series([4.5,7.2,-5.3,3.6], index=["d","b","a","c"])
obj2 = obj.reindex(["a","b","c","d","e"])                         # absent 'e' -> NaN (McKinney p156)
ff = pd.Series(["blue","purple","yellow"], index=[0,2,4]).reindex(np.arange(6), method="ffill")
data = pd.DataFrame(np.arange(16).reshape((4,4)), index=["Ohio","Colorado","Utah","New York"],
                    columns=["one","two","three","four"])
print(f"RESULT e_nan={bool(pd.isna(obj2['e']))} ffill1={ff[1]} dropped={list(data.drop(index=['Colorado','Ohio']).index)}")
```
`reindex` rearranges to a new index injecting NaN for absent labels; `method="ffill"` forward-fills ordered data; `drop` removes labels from an axis (McKinney p156-160).

### Selection: [] vs loc vs iloc (label vs position)
```python
import pandas as pd
import numpy as np
obj = pd.Series(np.arange(4.), index=["a","b","c","d"])
obj1 = pd.Series([1,2,3], index=[2,0,1])                          # integer index -> [] is label-based (McKinney p161)
print(f"RESULT loc={list(obj.loc[['b','a','d']])} iloc_slice={list(obj.iloc[2:4])} label2={int(obj1[2])} pos0={int(obj1.iloc[0])}")
```
Prefer `loc` (labels) and `iloc` (positions) to dodge the integer-index ambiguity: `[]` on an integer index is label-based
(`obj1[2]`=label-2 value 1), and `loc[[0,1,2]]` errors when the index is not integer (McKinney p160-162, p147).

### Boolean filtering and isin membership
```python
import pandas as pd
import numpy as np
obj = pd.Series(["c","a","d","a","a","b","b","c","c"])
mask = obj.isin(["b","c"])                                        # vectorised set membership (McKinney p189)
data = pd.DataFrame(np.arange(16).reshape((4,4)), index=["Ohio","Colorado","Utah","New York"],
                    columns=["one","two","three","four"])
print(f"RESULT isin_true={int(mask.sum())} rows_over5={list(data[data['three']>5].index)}")
```
`mask` is True at 5 positions; the row filter keeps Colorado/Utah/New York (three = 6,10,14 > 5). Boolean arrays work with `loc` but not `iloc` (McKinney p164, p168).

### Missing-value detection and dropna
```python
import pandas as pd
obj4 = pd.Series({"Ohio":35000,"Texas":71000,"Oregon":16000,"Utah":5000},
                 index=["California","Ohio","Oregon","Texas"])    # California absent -> NaN (McKinney p145)
print(f"RESULT isna={int(obj4.isna().sum())} notna={int(pd.notna(obj4).sum())} dropna_len={len(obj4.dropna())} dropna_sum={obj4.dropna().sum()}")
```
`isna`/`notna` flag missing data; dict-with-index turns an absent key (California) into NaN and an extra dict key (Utah) is excluded. `dropna()` keeps 3 values, sum 122000 (McKinney p144-145).

### Arithmetic and data alignment
```python
import pandas as pd
import numpy as np
obj3 = pd.Series({"Ohio":35000,"Texas":71000,"Oregon":16000,"Utah":5000})
obj4 = pd.Series({"Ohio":35000,"Texas":71000,"Oregon":16000,"Utah":5000},
                 index=["California","Ohio","Oregon","Texas"])
s1 = pd.Series([7.3,-2.5,3.4,1.5], index=["a","c","d","e"])
s2 = pd.Series([-2.1,3.6,-1.5,4,3.1], index=["a","c","e","f","g"])
print(f"RESULT ohio={obj3['Ohio']+obj4['Ohio']} s1s2_a={round(float((s1+s2)['a']),1)} s1s2_e={float((s1+s2)['e'])} s1s2_d_nan={bool(pd.isna((s1+s2)['d']))}")
```
Arithmetic aligns on index labels and the result index is the union; overlapping pairs add (Ohio 70000, a=5.2, e=0.0), non-overlap -> NaN which propagates in later ops (McKinney p145-146, p170-171).

### Flexible arithmetic with fill_value and broadcasting
```python
import pandas as pd
import numpy as np
df1 = pd.DataFrame(np.arange(12.).reshape((3,4)), columns=list("abcd"))
df2 = pd.DataFrame(np.arange(20.).reshape((4,5)), columns=list("abcde")); df2.loc[1,"b"] = np.nan
frame = pd.DataFrame(np.arange(12.).reshape((4,3)), columns=list("bde"), index=["Utah","Ohio","Texas","Oregon"])
series = frame.iloc[0]
print(f"RESULT add_e0={float(df1.add(df2, fill_value=0).loc[0,'e'])} rdiv_a0={float(df1.rdiv(1).loc[0,'b'])} bc_ohio_b={float((frame-series).loc['Ohio','b'])}")
```
`add(..., fill_value=0)` substitutes 0 for a label present in only one object (row0/e = 0+4 = 4.0); each operator has an `r`-prefixed counterpart (`1/df1` == `df1.rdiv(1)`).
DataFrame-Series matches the Series index on the columns and broadcasts down rows, so Ohio/b = 3-0 = 3.0 (McKinney p172-176).

## Output contract
Each recipe returns a pandas object or scalars: a `Series`/`DataFrame` from creation, `reindex`/`drop`/`dropna` return
new objects (original untouched); `loc`/`iloc` return a scalar, `Series` or `DataFrame` depending on selection shape;
`mask`/`isin` return a Boolean `Series`; `+`/`add` return an aligned object whose index is the union of the inputs. Every block prints `RESULT name=value` lines.

## Validity conditions
- `reindex` requires a unique `Index`; duplicate labels make the selection type vary (scalar vs Series) - check `index.is_unique` (McKinney p155, p182).
- Integer index: `[]` is label-oriented and only `iloc` is positional; `loc[[0,1,2]]` raises when the index has no integers (McKinney p160-162).
- Transposing a mixed-dtype DataFrame drops per-column dtypes (all become object), so `.T.T` is lossy (McKinney p152).
- `df["col"]` returns a view on the frame, not a copy; call `.copy()` before mutating (McKinney p152).

## Out of scope
- Reading/writing files (CSV/JSON/Excel/pickle/SQL) is the loading-IO skill; this file starts from in-memory data.
- Merging/joining/`concat`/`pivot` and hierarchical MultiIndex reshaping belong to the reshape-merging skill.
- Split-apply-combine `groupby`, `agg`/`apply` and `pivot_table`/`crosstab` belong to the group-by skill.
- The full missing-data policy (`fillna`, `dropna(how/thresh)`, duplicate removal, `replace`, dtype conversion) and
  string/categorical cleaning belong to the data-cleaning skill; here only object-level `isna`/`notna`/`dropna` show up.

## Conventions
- Import as `import numpy as np` / `import pandas as pd`; blocks are self-contained with inline data, seeded only via
  `numpy.random.default_rng(42)` if randomness is ever needed (McKinney p142).
- "missing"/"NA"/"null" are interchangeable and mean NaN; never invent a value - compute it and print the `RESULT` line.
- Source: McKinney, *Python for Data Analysis*, 3rd ed. (O'Reilly, 2022), Ch.5 pp.141-192 (PDF pages).
