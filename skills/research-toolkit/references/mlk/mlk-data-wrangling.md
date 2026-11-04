---
name: mlk-data-wrangling
description: Wrangle a pandas DataFrame end to end in pure Python (pandas/numpy) - creation, description (head/tail/shape/describe), loc vs iloc and conditional selection, replace and rename, min/max/sum/mean/count and unique/value_counts, missing-value detection, column and row deletion, deduplication with subset and keep, groupby aggregation with apply, resample over time bins, concat and merge with every join type. Use when asked to clean or reshape a table, drop duplicate rows, filter rows on a condition, group by a category, count unique classes, handle NaN, join two frames or resample a time series.
---

# Data Wrangling with pandas (Python)

## Purpose
Turn a raw DataFrame into a clean, well-structured set of observations for the next preprocessing step -
select, rename, aggregate, delete, dedupe, group, combine; rows are observations, columns are features (Albon p48-49).

## Inputs
- `dataframe`: tabular frame; worked examples use columns `Name, PClass, Age, Sex, Survived, SexCode`, `n = 1313` rows, `6` columns (Albon p51).
- `subset` / `keep` set the dedup scope and which survivor stays; `how` sets the join type; `axis=0` rows, `axis=1` columns.
- Treat every frame as immutable: reassign the returned frame, never `inplace=True` (Albon p63).

## Algorithms (all pure numpy/scipy)

### Creating and describing a frame (3.1-3.2)
```python
import numpy as np, pandas as pd
rng = np.random.default_rng(42); n = 1313
df = pd.DataFrame({"PClass": rng.permutation(np.array(["1st"]*322 + ["2nd"]*279 + ["3rd"]*711 + ["*"])), "Sex": rng.permutation(np.array(["female"]*462 + ["male"]*851)), "Survived": rng.permutation(np.array([1]*450 + [0]*863))})
obs = np.zeros(n, bool)
obs[rng.choice(np.flatnonzero(df["Sex"].eq("female")), 288, replace=False)] = True
obs[rng.choice(np.flatnonzero(df["Sex"].eq("male")), 468, replace=False)] = True
df["Age"] = np.where(obs, np.round(rng.uniform(0.17, 71.0, n), 2), np.nan)
df.insert(0, "Name", [f"P{i:04d}" for i in range(n)]); df["SexCode"] = df["Sex"].eq("female").astype(int)
d = df[["Name", "PClass", "Age", "Sex", "Survived", "SexCode"]].describe()
print("RESULT shape =", df.shape, "| head5 =", len(df.head(5)), "| tail2 =", len(df.tail(2)), "| count =", int(d.loc["count", "Age"]), "| missing =", int(df["Age"].isnull().sum()))
print(f"RESULT Survived_mean={d.loc['mean', 'Survived']:.6f} SexCode_mean={d.loc['mean', 'SexCode']:.6f}")
```
The frame carries the book's printed marginals (462 female / 851 male, 450 survivors, 288/468 observed ages;
Albon p59, p66, p70), so it reproduces the printed `describe` block: `shape = (1313, 6)`, `count = 756`,
`missing = 557`, means `0.342727` and `0.351866` (Albon p51); only the age values are seeded. `head`/`tail`,
`shape` and `describe` are the first three looks after loading, and `DataFrame.append` is gone - append a row
with `pd.concat([df, series.to_frame().T], ignore_index=True)` (Albon p50-52).

### Navigating, filtering and renaming (3.3-3.6)
```python
import pandas as pd
df = pd.DataFrame({"Name": ["Allen, Miss Elisabeth Walton", "Allison, Miss Helen Loraine", "Allison, Mr Hudson Joshua Creighton", "Crosby, Mrs Edward Gifford (Catherine...)"], "PClass": ["1st"]*4, "Age": [29.0, 2.0, 30.0, 69.0], "Sex": ["female", "female", "male", "female"], "Survived": [1, 0, 0, 1]}, index=[0, 1, 2, 73])
print("RESULT iloc0 =", df.iloc[0]["Name"], "| slice =", df.iloc[1:3].index.tolist(), "| upto3 =", len(df.iloc[:3]))
print("RESULT by_label =", df.set_index("Name").loc["Allen, Miss Elisabeth Walton", "PClass"])
w = df[(df["Sex"] == "female") & (df["Age"] >= 65)]
print("RESULT women_65plus =", len(w), "| they_survived =", w["Survived"].tolist(), "| replaced =", df["Sex"].replace(["female", "male"], ["Woman", "Man"]).tolist())
print("RESULT regex =", df.replace(r"1st", "First", regex=True)["PClass"][0], "| renamed =", df.rename(columns={"PClass": "Passenger Class", "Sex": "Gender"}).columns.tolist())
```
Expects `women_65plus = 1`, that passenger being the survivor Mrs Crosby (Albon p54). `loc` matches labels,
`iloc` matches positions; parenthesise each condition and join with `&`/`|`; `replace` accepts lists and
`regex=True`; `rename(columns=dict)` is the portable form, and a `collections.defaultdict(str)` filled from
`df.columns` builds a rename-all dict (Albon p52-57).

### Aggregates, unique classes and missing values (3.7-3.9)
```python
import numpy as np, pandas as pd, io
df = pd.DataFrame({"PClass": ["1st"]*4 + ["2nd"]*5 + ["3rd"]*7 + ["*"], "Sex": ["female"]*8 + ["male"]*9, "Age": list(range(16)) + [np.nan]})
print("RESULT max/min/mean/count =", df["Age"].max(), df["Age"].min(), round(df["Age"].mean(), 3), df["Age"].count(), "| unique =", df["PClass"].unique().tolist(), "| nunique =", df["PClass"].nunique())
print("RESULT value_counts =", df["PClass"].value_counts().to_dict(), "| isnull_rows =", len(df[df["Age"].isnull()]), "| notnull =", df["Age"].notnull().sum())
print("RESULT na_values_nulls =", pd.read_csv(io.StringIO("PClass,Age\n1st,-999\nNONE,25\n3rd,"), na_values=["NONE", -999])["Age"].isnull().sum())
```
`count` is 16 of 17 rows because every aggregate skips missing values - always report it beside the mean (the
book's `Age` is `count = 756` of 1313, mean 30.398, Albon p57-58). `Sex` has 2 uniques but `PClass` has
`nunique() = 4` thanks to the stray `*` class wrangling must catch (Albon p59). Use `np.nan`, never a bare
`NaN`; find gaps with `isnull()`/`notnull()`; declare sentinels like `-999`/`NONE` in `na_values=[...]` (Albon p60-61).

### Deleting columns, deleting rows, deduplicating (3.10-3.12)
```python
import pandas as pd
df = pd.DataFrame({"Name": list("ABCDEF"), "PClass": ["1st", "1st", "2nd", "2nd", "1st", "3rd"], "Age": [29.0, 29.0, 30.0, 25.0, 30.0, 40.0], "Sex": ["female", "female", "male", "male", "male", "female"], "Survived": [1, 1, 0, 0, 1, 0]})
print("RESULT drop_one =", list(df.drop("Age", axis=1).columns), "| drop_list =", list(df.drop(["Age", "Sex"], axis=1).columns), "| drop_pos =", list(df.drop(df.columns[1], axis=1).columns))
print("RESULT not_female =", df[df["Sex"] != "female"]["Name"].tolist(), "| index_ne_0 =", df[df.index != 0]["Name"].tolist())
dupe = pd.concat([df, df.iloc[[0, 2]]], ignore_index=True)
print("RESULT rows", len(dupe), "-> drop_duplicates()", len(dupe.drop_duplicates()), "-> subset=['Sex']", len(dupe.drop_duplicates(subset=["Sex"])), "-> keep='last'", dupe.drop_duplicates(subset=["Sex"], keep="last")["Age"].tolist(), "| duplicated()", int(dupe.duplicated().sum()))
```
On the Titanic frame dedup changes nothing, `1313 -> 1313`, because `drop_duplicates` needs an exact match on
all columns; `subset` dedupes on a few columns and keeps the first occurrence unless `keep=` says otherwise,
while `duplicated()` flags instead of deleting (Albon p64-65). Delete rows with a negated boolean rather than
`drop(axis=0)`; delete columns with `drop(..., axis=1)`; `del` works but is discouraged (Albon p61-63).

### Grouping rows and applying a function per group (3.13, 3.17)
```python
import pandas as pd
F_AGE, F_N, M_AGE, M_N, F_S, M_S, NF, NM = 29.396424, 288, 31.014338, 468, 0.666667, 0.166863, 462, 851
age_mean = (F_AGE*F_N + M_AGE*M_N) / (F_N + M_N); surv_mean = (F_S*NF + M_S*NM) / (NF + NM)
assert abs(age_mean - 30.397989) < 1e-5 and abs(surv_mean - 0.342727) < 1e-5
df = pd.DataFrame({"Name": list("ABCDEF"), "Sex": ["female", "female", "male", "male", "male", "female"], "Survived": [1, 0, 1, 0, 0, 1], "Age": [29.0, 25.0, 30.0, 40.0, 22.0, 35.0]})
print("RESULT age_mean =", round(age_mean, 6), "| survived_mean =", round(surv_mean, 6), "| group_mean =", df.groupby("Sex")["Age"].mean().round(6).to_dict())
print("RESULT by_survived =", df.groupby("Survived")["Name"].count().to_dict(), "| two_key =", df.groupby(["Sex", "Survived"])["Age"].mean().round(6).to_dict())
print("RESULT apply =", df.groupby("Sex").apply(lambda x: x.count(), include_groups=False).to_dict())
```
Weighting the printed per-group age means by the printed per-group non-missing counts (Albon p65, p70)
rebuilds the book's overall age mean `30.397989417989415` (Albon p57), and the same identity on `Survived`
returns `0.342727` = 450/1313 (Albon p51, p66). `groupby` alone returns an object, not numbers: pair it with a
statistic that suits the dtype, group on 2 keys for a MultiIndex, and use `groupby().apply(func)` for a custom
per-group statistic (Albon p65-66, p70).

### Grouping rows by time (3.14)
```python
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
ts = pd.DataFrame({"Sale_Amount": rng.integers(1, 10, 100000)}, index=pd.date_range("06/06/2017", periods=100000, freq="30s"))
print("RESULT weekly_sum_first =", int(ts.resample("W").sum()["Sale_Amount"].iloc[0]), "| 2W_mean =", ts.resample("2W").mean()["Sale_Amount"].round(6).tolist())
print("RESULT monthly_count =", [int(v) for v in ts.resample("ME").count()["Sale_Amount"]], "| label_right =", ts.resample("ME").count().index.strftime("%Y-%m-%d").tolist(), "| label_left =", ts.resample("ME", label="left").count().index.strftime("%Y-%m-%d").tolist())
```
`resample` needs a datetime-like index; `W`/`2W`/`ME` are offset aliases and the default label is the right
edge of the bin, moved with `label="left"` (Albon p66-68). The monthly counts `72000` and `28000` reproduce
the book exactly on a 30 s grid; the amounts, weekly sum and two-week means are seeded, not printed (Albon p68).

### Concatenating and merging frames (3.18-3.19)
```python
import pandas as pd
a = pd.DataFrame({"id": ["1", "2", "3"], "first": ["Alex", "Amy", "Allen"], "last": ["Anderson", "Ackerman", "Ali"]})
b = pd.DataFrame({"id": ["4", "5", "6"], "first": ["Billy", "Brian", "Bran"], "last": ["Bonder", "Black", "Balwner"]})
print("RESULT concat_rows =", pd.concat([a, b], axis=0).shape, "| concat_cols =", pd.concat([a, b], axis=1).shape, "| ids =", pd.concat([a, b], axis=0, ignore_index=True)["id"].tolist())
emp = pd.DataFrame({"employee_id": ["1", "2", "3", "4"], "name": ["Amy Jones", "Allen Keys", "Alice Bees", "Tim Horton"]})
sal = pd.DataFrame({"employee_id": ["3", "4", "5", "6"], "total_sales": [23456, 2512, 2345, 1455]})
inner = pd.merge(emp, sal, on="employee_id"); outer = pd.merge(emp, sal, on="employee_id", how="outer")
print("RESULT inner_rows =", len(inner), "| inner_sales =", int(inner["total_sales"].sum()), "| outer_rows =", len(outer), "| outer_NaN_names =", int(outer["name"].isnull().sum()))
print("RESULT left =", len(pd.merge(emp, sal, on="employee_id", how="left")), "| right =", len(pd.merge(emp, sal, on="employee_id", how="right")), "| left_on_right_on =", len(pd.merge(emp, sal, left_on="employee_id", right_on="employee_id")))
```
`axis=0` stacks rows (6x3), `axis=1` places frames side by side (3x6); the book's inner join is 2 rows worth
`23456 + 2512 = 25968` and its outer join is 6 rows with 2 NaN names (Albon p71, p73). `merge` defaults to
`how='inner'`; the first frame is the left one, `left_on`/`right_on` handle differently named keys, and
`left_index=True`/`right_index=True` join on indexes (Albon p72-75).

## Output contract
Each step returns a NEW frame, Series or scalar - print `RESULT name=value` per quantity. Aggregations return
one row per group key (MultiIndex for 2+ keys); `value_counts` returns one count per unique class; `merge`
with `on=` keeps the key column once; always print row counts before and after a `drop`, boolean filter or
`drop_duplicates`, since the operation may legitimately drop 0 rows.

## Validity conditions
- Features must be unique: keep either `Sex` or `SexCode`, never both (Albon p49).
- `describe` statistics are meaningless for 0/1 category codes such as `SexCode` (Albon p51-52).
- A group statistic needs a compatible dtype, and every printed aggregate skips NaN, so report `count` with the mean (Albon p57, p66).
- `resample` requires a datetime-like index (Albon p67).
- A merge needs a shared key of matching dtype; an unexpected class (`*`) or a duplicate-heavy key silently changes row counts (Albon p59, p64, p72).

## Reporting conventions
Report rows before/after every drop, filter, dedup and merge, the columns touched, the missing count per
column, and units for aggregated columns (years for `Age`); quote a group statistic with its group size, e.g.
"mean age 30.398 years (n = 756 of 1313 passengers)" (Albon p51, p57, p66).

## Out of scope
Encoding categories into numbers (one-hot/ordinal/label) is the categorical-features skill (Albon ch.5);
binning, outlier clipping, imputation, standardising and normalising are the numerical-features skill (Albon
ch.4 p76-95); reading files, Excel, JSON and SQL is the data-loading skill (Albon ch.2); MultiIndex
stack/unstack reshaping is the reshape skill.

## Conventions
- Seed all demo data with `numpy.random.default_rng(42)`; never hardcode a result - compute it, print it and cite the page whose printed number it reproduces.
- pandas >= 2.1 spellings used here: `freq="30s"` not `"30S"`, `resample("ME")` not `"M"`, `pd.concat` instead of the removed `DataFrame.append`, `groupby().apply(..., include_groups=False)` (Albon p50, p68, p70).
- Prefer vectorised methods or `apply` over a `for` loop on a column (Albon p69); never `inplace=True` (Albon p62-63); write nothing outside a `tempfile` dir.
- Source: Albon, *Machine Learning with Python Cookbook* 2nd ed., Ch.3 Data Wrangling, PDF pages 48-75
  (Recipes 3.0-3.19); the chapter's Titanic CSV is replaced by a seeded frame holding its printed marginals.
