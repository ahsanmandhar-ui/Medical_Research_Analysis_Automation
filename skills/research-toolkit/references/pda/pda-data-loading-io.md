---
name: pda-data-loading-io
description: Read data into pandas and write it back out - CSV/TSV and other delimiters, JSON, Excel (openpyxl), pickle, sqlite3 SQL - plus nrows/chunksize for huge files and skiprows/csv-module handling of malformed lines, in pure Python. Parquet and HDF5 are honest-degradation paths (pyarrow/tables not installed). Use when asked to load a file, parse a delimiter, honor missing-value sentinels, iterate a very large CSV, round-trip Excel/JSON/pickle, or query a SQLite table into a DataFrame.
---

# Data Loading & File I/O (Python)

## Purpose
Get bytes off disk or network into a pandas DataFrame and write a DataFrame back out, faithful to the
source format. Cleaning, imputing, de-duplicating and reshaping after ingestion are a sibling skill's job
(McKinney p193).

## Inputs
- `path`: a local file path, or a URL string (read_csv/read_json take URLs directly, McKinney p195).
- `sep`/`delimiter`: `","` default; `"\t"` for TSV; the regex `"\s+"` for whitespace-split fields.
- `na_values`, `keep_default_na`: missing-value sentinels (NA/NULL recognised by default, McKinney p198).
- `nrows`, `chunksize`: partial and piecemeal reads of large files (McKinney p201-202).
- Any synthesized demo file is seeded with `numpy.random.default_rng(42)`.

## Algorithms (all pure numpy/scipy)

### Delimited text: CSV/TSV read, write, missing sentinels
```python
import tempfile, os, pandas as pd
d = tempfile.mkdtemp()
csvp = os.path.join(d, "ex1.csv")
with open(csvp, "w", newline="") as f:
    f.write("a,b,c,d,message\n1,2,3,4,hello\n5,6,7,8,world\n9,10,11,12,foo\n")
df = pd.read_csv(csvp)
print("RESULT shape=%s cell_c_row2=%s" % (df.shape, df.loc[2, "c"]))
tsvp = os.path.join(d, "ex1.tsv")
df.to_csv(tsvp, sep="\t", index=False)
back = pd.read_csv(tsvp, sep="\t")
print("RESULT tsv_roundtrip=%s" % back.equals(df))
ex5 = os.path.join(d, "ex5.csv")
with open(ex5, "w", newline="") as f:
    f.write("something,a,b,c,d,message\none,1,2,3,4,NA\ntwo,5,6,,8,world\nthree,9,10,11,12,foo\n")
m = pd.read_csv(ex5)
print("RESULT n_missing=%d" % int(m.isna().sum().sum()))
```
The ex1 file reads to `(3, 5)` with `c`-column row-2 value 11, and ex5 yields exactly two nulls (the `NA`
and the empty field) - both reproduce the book tables (McKinney p195, p198).

### Large files: nrows head and chunksize aggregation
```python
import tempfile, os, numpy as np, pandas as pd
d = tempfile.mkdtemp()
rng = np.random.default_rng(42)
big = pd.DataFrame(rng.standard_normal((10000, 4)), columns=["one", "two", "three", "four"])
big["key"] = rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), size=10000)
p = os.path.join(d, "ex6.csv"); big.to_csv(p, index=False)
head = pd.read_csv(p, nrows=5)
tot = pd.Series([], dtype="int64"); n_chunks = 0
for piece in pd.read_csv(p, chunksize=1000):
    tot = tot.add(piece["key"].value_counts(), fill_value=0); n_chunks += 1
print("RESULT nrows_head=%d total_rows=%d chunks=%d top_key=%s" % (len(head), int(tot.sum()), n_chunks, tot.idxmax()))
```
`chunksize` returns a TextFileReader you iterate row-block by row-block, aggregating without loading the
whole 10000x5 file into memory (McKinney p201-202).

### Malformed and comment lines
```python
import tempfile, os, csv, pandas as pd
d = tempfile.mkdtemp()
ex4 = os.path.join(d, "ex4.csv")
with open(ex4, "w", newline="") as f:
    f.write("# hey!\na,b,c,d,message\n# just wanted to make things harder\n# who reads CSV by hand?\n1,2,3,4,hello\n5,6,7,8,world\n9,10,11,12,foo\n")
sk = pd.read_csv(ex4, skiprows=[0, 2, 3])
print("RESULT skiprows_shape=%s" % (sk.shape,))
ex7 = os.path.join(d, "ex7.csv")
with open(ex7, "w", newline="") as f:
    f.write('"a","b","c"\n"1","2","3"\n"1","2","3"\n')
with open(ex7, newline="") as f:
    lines = list(csv.reader(f))
header, values = lines[0], lines[1:]
data_dict = {h: v for h, v in zip(header, zip(*values))}
print("RESULT malformed_keys=%s col_a=%s" % (sorted(data_dict), list(data_dict["a"])))
```
`skiprows` drops stray comment rows so a header lands correctly (McKinney p197); the stdlib `csv.reader`
plus `zip(*values)` transposes irregular quoted lines into a column dict (McKinney p203-204).

### JSON: string, file, DataFrame round-trip
```python
import tempfile, os, json, pandas as pd
d = tempfile.mkdtemp()
obj = '{"name": "Wes", "cities_lived": ["Akron","Nashville","New York","San Francisco"], "pet": null, "siblings": [{"name": "Scott", "age": 34, "hobbies": ["guitars","soccer"]}, {"name": "Katie", "age": 42, "hobbies": ["diving","art"]}]}'
siblings = pd.DataFrame(json.loads(obj)["siblings"], columns=["name", "age"])
print("RESULT age_sum=%d" % int(siblings["age"].sum()))
ej = os.path.join(d, "example.json")
with open(ej, "w") as f:
    f.write('[{"a": 1, "b": 2, "c": 3}, {"a": 4, "b": 5, "c": 6}, {"a": 7, "b": 8, "c": 9}]')
data = pd.read_json(ej)
out = os.path.join(d, "round.json"); data.to_json(out, orient="records")
print("RESULT json_shape=%s a_sum=%d roundtrip=%s" % (data.shape, int(data["a"].sum()), pd.read_json(out).equals(data)))
```
`age_sum=76` (Scott 34 + Katie 42) and `a_sum=12` reproduce the Wes and example.json tables (McKinney p205-206).

### Excel and pickle round-trips
```python
import tempfile, os, numpy as np, pandas as pd
d = tempfile.mkdtemp()
rng = np.random.default_rng(42)
frame = pd.DataFrame({"a": rng.integers(0, 10, 3), "b": ["x", "y", "z"]})
xp = os.path.join(d, "ex1.xlsx")
frame.to_excel(xp, index=False, sheet_name="Sheet1")
xl = pd.ExcelFile(xp)
print("RESULT sheets=%s xlsx_equal=%s" % (xl.sheet_names, xl.parse("Sheet1").equals(frame)))
pp = os.path.join(d, "frame_pickle")
frame.to_pickle(pp)
print("RESULT pickle_equal=%s" % pd.read_pickle(pp).equals(frame))
```
Excel goes through openpyxl; pickle is a short-term binary store only - an object pickled today may not
unpickle in a later library version (McKinney p211-213).

### SQL: sqlite3 build and read_sql
```python
import tempfile, os, sqlite3, pandas as pd
d = tempfile.mkdtemp()
con = sqlite3.connect(os.path.join(d, "mydata.sqlite"))
con.execute("CREATE TABLE test (a VARCHAR(20), b VARCHAR(20), c REAL, d INTEGER)")
con.executemany("INSERT INTO test VALUES(?, ?, ?, ?)",
                [("Atlanta", "Georgia", 1.25, 6), ("Tallahassee", "Florida", 2.6, 3), ("Sacramento", "California", 1.7, 5)])
con.commit()
res = pd.read_sql("SELECT * FROM test", con)
print("RESULT db_rows=%d mean_c=%.2f" % (len(res), res["c"].mean()))
con.close()
```
Insert Atlanta/Tallahassee/Sacramento, read the table back; `mean_c=1.85` is the mean of c=(1.25,2.6,1.7)
(McKinney p217-219). With SQLAlchemy installed the same is `pd.read_sql(sql, sqla.create_engine("sqlite:///..."))`.

## Output contract
Each reader returns a DataFrame (a Series with `squeeze`); each writer produces a file at the given path.
Every block prints `RESULT name=value`: `shape`, `n_missing`, `total_rows`, `chunks`, `json_shape`,
`sheets`, `db_rows`, `mean_c`, plus `*_roundtrip` / `*_equal` booleans that prove lossless write-then-read.

## Parsing rules
- `header=0` by default; pass `header=None` or `names=[...]` for a file with no header row (McKinney p195-196).
- One fewer column name than data fields makes pandas use the first column as the index (McKinney p197).
- `index_col` (single name or list) sets the row / hierarchical index at read time (McKinney p196-197).
- `na_values` adds to the default sentinel list unless `keep_default_na=False`; a per-column dict is allowed (McKinney p199-200).

## Validity conditions
- Type inference is not part of text formats, so numeric/int/bool/date columns can need explicit converters (McKinney p195).
- `pickle` format is not guaranteed stable across library versions - never use it for long-term storage (McKinney p211).
- HDF5 is write-once/read-many; simultaneous writers corrupt the file (McKinney p215).

## Out of scope
Missing-value imputation, string/type cleaning, de-duplication, pivot/melt and merge/join/concat belong to
the data-cleaning and data-reshaping siblings (McKinney Ch.7-8).

## Conventions
- Delimiters: default `sep=","`; TSV `sep="\t"`; whitespace `sep="\s+"`; encoding defaults to utf-8 (McKinney p200).
- Demo CSV/JSON/XLSX/SQLite files are synthesized inline inside a `tempfile` dir; seed with `numpy.random.default_rng(42)`; never hardcode a value - always compute it and print `RESULT name=value`.
- Optional-dependency formats are honestly degraded, never imported blind: Parquet (`read_parquet`) needs
  pyarrow - user installs with `conda install pyarrow` (McKinney p212); HDF5 (`HDFStore`/`read_hdf`) needs
  PyTables - user installs with `conda install pytables` (PyPI name `tables`, McKinney p213-215). Neither is
  present here, so ship that data through CSV or `to_pickle`/`read_pickle`; do not run these installs yourself.
- SQLAlchemy is optional; stdlib `sqlite3` + `pd.read_sql(sql, conn)` covers single-file databases (McKinney p217).
- Source: McKinney, *Python for Data Analysis*, 3rd ed. (O'Reilly, 2022), Ch.6, pp.193-220.
