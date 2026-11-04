---
name: mlk-data-loading
description: Load data into a training-ready pandas/sklearn frame from every source in Albon ch.2 - scikit-learn toy datasets, seeded simulated generators (make_regression/make_classification/make_blobs), and CSV/Excel/JSON/SQL files read via pandas read_csv/read_excel/read_json/read_sql_query (sqlite3). Produces an aligned (features, target) pair plus shape/dtype checks. Use when asked to "load a sample/simulated dataset", "import a CSV/Excel/JSON", or "query a database into a DataFrame".
---

# Loading Data into a Training-Ready Frame (Python)

## Purpose
Get raw data from any source into one tidy frame, then split it into a numeric feature matrix `X` and a
target vector `y` ready for a model. pandas does the file reading; scikit-learn supplies toy and simulated
data (Albon ch.2 p38-47).

## Inputs
- A source: a sklearn loader name, generator hyper-parameters, or a file/DB path.
- Toy/simulated data are offline and deterministic; **never** fetch the book's `tinyurl` URLs - build small
  seeded frames with `numpy.random.default_rng(42)` that keep the original shapes and dtypes (Albon p43).
- Column contract for the CSV/Excel/JSON example: `integer` int, `datetime` datetime, `category` int (100 rows).

## Algorithms (all pure numpy/scipy)

### 2.1 Load a sample (toy) dataset (Albon p38-39)
Split `.data`/`.target`; toy sets are clean, so no preprocessing is needed before a model.
```python
from sklearn import datasets
digits = datasets.load_digits()          # 1,797 handwritten-digit images
X_d, y_d = digits.data, digits.target    # training-ready feature matrix + target vector
iris = datasets.load_iris()              # 150 flower measurements
print(f"RESULT digits={X_d.shape} classes={sorted(set(y_d.tolist()))} iris={iris.data.shape}")
```
Prints `digits=(1797, 64) classes=[0..9] iris=(150, 4)` - the 1,797 count is the book's figure (p39).

### 2.2 Create a simulated dataset (Albon p40-42)
Pick the generator by task; `n_informative < n_features` leaves redundant features; `weights=[.25,.75]`
makes an imbalanced classification set; `centers` sets the blob count.
```python
import tempfile, os
import numpy as np
from sklearn.datasets import make_regression, make_classification, make_blobs
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
Xr, yr, coef = make_regression(n_samples=100, n_features=3, n_informative=3,
                               n_targets=1, noise=0.0, coef=True, random_state=1)
Xc, yc = make_classification(n_samples=100, n_features=3, n_informative=3,
                             n_redundant=0, n_classes=2, weights=[.25, .75], random_state=1)
Xb, yb = make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=1)
fig, ax = plt.subplots(); ax.scatter(Xb[:, 0], Xb[:, 1], c=yb)          # visualize the 3 clusters
out = os.path.join(tempfile.mkdtemp(), "blobs.png"); fig.savefig(out); plt.close(fig)
print(f"RESULT reg_target3={np.round(yr[:3],2).tolist()} cls_counts={np.bincount(yc).tolist()} "
      f"blobs_target3={yb[:3].tolist()} n_centers={len(set(yb.tolist()))}")
```
Prints `reg_target3=[-10.38, 25.51, 19.68] cls_counts=[25, 75] blobs_target3=[0, 1, 1] n_centers=3` - these
reproduce the book's printed target vectors and the 25%/75% class balance exactly (Albon p40-42).

### 2.3-2.5 Load CSV / Excel / JSON files (Albon p43-45)
Read from a local/seeded frame, not the URL. `sep`/`header` fix CSV layout; `sheet_name` + `header=1`
skip the Excel title row; `orient` tells pandas how the JSON is nested.
```python
import tempfile, os
import numpy as np, pandas as pd
rng = np.random.default_rng(42); td = tempfile.mkdtemp()
df = pd.DataFrame({"integer": rng.integers(1, 6, 100),
                   "datetime": pd.date_range("2015-01-01", periods=100, freq="s"),
                   "category": rng.integers(0, 2, 100)})
df.to_csv(os.path.join(td, "sim.csv"), index=False)
df.to_excel(os.path.join(td, "sim.xlsx"), index=False, startrow=1)      # row 0 is a title -> header=1
df.to_json(os.path.join(td, "sim.json"), orient="columns")
csv = pd.read_csv(os.path.join(td, "sim.csv"), sep=",", header=0, parse_dates=["datetime"])
xls = pd.read_excel(os.path.join(td, "sim.xlsx"), sheet_name=0, header=1)
jsn = pd.read_json(os.path.join(td, "sim.json"), orient="columns")
print(f"RESULT csv={csv.shape} dtypes={dict(csv.dtypes.astype(str))} xls={xls.shape} json={jsn.shape}")
```
Prints `csv=(100, 3) dtypes={'integer':'int64','datetime':'datetime64[us]','category':'int64'}` and matching
`xls`/`json` shapes - the 100x3 integer/datetime/category grid the book loads (Albon p43-45).

### 2.6 Query a SQL database (Albon p45-46)
`read_sql_query` runs SQL against a connection and returns a frame; `SELECT * FROM data` pulls all columns.
```python
import tempfile, os, sqlite3
import pandas as pd
td = tempfile.mkdtemp(); con = sqlite3.connect(os.path.join(td, "sample.db"))
rows = pd.DataFrame({"first_name": ["Jason", "Molly", "John", "David"],
                     "last_name": ["Miller", "Jacobson", "Johnson", "Hill"],
                     "age": [42, 52, 36, 91], "preTestScore": [4, 24, 31, 2],
                     "postTestScore": [25, 94, 14, 36]})
rows.to_sql("data", con, index=False)
data = pd.read_sql_query("SELECT * FROM data", con); con.close()
print(f"RESULT sql={data.shape} row0={data.iloc[0].tolist()} cols={list(data.columns)}")
```
Prints `row0=['Jason', 'Miller', 42, 4, 25]` - the book's first SQL row, reproduced verbatim (Albon p46).

## Output contract
Every path yields a `(features, target)` pair or a single `DataFrame`. Report `RESULT name=value` lines with
`shape`, `dtypes`, target `classes`/`counts`, and confirm `len(X) == len(y)` before handing to a model. Toy
generators return float features with integer targets; `make_regression` returns float targets (Albon p41).

## Validity conditions
- Align `X` and `y` on the same row count/order; a shuffled target invalidates supervised training (Albon p41).
- CSV header may sit on row 0 or be absent (`header=None`); a wrong `header`/`sep` silently shifts every column (Albon p43).
- For JSON, the correct `orient` (`split`, `records`, `index`, `columns`, `values`) must match the file's nesting or the frame comes out transposed (Albon p45).

## Conventions
- Seeds: generators use `random_state=1` to match the book; synthetic CSV/Excel/JSON frames use `default_rng(42)`. Never fetch the `tinyurl` files.
- API drift vs the 2018 book (all run on current pandas/sklearn): `read_excel(sheetname=)` is now `sheet_name=`, and `read_csv`'s `sep`/`header` names are unchanged (Albon p44).
- `read_sql_query` accepts a stdlib `sqlite3` connection directly; the book's `sqlalchemy.create_engine` is optional and not installed - use `sqlite3`.
- `load_boston` (503 obs, Albon p39) was removed from scikit-learn for ethical reasons - use `load_digits`/`load_iris` or `fetch_california_housing`; do not call it.
- Headless plots use `matplotlib.use("Agg")` and save into a `tempfile` dir; write files only inside `tempfile`, never the CWD.

## Out of scope
Feature/type encoding of categorical/string columns belongs to the categorical-features skill; NA handling,
dedup and reshaping belong to the cleaning/wrangling skills; train/test splitting and scaling are separate.
This skill stops at a loaded, aligned, dtype-correct frame (Albon ch.2 p38-47).
