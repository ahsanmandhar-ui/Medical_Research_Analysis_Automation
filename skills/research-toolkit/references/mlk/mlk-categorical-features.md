---
name: mlk-categorical-features
description: "Encode categorical/string features into numeric ML inputs in pure Python (sklearn/pandas/numpy): label & ordinal reduction with LabelEncoder/OrdinalEncoder and custom rank maps, nominal one-hot/dummy via LabelBinarizer / pd.get_dummies / OneHotEncoder(drop='first') and MultiLabelBinarizer, unseen-category handling with handle_unknown='ignore', dictionary-of-features via DictVectorizer, plus frequency/count and smoothed target (uncertainty) encodings reimplemented on pandas/numpy because category_encoders is not installed. Use when asked to one-hot / dummy / ordinalize / label-encode a column, tame high-cardinality features, target-encode a group, or stop one-hot from blowing up memory."
---

# Categorical Feature Encoding (Python)

## Purpose
Turn a string/category column into numbers a model can read, choosing the transform by whether the
classes carry an ordering. Nominal sets (Blue/Red/Green, Texas/California) have no rank so they are
expanded; ordinal sets (Low/Medium/High) keep a rank in one column (Albon ch.5 p96-97).

## Inputs
- One or more categorical columns (object/string) `X[cat]`, or a list of dict rows, or `(cat, y)` pairs
  for count/target encoding.
- `ALPHA = 1.0` target-encoding smoothing (back-off weight toward the global prior); `rng = default_rng(42)`
  for any demo sampling.

## Algorithms (all pure numpy/scipy)
### Nominal: one-hot / dummy (Albon ch.5 p97-99)
```python
import numpy as np, pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, MultiLabelBinarizer
feat = np.array([["Texas"],["California"],["Texas"],["Delaware"],["Texas"]])   # Albon ch.5 p97
oh = LabelBinarizer(); X = oh.fit_transform(feat); cls = [str(c) for c in oh.classes_]
print(f"RESULT onehot_classes={cls} n_binary_features={X.shape[1]} Texas_col_sum={int(X[:, cls.index('Texas')].sum())}")
# dummy-variable trap: keep k-1 columns to break the linear dependence one-hot introduces (Albon ch.5 p99)
Xdrop = OneHotEncoder(drop="first", sparse_output=False).fit_transform(feat)
print(f"RESULT drop_first_shape={Xdrop.shape}")
print(f"RESULT get_dummies_cols={list(pd.get_dummies(feat[:,0]).columns)}")
m = MultiLabelBinarizer().fit_transform([("Texas","Florida"),("California","Alabama")])
print(f"RESULT multilabel_row0={m[0].tolist()}")   # each observation may list several classes (Albon ch.5 p98)
```
Prints `onehot_classes=['California','Delaware','Texas'] n_binary_features=3 Texas_col_sum=3` (Albon ch.5 p97-98).
Assigning Texas=1/California=2 to a nominal set would invent a false hierarchy (Albon ch.5 p99).

### Ordinal: rank map + reduction (Albon ch.5 p99-101)
```python
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
s = pd.DataFrame({"Score":["Low","Low","Medium","Medium","High"]})
print(f"RESULT custom_map={s['Score'].replace({'Low':1,'Medium':2,'High':3}).tolist()}")   # -> [1,1,2,2,3] Albon ch.5 p100
print(f"RESULT ordinal_reduction={OrdinalEncoder(categories=[['Low','Medium','High']]).fit_transform(s).ravel().astype(int).tolist()}")
# equal-interval trap: a near-tie class makes 'High'!=3x'Low' - choose values from prior knowledge (Albon ch.5 p101)
trap = pd.Series(["Low","Medium","Barely More Than Medium","High"])
print(f"RESULT naive={trap.replace({'Low':1,'Medium':2,'Barely More Than Medium':3,'High':4}).tolist()}")
print(f"RESULT conscious={trap.replace({'Low':1,'Medium':2,'Barely More Than Medium':2.1,'High':3}).tolist()}")
```
`custom_map` prints `[1, 1, 2, 2, 3]` (Albon ch.5 p100); the conscious map keeps Medium and its
near-tie 0.1 apart rather than a full step apart (Albon ch.5 p101).

### Custom features from dictionaries (Albon ch.5 p101-103)
```python
from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer(sparse=False)
F = dv.fit_transform([{"Red":2,"Blue":4},{"Red":4,"Blue":3},{"Red":1,"Yellow":2},{"Red":2,"Yellow":2}])
names = [str(c) for c in dv.get_feature_names_out()]
print(f"RESULT dict_names={names} row0={F[0].tolist()} col_Blue_total={int(F[:, names.index('Blue')].sum())}")
```
Prints `dict_names=['Blue','Red','Yellow'] row0=[4.0, 2.0, 0.0]` - keys become columns, missing keys fill
with 0; set `sparse=True` for huge word-count matrices (Albon ch.5 p102-103).

### Unseen categories at transform time
```python
import numpy as np
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(np.array([["Texas"],["California"],["Delaware"]]))
out = enc.transform(np.array([["Texas"],["Vermont"]]))     # 'Vermont' never seen in fit
print(f"RESULT unseen_row={out[1].tolist()} n_seen_classes={len(enc.categories_[0])}")
```
`handle_unknown="ignore"` maps an unseen class to an all-zero vector instead of raising; the default
`handle_unknown="error"` aborts. Use it when new categories are expected in production.

### Frequency / count encoding (category_encoders re-implemented)
```python
import numpy as np, pandas as pd
cat = np.array(["a","b","a","c","a","b","a"])
counts = pd.Series(cat).value_counts()
freq = (pd.Series(cat).map(counts) / len(cat)).round(4)
print(f"RESULT count_map={ {k:int(v) for k,v in counts.items()} } n_unique={counts.size}")
print(f"RESULT freq_enc={freq.tolist()}")
```
Replaces each class with how often it appears (or that share of n) - one column, cheap for high
cardinality where one-hot explodes the feature count.

### Target / uncertainty encoding (smoothed, category_encoders re-implemented)
```python
import numpy as np, pandas as pd
cat = np.array(["a","b","a","c","a","b","a"]); y = np.array([1,0,1,0,1,1,1]); ALPHA = 1.0
d = pd.DataFrame({"cat":cat,"y":y}); prior = d["y"].mean()
raw  = {c: float(g["y"].mean()) for c,g in d.groupby("cat")}
enc  = {c: float((g["y"].sum()+ALPHA*prior)/(len(g)+ALPHA)) for c,g in d.groupby("cat")}
print(f"RESULT prior={prior:.4f} raw_rate={ {k:round(v,3) for k,v in raw.items()} }")
print(f"RESULT smoothed={ {k:round(v,4) for k,v in enc.items()} }")
```
Encode a class by its mean target, but pull rare classes toward the prior by `ALPHA` (a class of n=1 is
high-uncertainty, so it cannot claim the raw 0/1). Fit only on the training folds.

## Output contract
Each block prints `RESULT name=value` lines. Downstream you hand the model: one-hot/dummy = a
`[n_samples, k]` or `[n_samples, k-1]` float matrix; ordinal/label/count/target = a single `[n_samples]`
numeric column; DictVectorizer = a dense or sparse `[n_samples, n_keys]` matrix. Preserve the encoder
(`classes_`, `categories_`, the mapper dict, `get_feature_names_out()`) so `transform`/`inverse_transform`
stay aligned with training.

## Validity conditions
- One-hot adds `k` (or `k-1`) columns per feature: with high cardinality it blows up memory and the
  Euclidean distances most models compute (Albon ch.5 p96-97); switch to count/ordinal/target.
- Never assign ordered integers to a nominal set - it fabricates a ranking the data lacks (Albon ch.5 p99).
- Ordinal numeric values must reflect real prior intervals, not just the label order (Albon ch.5 p100-101).
- Target encoding leaks the label unless the statistics are fit on train only (cross-fold), and small
  groups must be smoothed toward the prior (Albon ch.5 p97 rationale for shrinking high-variance encodings).

## Conventions
- `category_encoders` is NOT installed here; count/frequency and smoothed target/uncertainty encodings are
  re-implemented on `pandas`/`numpy`/`sklearn` (the two blocks above), and `DictVectorizer`/`OneHotEncoder`
  cover the rest. Label reduction uses `sklearn.preprocessing.LabelEncoder`/`OrdinalEncoder`.
- `sparse_output=False` for readable dense arrays; `get_feature_names_out()` (not the removed
  `get_feature_names`); use `drop="first"` only when the model needs no dummy-trap collinearity.
- Seeded demo data only via `numpy.random.default_rng(42)`; compute and print every value - never hardcode
  a result. Source: Albon, *Machine Learning with Python Cookbook* 2e, Ch.5 "Handling Categorical Data" (Albon ch.5 p96-109).

## Out of scope
Imputing missing categorical values (Albon ch.5 p103-105) and imbalanced-class handling - class_weight,
down/upsampling (Albon ch.5 p105-109) - are separate skills; this one only transforms observed categories
into numbers.
