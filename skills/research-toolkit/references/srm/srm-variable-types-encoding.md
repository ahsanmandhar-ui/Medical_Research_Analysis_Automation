---
name: srm-variable-types-encoding
description: Classify every variable as quantitative (discrete or continuous) or qualitative (nominal or ordinal) and encode it into the correct pandas dtype and declared categorical levels, pure Python (pandas/numpy). Applies the measurement-process definitions, the numeric-but-qualitative trap (postcodes), the ordinal natural-order rule, the jamovi/SPSS declaration map and the summaries-are-not-variables check. Use when asked what type a variable is, how to set ordered vs unordered categoricals, how many levels a variable has, whether a numeric column is a measurement or a label, or how to declare variables before describing or plotting them.
---

# Variable Types and Encoding (Python)

## Purpose
Sort each column into Dunn's two-by-two - quantitative (discrete/continuous) or qualitative
(nominal/ordinal) - then encode it: numeric dtypes, or ordered/unordered `pandas` Categorical (Dunn p161-166).

## Inputs
- `df`: one row per unit of analysis, one column per variable; the variable is what varies, the data are
  the values recorded on the individuals (Dunn p161-162).
- `kind`: what the instrument does - `"counted"`, `"measured"`, or `"label"`; `levels` (qualitative only)
  are the names of the distinct categories and `ordered` says whether they have a natural order.
- `DECIMALS`: recording precision, which never changes the class (Dunn p163).

## Algorithms (all pure numpy/scipy)

### Classify by measurement process (Definitions 11.1-11.7, Dunn p162-164)
```python
import numpy as np, pandas as pd
rng = np.random.default_rng(42)

def classify(kind, ordered=False):
    if kind == "label":                                  # numbers with no numerical meaning
        return ("qualitative", "ordinal" if ordered else "nominal")
    return ("quantitative", "discrete" if kind == "counted" else "continuous")

def naive_from_values(col):                              # what a dtype-only check would answer
    v = pd.to_numeric(pd.Series(col), errors="raise").to_numpy()
    return ("quantitative", "discrete" if (v % 1 == 0).all() else "continuous")

heart = rng.poisson(0.35, 60)                  # Ex 11.3 heart attacks: 0, 1, 2, ...
crack = rng.binomial(12, 0.08, 24)             # Ex 11.3 cracked eggs in a carton of 12: 0..12
w6 = np.round(rng.normal(20.4, 3.1, 40), 0)    # Ex 11.4 6-year-old weight, nearest kg
code = np.array([4510, 4555, 4556, 4556, 4510])  # Ex 11.2/11.6 postcodes are labels
print("RESULT heart=%s cracked_eggs=%s weight=%s postcode=%s"
      % (classify("counted"), classify("counted"), classify("measured"), classify("label")))
print("RESULT weight_naive=%s postcode_naive=%s postcode_spacing=%d heart_range=[%d,%d] eggs=[%d,%d]"
      % (naive_from_values(w6), naive_from_values(code), int(np.diff(np.unique(code)).min()),
         int(heart.min()), int(heart.max()), int(crack.min()), int(crack.max())))
```
Integer decimals are not the discrete test: recorded to the nearest kg, weight still has values between
any two weights when measured more finely, so it stays continuous (Dunn p163). Postcodes pass every numeric
test, yet 4556 is not one "more" than 4555 - being numbers does not make data quantitative (Dunn p162, p164).

### Encode qualitative variables with declared levels (Dunn p163-165)
```python
import pandas as pd
blood = ["A", "O", "B", "AB", "O", "A"]                         # Think 11.1: nominal
agree = ["Agree", "Strongly disagree", "Neither", "Disagree"]    # Ex 11.8: ordinal Likert
brand = ["Huawei", "Apple", "Samsung", "Google", "Other"]        # Ex 11.5: five levels
trans = ["Car", "Bus", "Ride bicycle or walk", "Other"]          # Ex 11.7: four levels, nominal
LIKERT = ["Strongly disagree", "Disagree", "Neither", "Agree", "Strongly agree"]
AGEGR = pd.CategoricalDtype(["Under 20", "20 to under 30", "30 to under 50", "50 or over"],
                            ordered=True)                       # Think 11.1 / Ex 11.9
c_blood = pd.Series(blood).astype(pd.CategoricalDtype(["A", "B", "AB", "O"]))
c_agree = pd.Series(agree).astype(pd.CategoricalDtype(LIKERT, ordered=True))
c_age = pd.Series(["50 or over", "Under 20"]).astype(AGEGR)
print("RESULT blood_levels=%d blood_ordered=%s likert_levels=%d likert_ordered=%s"
      % (c_blood.cat.categories.size, c_blood.cat.ordered,
         c_agree.cat.categories.size, c_agree.cat.ordered))
print("RESULT brand_levels=%d travel_levels=%d agegroup_levels=%d sorted_by_rank=%s"
      % (len(set(brand)), len(set(trans)), AGEGR.categories.size, list(c_age.sort_values())))
```
Four blood-type, five Likert, five phone-brand, four travel and four age-group levels (Dunn p163-165).
Nominal levels may go in any useful order; ordinal levels keep their natural order, never alphabetical
(Dunn p164-165).

### The declared frame - orthoses, 15 subjects (Ex 11.4, Table 11.2, p169)
```python
import pandas as pd
df = pd.DataFrame({
    "gender": ["M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "F", "M", "M", "F", "F"],
    "age_yr": [9, 7, 7, 12, 11, 5, 6, 8, 8, 6, 7, 11, 7, 9, 8],
    "height_cm": [136, 106, 129, 152, 146, 113, 112, 112, 138, 116, 113, 141, 136, 128, 133],
    "weight_kg": [34.5, 16.2, 21.1, 40.4, 39.3, 18.1, 16.7, 19.1, 28.6, 19.3, 17.6, 34.9, 34.5, 21.9, 23.0],
    "gmfcs": [1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1]})
enc = {"gender": pd.CategoricalDtype(["F", "M"]),           # nominal, 2 levels
       "gmfcs": pd.CategoricalDtype([1, 2], ordered=True),  # ordinal: lower = better function
       "age_yr": "Int64", "height_cm": "Int64", "weight_kg": "Float64"}
for col, d in enc.items():
    df[col] = df[col].astype(d)
print("RESULT n=%d gender_levels=%d gender_counts=%s gmfcs_ordered=%s gmfcs_counts=%s"
      % (len(df), df["gender"].cat.categories.size, df["gender"].value_counts().to_dict(),
         df["gmfcs"].cat.ordered, df["gmfcs"].value_counts().sort_index().to_dict()))
print("RESULT age_dtype=%s height_dtype=%s weight_dtype=%s"
      % (df["age_yr"].dtype, df["height_cm"].dtype, df["weight_kg"].dtype))
```
GMFCS codes carry rank, not arithmetic, so an ordinal variable must not be averaged as if it were a
measurement; gender is nominal with two levels (Dunn p163-164, p169).

### Software declaration map (Table 11.1, jamovi and SPSS, Dunn p166-167)
```python
import pandas as pd
MAP = {("qualitative", "nominal"): ("Nominal", "Nominal", "CategoricalDtype unordered"),
       ("qualitative", "ordinal"): ("Ordinal", "Ordinal", "CategoricalDtype ordered"),
       ("quantitative", "discrete"): ("Continuous, Integer", "Scale", "Int64"),
       ("quantitative", "continuous"): ("Continuous, Decimal", "Scale", "Float64")}
rows = [("qualitative", "nominal"), ("qualitative", "ordinal"),
        ("quantitative", "discrete"), ("quantitative", "continuous")]
t = pd.DataFrame([r + MAP[r] for r in rows],
                 columns=["class", "sub", "jamovi", "SPSS", "pandas_dtype"])
print("RESULT map_rows=%d spss_scale_declarations=%d discrete_jamovi_label=%s"
      % (len(t), int((t["SPSS"] == "Scale").sum()), MAP[("quantitative", "discrete")][0]))
print(t.to_string(index=False))
```
jamovi splits the qualitative pair but names both quantitative pairs *Continuous* (integer vs decimal,
confusingly), and SPSS collapses every quantitative variable into one *Scale* type (Dunn p166-167), so a
count must be re-declared by hand before summarising.

### Summaries are not variables (Ex 11.7, Table 11.3, p170)
```python
import pandas as pd
counts = pd.Series({"Nature": 48, "Train": 22, "Buildings": 17, "Road": 12,
                    "Dams": 7, "Fields": 4, "Others": 1})
total = int(counts.sum())
location = pd.Series(counts.index, dtype="category")   # the only variable in the table
pct = (counts / total * 100).round(1)                   # a summary of it, not a variable
print("RESULT location_levels=%d total=%d pct_largest=%s pct_smallest=%s"
      % (location.cat.categories.size, total, pct.max(), pct.min()))
```
Only *location* is measured on each person; the number and the percentage are summaries of that one
qualitative nominal variable (largest location 43.2%), and encoding them as columns would double-count
the rows (Dunn p161-162, p170).

## Output contract
Per column return `variable`, `data_class` (`quantitative` | `qualitative`), `sub_type` (`discrete` |
`continuous` | `nominal` | `ordinal`), `pandas_dtype` (`Int64` | `Float64` | `CategoricalDtype`), `levels`,
`n_levels`, `ordered`, `software` = {jamovi, SPSS}, plus the re-encoded frame; print one `RESULT` line per
quantity.

## Validity conditions
- The class follows the measurement process, not the stored values: numbers can be qualitative data
  (Dunn p162, p164).
- Discrete = a countable number of possible values between any two given values; continuous = another
  value always exists between any two when measured to more decimal places, and all continuous data are
  rounded, so recording decimals never prove discreteness (Dunn p162-163).
- One research question can be collected two ways: pasta *weight* (quantitative continuous) vs each bag
  *at least 500g or not* (qualitative, 2 levels) - declare the one the RQ asks for (Dunn p165).
- Nominal vs ordinal is a claim about the levels' order and cannot be inferred from the values; `age`
  (continuous) and `age group` (ordinal, 4 levels) are different variables (Dunn p164-165).

## Conventions
- Qualitative -> `pd.CategoricalDtype`, `ordered=True` only for ordinal variables with levels in natural
  (or exactly reverse) order. Quantitative discrete -> `Int64`, continuous -> `Float64`.
- Never `pd.get_dummies` at this stage: dummy columns erase the nominal/ordinal distinction and inflate
  the variable count; one-hot encoding is a modelling step owned elsewhere.
- Demo counts come from `numpy.random.default_rng(42)` only; Table 11.2 and 11.3 values are typed inline.
  Keep the source's software words in the map (*Nominal*, *Ordinal*, *Continuous Decimal/Integer*, *Scale*).
- Source: Dunn, *Scientific Research and Methodology* (Springer, 2024) Ch.11, pp.161-170.

## Out of scope
Summarising the values once typed - proportions, percentages, odds ratios, means, sd, IQR - belongs to
the numerical-summary skills, and choosing or rendering the graph for a variable combination belongs to
the graphical-summary skill; this skill stops at a correctly classified and encoded frame.
