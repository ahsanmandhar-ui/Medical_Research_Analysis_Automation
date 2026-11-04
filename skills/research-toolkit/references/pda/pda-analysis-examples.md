---
name: pda-analysis-examples
description: End-to-end worked pandas pipelines in the shape of McKinney's real-world analysis examples (URL-click event log, three-table movie ratings, yearly baby-name files, nested food JSON, campaign-contribution CSV), pure Python (numpy/pandas/matplotlib/seaborn). Each block assembles a frame, cleans labels, aggregates with groupby/pivot_table, applies a support floor, ranks, normalises within group and plots headless. Use when asked for a complete "real dataset" analysis example, a raw-records-to-ranked-board recipe, a groupby+pivot board with within-group shares, a multi-label explode analysis, or how to chain these steps end to end.
---

# End-to-End pandas Analysis Examples (Python)

## Purpose
Run the whole worked-example recipe - assemble, clean labels, aggregate, apply a support floor, rank,
normalise within group, plot - on five real dataset shapes (McKinney p443-444).

## Inputs
- **event log** (Bitly/USA.gov): 18 cols x 3560 rows (`a, c, nk, tz, gr, g, h, l, al, hh, r, u, t, hc, cy,
  ll, _heartbeat_, kw`), `tz` non-null 3440, blank = unknown (McKinney p447).
- **three rating tables** (MovieLens): `ratings(user_id, movie_id, rating, timestamp)`, `users(user_id,
  gender, age, occupation, zip)`, `movies(movie_id, title, genres)` -> 1000209 x 10 merged (McKinney p455-457).
- **yearly name files** (SSA): `name, sex, births` per year 1880-2010 -> 1690784 x 4 after concat, top 1000
  names per sex-year for the trend work (McKinney p444-447, p463).
- **nested food JSON** (USDA): 6636 records with a `nutrients` list -> 389355 long rows, 375176 after dedupe.
- **contribution CSV** (FEC 2012): 1001731 x 16 with `cand_nm, contbr_occupation, contbr_st, contb_receipt_amt` (McKinney p463).
  The book's files are not shipped: every frame below is synthesised with `numpy.random.default_rng(42)` in the same column shape.
- Constants: `SEED = 42`, `TOP_N = 10`, `MIN_RATINGS = 250` (McKinney p438), `MIN_TOTAL = 2_000_000` (p467),
  `MIN_STATE = 100_000` (p471), `BINS = [0, 1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7]` (p469).

## Workflow
One DataFrame per step, none skipped: raw records -> tidy long frame -> labelled frame -> aggregate board -> support-filtered ranking -> normalised shares -> headless figure (McKinney p443-444).

## Algorithms (all pure numpy/scipy)

### 1. Event log: value_counts, a two-way count board, within-group shares (McKinney p445-453)
```python
import numpy as np, pandas as pd
rng, TOP_N = np.random.default_rng(42), 10
tzs = np.array(["America/New_York", "America/Chicago", "America/Los_Angeles", "America/Denver", "Europe/London", "Asia/Tokyo", "Pacific/Honolulu", ""])   # "" = unknown (p445)
w = np.array([1251, 400, 382, 191, 74, 37, 36, 521], float); w = w / w.sum()    # source tz counts (p446)
frame = pd.DataFrame({"tz": rng.choice(tzs, 3560, p=w), "a": rng.choice(["Mozilla/5.0 (Windows NT 6.1)", "Mozilla/5.0 (X11; Linux)", "GoogleMaps/RochesterNY"], 3560, p=[0.7, 0.2, 0.1])})
frame.loc[rng.choice(3560, 120, replace=False), "tz"] = np.nan                  # 3440 non-null, source shape
for col in "c nk gr g h l al hh r u t hc cy ll _heartbeat_ kw".split(): frame[col] = "x"   # 18 cols (p447)
clean = frame["tz"].fillna("Missing"); clean = clean.mask(clean == "", "Unknown"); counts = clean.value_counts()   # blank != NaN (p448)
win = frame.loc[frame["a"].notna()].copy(); win["os"] = np.where(win["a"].str.contains("Windows"), "Windows", "Not Windows")   # drop rows with no agent (p449)
win["tz"] = clean.reindex(win.index)
board = win.groupby(["tz", "os"]).size().unstack(fill_value=0)                  # 2-way counts (p450)
long = board.loc[board.sum(axis="columns").nlargest(TOP_N).index].stack().rename("total").reset_index()   # nlargest beats argsort+take (p451)
long["normed_total"] = long["total"] / long.groupby("tz")["total"].transform("sum")   # shares, not raw counts (p453)
import matplotlib; matplotlib.use("Agg"); import os, tempfile, matplotlib.pyplot as plt, seaborn as sns
ax = sns.barplot(x="normed_total", y="tz", hue="os", data=long)               # tidy frame, Figure 13-3 analogue
png = os.path.join(tempfile.mkdtemp(prefix="flat1_"), "tz_by_os.png"); ax.get_figure().savefig(png); plt.close("all")
print(f"RESULT rows={len(frame)} cols={frame.shape[1]} tz_nonnull={int(frame['tz'].notna().sum())} top_tz={counts.index[0]} ny_n={int(counts['America/New_York'])} unknown={int(counts['Unknown'])} missing={int(counts['Missing'])} windows_share={float(long.loc[long['os'] == 'Windows', 'normed_total'].mean()):.4f} figure={os.path.exists(png)}")
```
`rows=3560 cols=18 tz_nonnull=3440` reproduces the source frame shape; `windows_share` is the synthetic stand-in for Figure 13-3's percentage bars.

### 2. Three-table join -> pivot board -> support floor -> divisive rankings -> explode multi-label tags (McKinney p455-461)
```python
import numpy as np, pandas as pd
rng = np.random.default_rng(42); MIN_RATINGS, U, M, R = 250, 600, 150, 60_000
users = pd.DataFrame({"user_id": np.arange(1, U + 1), "gender": rng.choice(list("FM"), U, p=[0.44, 0.56]), "age": rng.choice([1, 18, 25, 35, 45, 50, 56], U), "occupation": rng.integers(1, 21, U), "zip": rng.integers(10_000, 99_999, U)})
movies = pd.DataFrame({"movie_id": np.arange(1, M + 1), "title": [f"Film {i} (199{i % 10})" for i in range(1, M + 1)], "genres": rng.choice(["Drama", "Comedy|Romance", "Action|Adventure|Thriller", "Animation|Children's|Comedy"], M)})
ratings = pd.DataFrame({"user_id": rng.integers(1, U + 1, R), "rating": rng.integers(1, 6, R), "movie_id": rng.pareto(0.9, R).astype(int).clip(0, M - 1) + 1, "timestamp": rng.integers(956_700_000, 979_000_000, R)})   # skewed audience
data = pd.merge(pd.merge(ratings, users), movies)               # overlapping names are the join keys (p455)
mean_ratings = data.pivot_table("rating", index="title", columns="gender", aggfunc="mean")
support = data.groupby("title").size(); active = support.index[support >= MIN_RATINGS]   # support floor (p438)
mr = mean_ratings.loc[active].join(data.groupby("title")["rating"].std().loc[active].rename("std"))
mr["diff"] = mr["M"] - mr["F"]                                  # signed gap (p439); std ranks disagreement (p440)
mg = movies.assign(genre=movies["genres"].str.split("|")).explode("genre")        # one row per genre tag (p459)
genre_age = pd.merge(pd.merge(mg, ratings), users).groupby(["genre", "age"])["rating"].mean().unstack("age")
print(f"RESULT merged={data.shape[0]}x{data.shape[1]} titles={len(support)} active={len(active)} liked_by_F='{mr['diff'].idxmin()}' liked_by_M='{mr['diff'].idxmax()}' most_divisive='{mr['std'].idxmax()}' genre_board={genre_age.shape[0]}x{genre_age.shape[1]}")
```

### 3. Yearly files -> concat -> per-group proportion -> diversity quantile count -> figure (McKinney p444-456)
```python
import numpy as np, pandas as pd
rng = np.random.default_rng(42); parts = []
for year in range(1880, 2011):                            # one text file per year -> concat (p445)
    for sex, flat in (("F", 0.55), ("M", 0.40)):          # girl names flatter, both flatten over time (p452)
        t = (year - 1880) / 130; b = 40_000 * (1 - 0.55 * t) * (np.arange(250) + 1) ** (-(0.85 - flat * t))
        parts.append(pd.DataFrame({"name": [f"{sex}{i}" for i in range(250)], "sex": sex, "year": year, "births": np.maximum(5, b * rng.uniform(0.9, 1.1, 250)).astype(int)}))
names = pd.concat(parts, ignore_index=True)                 # source row ids carry no meaning
total_births = names.pivot_table("births", index="year", columns="sex", aggfunc="sum")    # Figure 13-4 step
names["prop"] = names["births"] / names.groupby(["year", "sex"])["births"].transform("sum")
sanity = float((names.groupby(["year", "sex"])["prop"].sum() - 1).abs().max())   # every group must be 1 (p447)
mary_1880 = 7065 / 90993                                   # source: Mary among 90993 female births (p447)
def quantile_count(g, q=0.5):                              # cumsum + searchsorted, no python loop (p451)
    return g["prop"].sort_values(ascending=False).cumsum().searchsorted(q) + 1
diversity = names.groupby(["year", "sex"]).apply(quantile_count).unstack("sex")
within_year = total_births.div(total_births.sum(axis="columns"), axis="index")   # row-wise shares (p456)
import matplotlib; matplotlib.use("Agg"); import os, tempfile, matplotlib.pyplot as plt
ax = diversity.plot(title="Names needed for the top 50% of births")              # Figure 13-7 analogue
png = os.path.join(tempfile.mkdtemp(prefix="flat3_"), "diversity.png"); ax.get_figure().savefig(png); plt.close("all")
print(f"RESULT rows={len(names)} prop_sanity={sanity:.1e} mary_prop={mary_1880:.6f} div_1880_M={int(diversity['M'].iloc[0])} div_2010_M={int(diversity['M'].iloc[-1])} div_2010_F={int(diversity['F'].iloc[-1])} f_share_2010={float(within_year['F'].iloc[-1]):.4f} figure={os.path.exists(png)}")
```
`mary_prop=0.077643` is the book's own 1880 proportion for Mary; the diversity count rising with the years is the source finding (p451-452).

### 4. Nested JSON -> stacked long frame -> dedupe -> merge -> group median and per-nutrient idxmax (McKinney p457-462)
```python
import numpy as np, pandas as pd
rng, db = np.random.default_rng(42), []
spec = [("Protein", "g", "Composition"), ("Total lipid (fat)", "g", "Composition"), ("Energy", "kcal", "Energy"), ("Zinc, Zn", "mg", "Elements"), ("Sodium", "mg", "Elements"), ("Glycine", "g", "Amino Acids"), ("Alanine", "g", "Amino Acids")]
grp = ["Dairy and Egg Products", "Vegetables and Vegetable Products", "Fast Foods", "Baked Products"]
for i in range(1200):                                      # 6636 foods in the source file (p457)
    ns = [{"value": round(float(rng.gamma(1.2, 8)), 3), "units": u, "description": d, "group": g} for d, u, g in spec]
    db.append({"id": 1000 + i, "description": f"Food {1000 + i}", "group": str(rng.choice(grp)), "manufacturer": str(rng.choice(["Acme Co", ""])), "nutrients": ns + ns[:2]})   # injected duplicates
nutrients = pd.concat([pd.DataFrame(r["nutrients"]).assign(id=r["id"]) for r in db], ignore_index=True)
dupes = int(nutrients.duplicated().sum()); nutrients = nutrients.drop_duplicates()    # dedupe before joining
info = pd.DataFrame(db, columns=["description", "group", "id", "manufacturer"]).rename(columns={"description": "food", "group": "fgroup"})
ndata = pd.merge(nutrients.rename(columns={"description": "nutrient", "group": "nutgroup"}), info, on="id")
zinc = ndata.groupby(["nutrient", "fgroup"])["value"].quantile(0.5).loc["Zinc, Zn"].sort_values()   # median per group
richest = ndata.groupby(["nutgroup", "nutrient"]).apply(lambda x: x.loc[x["value"].idxmax()])[["value", "food"]]
assert 389_355 - 14_179 == 375_176                                              # the source's own row identity
print(f"RESULT long_rows={len(nutrients) + dupes} dupes={dupes} deduped={len(nutrients)} merged={len(ndata)}x{ndata.shape[1]} foods={len(info)} manuf_blank={int((info['manufacturer'] == '').sum())} zinc_leader='{zinc.index[-1]}' amino_leader='{richest.loc['Amino Acids']['food'].iloc[0]}'")
```

### 5. Contributions: label map -> party pivot -> cut bins -> normalised buckets -> state shares (McKinney p463-471)
```python
import numpy as np, pandas as pd
rng = np.random.default_rng(42); N, MIN_TOTAL, MIN_STATE = 60_000, 2_000_000, 100_000
cand = rng.choice(["Obama, Barack", "Romney, Mitt"], N, p=[0.55, 0.45]); levels = np.array([0.0, 5.0, 25.0, 250.0, 2_500.0, 25_000.0, -50.0])   # negatives are refunds
amt = np.where(cand == "Obama, Barack", rng.choice(levels, N, p=[0.02, 0.30, 0.38, 0.17, 0.09, 0.02, 0.02]), rng.choice(levels, N, p=[0.02, 0.18, 0.31, 0.26, 0.15, 0.06, 0.02]))
fec = pd.DataFrame({"cand_nm": cand, "contbr_st": rng.choice(["CA", "NY", "TX", "FL", "AZ", "CT", "DC"], N), "contbr_occupation": rng.choice(["RETIRED", "ATTORNEY", "INFORMATION REQUESTED", "INFORMATION REQUESTED PER BEST EFFORTS", "C.E.O.", "TEACHER"], N, p=[0.24, 0.16, 0.18, 0.14, 0.08, 0.20]), "contb_receipt_amt": np.round(amt * rng.uniform(0.8, 1.6, N), 2), "form_type": "SA17A"})
fec["party"] = fec["cand_nm"].map({"Obama, Barack": "Democrat", "Romney, Mitt": "Republican"})   # party absent from the raw CSV (p465)
signs = (fec["contb_receipt_amt"] > 0).value_counts(); fec = fec.loc[fec["contb_receipt_amt"] > 0].copy()   # drop refunds (p466)
occ = {"INFORMATION REQUESTED": "NOT PROVIDED", "INFORMATION REQUESTED PER BEST EFFORTS": "NOT PROVIDED", "C.E.O.": "CEO"}
fec["contbr_occupation"] = fec["contbr_occupation"].map(lambda x: occ.get(x, x))    # .get lets other labels pass through
by_occ = fec.pivot_table("contb_receipt_amt", index="contbr_occupation", columns="party", aggfunc="sum")
over_2mm = by_occ[by_occ.sum(axis="columns") > MIN_TOTAL]                           # support floor (p467)
mrbo = fec.loc[fec["cand_nm"].isin(["Obama, Barack", "Romney, Mitt"])]
labels = pd.cut(mrbo["contb_receipt_amt"], np.array([0, 1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]))   # right-closed buckets (p469)
g = mrbo.groupby(["cand_nm", labels], observed=True); buckets = g.size().unstack(level=0)
normed = g["contb_receipt_amt"].sum().unstack(level=0); normed = normed.div(normed.sum(axis="columns"), axis="index")[:-2]   # top bins are not individuals (p472)
totals = mrbo.groupby(["cand_nm", "contbr_st"])["contb_receipt_amt"].sum().unstack(level=0).fillna(0)
percent = totals[totals.sum(axis="columns") > MIN_STATE].div(totals.sum(axis="columns"), axis="index")
small = pd.Interval(1, 10, closed="right"); assert 1_001_731 - 10_256 == 991_475 and 593_746 + 407_985 == 1_001_731   # source row identities (p463-465)
print(f"RESULT rows={N} pos={int(signs[True])} neg={int(signs[False])} over_2mm={len(over_2mm)} categories={len(labels.cat.categories)} obama_small={int(buckets.loc[small, 'Obama, Barack'])} romney_small={int(buckets.loc[small, 'Romney, Mitt'])} obama_ca_share={float(percent.loc['CA', 'Obama, Barack']):.4f} states={len(percent)}")
```

## Output contract
Each block prints one `RESULT name=value` line and leaves its objects in scope: `counts` (tz -> n) with
`board` (`tz` x `os`) and tidy `long` (`tz, os, total, normed_total`); `mr` (`title` x `F, M, std, diff`),
`active` (Index), `genre_age` (`genre` x `age`); `total_births`, `diversity`, `within_year` (`year` x `F, M`);
`ndata` (8 cols) with `zinc` (food group -> median) and `richest`; `over_2mm`, `buckets`, `normed`, `percent`
(one column per candidate). Figures go only into a `tempfile` dir the block itself created.

## Validity conditions
- Verify a per-group proportion sums to 1 in every group before reading a trend from it (McKinney p465).
- Rank only above a support floor - 250 ratings, 2M donated, 100k per state - below it the order is noise (p438, p467, p471).
- Drop rows the derived label needs (non-null agent string) and label blanks explicitly - `Missing` (absent field) and `Unknown` (empty string) differ (p448-449).
- Dedupe the stacked long frame before merging or the join multiplies rows (p460-461); exclude refunds before
  summing and drop the two largest donation bins when normalising size (p466, p472).

## Conventions
- `SEED = 42` per block, `TOP_N = 10`, right-closed `BINS`; keep the source column names (`prop`,
  `normed_total`, `contb_receipt_amt`) so a result is comparable with the book's tables.
- Prefer `transform("sum")` over `groupby.apply` for a per-group divisor (p453), `nlargest` over `argsort` +
  `take` for top-N (p451); seaborn's `barplot(x=, y=, hue=, data=long)` takes the tidy frame, not the board.
- Headless only: `matplotlib.use("Agg")`, `savefig` into `tempfile`. Column-wise (`x / x.sum()`) and row-wise
  (`.div(..., axis="index")`) normalisation are different tables - never invent a value, compute it and cite the page.

## Out of scope
Split-apply-combine and `pivot_table`/`crosstab` mechanics, resample/rolling windows, merge/concat and
reshape rules, the plotting API, file IO, dedupe and missing-value policy and model fitting belong to sibling
skills; this file only chains them into worked examples.
