---
name: pda-plotting-visualization
description: Build publication figures from pandas frames with the matplotlib API and seaborn statistical graphics, pure Python (matplotlib/seaborn/pandas/numpy). Drives figures and subplot grids with subplots_adjust, colors/markers/drawstyle, ticks/labels/legends/annotations/patches, savefig with dpi, plt.rc options, the pandas .plot family (line/bar/barh/hist/density-KDE), seaborn barplot means with 95% CI error bars, histplot, regplot/pairplot scatter grids and catplot facet grids. Use when asked to plot a Series or DataFrame, make a histogram or KDE, lay out subplots/panels, add a legend or error bars, save a figure headless, or split a plot by categorical variables into a facet grid.
---

# Plotting and Visualization - pandas, matplotlib, seaborn (Python)

## Purpose
Turn a `Series`/`DataFrame` into a static publication figure: drive matplotlib directly, or let pandas' `.plot` family and seaborn (a high-level statistical graphics layer on top of matplotlib) assemble it (McKinney p299-316).

## Inputs
- A `pandas.Series`/`DataFrame` plus column-name strings for `x=`, `y=`, `hue=`, `row=`, `col=`; tidy frames suit seaborn (McKinney p325, p332).
- Axes handles: `fig, axes = plt.subplots(nrows, ncols)`; nearly every pandas/seaborn call takes `ax=` for placement in a grid (McKinney p304, p317).
- Knobs: `kind`, `bins`, `alpha`, `rot`, `grid`, `xticks/ylim`, `stacked`, `subplots/layout`, `diag_kind`, `plot_kws`, `dpi`, `figsize` (McKinney p305, p317-319, p330).

## Algorithms (all pure numpy/scipy)
### Figure, subplot grid, spacing, savefig, rc options (McKinney p301-305, p314-315)
```python
import matplotlib, tempfile, numpy as np; from PIL import Image
matplotlib.use("Agg")
import matplotlib.pyplot as plt
rng = np.random.default_rng(42); out = tempfile.mkdtemp()
fig = plt.figure(figsize=(6, 6)); ax1, ax2, ax3 = (fig.add_subplot(2, 2, k) for k in (1, 2, 3))
ax1.hist(rng.standard_normal(100), bins=20, color="black", alpha=0.3); ax2.scatter(np.arange(30), np.arange(30) + 3 * rng.standard_normal(30))
line = ax3.plot(rng.standard_normal(50).cumsum(), color="black", linestyle="dashed")[0]
grid, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(8, 6)); grid.subplots_adjust(wspace=0, hspace=0)
plt.rc("figure", figsize=(3, 2)); plt.rc("font", family="monospace", weight="bold", size=8); mini = plt.figure()
mini.savefig(f"{out}/figpath.png", dpi=400); px = Image.open(f"{out}/figpath.png").size; plt.rcdefaults()
print(f"RESULT artist={type(line).__name__} fig_axes={len(fig.axes)} grid={axes.shape} shared_y={len({tuple(a.get_ylim()) for a in axes.ravel()}) == 1} png_px={px[0]}x{px[1]} rcdefaults_figsize={tuple(plt.rcParams['figure.figsize'])}")
```
A `Figure` holds plots but a blank one takes no data: `add_subplot(2, 2, k)` is rows, columns, 1-based slot, and `plt.subplots(2, 3)` returns a 2-D axes array indexed `axes[0, 1]`, taking `nrows, ncols, sharex, sharey, subplot_kw` plus figure keywords such as `figsize=(8, 6)` (McKinney p301, p304-305). Prefer axis methods (`ax.plot/hist/scatter`) over top-level `plt.*`; they return artists such as `Line2D` you ignore or `;`-suppress, and `sharex`/`sharey` link ticks and limits, else every subplot autoscales independently (McKinney p302-304). `subplots_adjust(left, bottom, right, top, wspace, hspace)` sets outer padding and inter-subplot spacing as a fraction of figure size; `wspace=0, hspace=0` tiles panels seamlessly and any overlapping tick labels are yours to fix (McKinney p305-306). `savefig` infers format from the extension and `dpi` sets resolution (default 100 IPython, 72 Jupyter), so `figsize=(3, 2)` at `dpi=400` is 1200x800 px; options `fname, dpi, facecolor/edgecolor ("w"), format` (McKinney p314-315). `plt.rc(component, ...)` - "figure", "axes", "xtick", "ytick", "grid", "legend", "font" - writes global defaults readable in `plt.rcParams`, and `plt.rcdefaults()` restores them (McKinney p315).
### Colors, markers, drawstyle, ticks, legend, annotations, patches (McKinney p306-313)
```python
import matplotlib, tempfile, numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
rng = np.random.default_rng(42); out = tempfile.mkdtemp()
fig, (ax, ts, sh) = plt.subplots(1, 3, figsize=(11, 3)); data = rng.standard_normal(30).cumsum()
ax.plot(data, color="black", linestyle="dashed", marker="o", label="default"); ax.plot(data, linestyle="--", color="#CECECE", drawstyle="steps-post", label="steps-post")
ax.plot(data, linestyle="dotted", color="black", label="_nolegend_"); ax.legend(); ax.set(title="My first matplotlib plot", xlabel="Stages", ylabel="Cumsum")
ts.plot(np.arange(120), 1000 + np.cumsum(rng.standard_normal(120) * 20), color="black"); y = ts.get_lines()[0].get_ydata(); i = int(np.argmin(y))
ts.annotate("local low", xy=(i, y[i] + 75), xytext=(i, y[i] + 225), arrowprops=dict(facecolor="black", headwidth=4, width=2, headlength=4), horizontalalignment="left", verticalalignment="top")
ts.text(5, y.max() + 40, "Hello world!", family="monospace", fontsize=10); ts.set_xticks([0, 25, 50, 75, 100])
ts.set_xticklabels(["one", "two", "three", "four", "five"], rotation=30, fontsize=8); ts.set_xlim([0, 119]); ts.set_ylim([600, 1800])
for p in (plt.Rectangle((0.2, 0.75), 0.4, 0.15, color="black", alpha=0.3), plt.Circle((0.7, 0.2), 0.15, color="blue", alpha=0.3), plt.Polygon([[0.15, 0.15], [0.35, 0.4], [0.2, 0.6]], color="green", alpha=0.5)): sh.add_patch(p)
fig.savefig(f"{out}/styling.png", dpi=120)
print(f"RESULT legend_rows={len(ax.get_legend().get_texts())} xticklabels={[t.get_text() for t in ts.get_xticklabels()][:2]} xlim={tuple(map(float, ts.get_xlim()))} ylim={tuple(map(float, ts.get_ylim()))} patches={len(sh.patches)}")
```
`plot` takes `linestyle`, `color` (a name or hex such as `#CECECE`), `marker="o"` to expose the interpolated points and `drawstyle="steps-post"` (McKinney p306-307). `xlim/xticks/xticklabels` read the current value with no argument and set it with one; on an axes use `set_xticks`, `set_xticklabels(rotation=30, fontsize=8)`, `set_xlabel`, `set_title`, or batch them with `ax.set(title=..., xlabel=...)` (McKinney p308-310). `label=` alone is not enough - **you must call `ax.legend()`**; `loc` defaults to "best" and `label="_nolegend_"` excludes an artist (McKinney p308, p311). `annotate` draws text and an arrow between `xy` and `xytext` (the book offsets them +75/+225), `text` draws a string at `(x, y)`, `set_xlim` accepts date strings, and shapes are `Rectangle`/`Circle`/`Polygon` patches added with `ax.add_patch` - many plot types are built from patches (McKinney p312-313).
### pandas plotting: line, panel subplots, kinds, bars, normalized crosstab (McKinney p316-323)
```python
import matplotlib, tempfile, numpy as np, pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
rng = np.random.default_rng(42); out = tempfile.mkdtemp(); plt.style.use("grayscale")
s = pd.Series(rng.standard_normal(10).cumsum(), index=np.arange(0, 100, 10))
df = pd.DataFrame(rng.standard_normal((10, 4)).cumsum(0), columns=pd.Index(list("ABCD"), name="Genus"), index=np.arange(0, 100, 10))
fig1, ax_s = plt.subplots(figsize=(5, 3)); s.plot(kind="line", title="Simple Series plot", xticks=[0, 50, 90], xlim=[0, 95], ylim=[-6, 6], rot=30, grid=True, style="ko--", ax=ax_s)
fig2, ax_d = plt.subplots(figsize=(5, 3)); df.plot(ax=ax_d); panels = df.plot(subplots=True, layout=(2, 2), sharex=True, sharey=True, figsize=(5, 5))
bars = pd.DataFrame(rng.uniform(size=(6, 4)), index=["one", "two", "three", "four", "five", "six"], columns=df.columns)
party = pd.DataFrame([[2, 53, 18, 13, 1, 0], [0, 39, 15, 18, 3, 1], [1, 48, 4, 5, 1, 3], [1, 16, 1, 1, 0, 0]], index=["Sat", "Sun", "Thur", "Fri"], columns=[1, 2, 3, 4, 5, 6])
party = party.reindex(index=["Thur", "Fri", "Sat", "Sun"]).loc[:, 2:5]; pcts = party.div(party.sum(axis="columns"), axis="index")
fig3, (grp, stk, pn) = plt.subplots(1, 3, figsize=(11, 3)); bars.plot.bar(ax=grp); bars.plot.barh(stacked=True, alpha=0.5, ax=stk); pcts.plot.bar(stacked=True, ax=pn)
fig4, ax_v = plt.subplots(figsize=(4, 3)); vc = pd.Series(rng.choice(list("abcd"), 200)).value_counts(); vc.plot.bar(ax=ax_v, color="black", alpha=0.7); fig3.savefig(f"{out}/party_pcts.png", dpi=150)
kinds = [k for k in ("area", "bar", "barh", "density", "hist", "kde", "line", "pie") if callable(getattr(s.plot, k, None))]
print(pcts.round(6).to_string()); print(f"RESULT thur_size2={pcts.loc['Thur', 2]:.6f} sun_size4={pcts.loc['Sun', 4]:.6f} rows_sum_1={np.allclose(pcts.sum(axis='columns'), 1.0)} xticks={list(map(int, ax_s.get_xticks()))} grid={ax_s.get_xaxis().get_gridlines()[0].get_visible()} df_lines={len(ax_d.get_lines())} auto_legend={ax_d.get_legend() is not None} panel_shape={panels.shape} kinds={kinds} legend_title={grp.get_legend().get_title().get_text()!r} grouped={sum(len(c.patches) for c in grp.containers)} stacked={sum(len(c.patches) for c in stk.containers)} vc_bars={len(ax_v.patches)}")
```
`Series.plot` sends its index to the x-axis (`use_index=False` stops that) and takes `label, ax, style` (e.g. `"ko--"`), `alpha`, `figsize`, `logx/logy` (`"sym"` permits negatives), `title`, `rot` 0-360, `xticks`, `yticks`, `xlim`, `ylim`, `grid` (off by default); `kind` is `area, bar, barh, density, hist, kde, line, pie`, default `line` (McKinney p316-317). `DataFrame.plot` draws one line per column and adds the legend itself, `df.plot()` equals `df.plot.line()`, and DataFrame-only options are `subplots`, `layout=(rows, cols)`, `sharex`, `sharey`, `legend`, `sort_columns` (McKinney p318-319); spare keywords pass through to matplotlib and `plt.style.use("grayscale")` suits black-and-white print (McKinney p318). `plot.bar`/`plot.barh` use the index as tick labels, group each row's columns side by side and stack them with `stacked=True`; the columns index *name* titles the legend (`Genus`) (McKinney p319-322); `s.value_counts().plot.bar()` is the category-frequency recipe (McKinney p322). `party` is `pd.crosstab(tips["day"], tips["size"])` exactly as printed on p323; after reordering and dropping sizes 1 and 6, `div(party.sum(axis="columns"), axis="index")` normalizes each row to 1 and reproduces the book (McKinney p323-324):
Thur 0.827586 / 0.068966 / 0.086207 / 0.017241, Fri 0.888889 / 0.055556 / 0.055556 / 0.000000, Sat 0.623529 / 0.211765 / 0.152941 / 0.011765, Sun 0.520000 / 0.200000 / 0.240000 / 0.040000 - party sizes grow at the weekend.
### seaborn aggregates: mean bars with 95% CI, histogram, KDE (McKinney p324-328)
```python
import matplotlib, tempfile, numpy as np, pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
rng = np.random.default_rng(42); out = tempfile.mkdtemp()
tip_pct = [t / (b - t) for b, t in ((16.99, 1.01), (10.34, 1.66), (21.01, 3.50), (23.68, 3.31), (24.59, 3.61))]
tips = pd.DataFrame({"day": rng.choice(["Thur", "Fri", "Sat", "Sun"], 200), "time": rng.choice(["Lunch", "Dinner"], 200), "total_bill": rng.uniform(5, 50, 200)})
tips["tip"] = (tips["total_bill"] * rng.uniform(0.10, 0.25, 200)).round(2); tips["tip_pct"] = tips["tip"] / (tips["total_bill"] - tips["tip"])
sns.set_style("whitegrid"); sns.set_palette("Greys_r")
fig1, b1 = plt.subplots(figsize=(5, 3)); sns.barplot(x="tip_pct", y="day", data=tips, orient="h", ax=b1)
fig2, b2 = plt.subplots(figsize=(5, 3)); sns.barplot(x="tip_pct", y="day", hue="time", data=tips, orient="h", ax=b2)
fig3, (hx, kd) = plt.subplots(1, 2, figsize=(8, 3)); tips["tip_pct"].plot.hist(bins=50, ax=hx); tips["tip_pct"].plot.density(ax=kd)
comp = pd.Series(np.concatenate([rng.standard_normal(200), 10 + 2 * rng.standard_normal(200)]))
fig4, hp = plt.subplots(figsize=(5, 3)); sns.histplot(comp, bins=100, color="black", ax=hp); fig4.savefig(f"{out}/hist_kde.png", dpi=150)
means = tips.groupby("day")["tip_pct"].mean()
print(f"RESULT head_pct={[f'{v:.6f}' for v in tip_pct]} bars={len(b1.patches)} ci_lines={len(b1.lines)} bar_is_mean={np.allclose(sorted(p.get_width() for p in b1.patches), sorted(means))} hue_bars={sum(len(c.patches) for c in b2.containers)} hist_bins={len(hx.patches)} kde_points={len(kd.lines[0].get_xdata())} sns_hist_bins={len(hp.containers[0].patches)} mixture_means={comp[comp < 5].mean():.2f},{comp[comp >= 5].mean():.2f}")
```
Seaborn calls take `data=` plus column names, and `tip_pct = tip / (total_bill - tip)`: on the book's first five rows that is 0.063204, 0.191244, 0.199886, 0.162494, 0.172069 (McKinney p324). A histogram is a discretized value-frequency count over evenly spaced bins (`plot.hist(bins=50)`); a density plot is a kernel density estimate - a mixture-of-normals approximation of the generating distribution, via `plot.density()`, and it needs SciPy (McKinney p327-328). `sns.histplot` draws both at once; the book's example is bimodal, 200 draws of `N(0,1)` plus 200 of `10 + 2*N(0,1)` at `bins=100` (McKinney p328). With several observations per day, `barplot` bars are the **mean** `tip_pct` and the black line is the 95% confidence interval (configurable), hence the `bar_is_mean` check (McKinney p325); `hue=` splits each bar by one more category (McKinney p326). Seaborn replaces matplotlib's palette, background and grid defaults - switch them with `set_style("whitegrid")` and `set_palette("Greys_r")` for greyscale print (McKinney p326).
### Scatter, pairs plots and facet grids (McKinney p329-334)
```python
import matplotlib, tempfile, numpy as np, pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
rng = np.random.default_rng(42); out = tempfile.mkdtemp()
levels = pd.DataFrame(np.exp(np.cumsum(rng.standard_normal((201, 3)) * 0.02, axis=0)) * [200, 8, 5], columns=["cpi", "m1", "tbilrate"])
trans = np.log(levels).diff().dropna(); slope = float(np.polyfit(trans["m1"], trans["tbilrate"], 1)[0])
fig1, rg = plt.subplots(figsize=(4, 3)); sns.regplot(x="m1", y="tbilrate", data=trans, ax=rg)
drawn = float(np.polyfit(rg.lines[0].get_xdata(), rg.lines[0].get_ydata(), 1)[0])
pg = sns.pairplot(trans, diag_kind="kde", plot_kws={"alpha": 0.2})
tips = pd.DataFrame({"day": rng.choice(["Thur", "Fri", "Sat", "Sun"], 120), "time": rng.choice(["Lunch", "Dinner"], 120), "smoker": rng.choice(["Yes", "No"], 120), "tip_pct": rng.uniform(0.05, 0.35, 120)})
g1 = sns.catplot(x="day", y="tip_pct", hue="time", col="smoker", kind="bar", data=tips[tips.tip_pct < 1])
g2 = sns.catplot(x="day", y="tip_pct", row="time", col="smoker", kind="bar", data=tips[tips.tip_pct < 1])
g3 = sns.catplot(x="tip_pct", y="day", kind="box", data=tips[tips.tip_pct < 0.5]); g3.savefig(f"{out}/cat_box.png", dpi=120); plt.close("all")
print(f"RESULT rows={len(trans)} logdiff_err={np.max(np.abs(trans.to_numpy() - np.log(levels.iloc[1:].to_numpy() / levels.iloc[:-1].to_numpy()))):.1e} reg_slope={slope:.4f} drawn_slope_err={abs(drawn - slope):.1e} pair_shape={pg.axes.shape} facets_hue={g1.axes.shape} facets_row_col={g2.axes.shape} box_facets={g3.axes.shape}")
```
Scatter plots examine two 1-D series; the book reads `macrodata.csv`, keeps `cpi, m1, tbilrate, unemp` and takes log differences with
`np.log(data).diff().dropna()` (201 levels give 200 differenced rows) (McKinney p329). `sns.regplot` scatters and fits a line; `sns.pairplot` is the
pairs/scatter matrix with `diag_kind="kde"` on the diagonal and `plot_kws` handed to the off-diagonal calls (McKinney p330-331). A facet grid is a
2-D layout of plots whose data is split by the distinct values of a variable: `sns.catplot` colors groups inside one facet with `hue=`, adds a
panel per level with `row=`/`col=`, and picks the mark with `kind="bar"` or `kind="box"` (median, quartiles, outliers) (McKinney p332-334);
`sns.FacetGrid` is the general class for hand-built facets (McKinney p334).

## Output contract
Each block prints one `RESULT name=value ...` line (and, for the crosstab, the normalized table) and writes PNGs only into its own
`tempfile.mkdtemp()` dir (`figpath.png`, `party_pcts.png`, `hist_kde.png`, `cat_box.png`). Returns: a `Figure` and an `Axes` array from
`plt.subplots` (index `axes[i, j]`), an `Axes` from every `df.plot.*` and seaborn axes-level call (pass it back as `ax=`), and a
`PairGrid`/`FacetGrid` from `pairplot`/`catplot` - call `.savefig(path, dpi=)` on those; format is inferred from the extension (McKinney p304, p314-315).

## Workflow
1. Pick the highest-level call that fits the data shape (pandas `.plot.<kind>`, then seaborn) and drop to the matplotlib API only for options it does not expose (McKinney p301, p316).
2. Build the frame, plot on each `ax`, then `subplots_adjust`, `set`/`legend`/`annotate`, then `savefig(..., dpi=400)`; `plt.close("all")` when done (McKinney p304-305, p310, p314).

## Validity conditions
- A figure needs at least one subplot before it can hold data (McKinney p301).
- `sharex`/`sharey` are only correct when the series share a scale, and `wspace=0, hspace=0` can overlap tick labels - set explicit ticks and labels
  (McKinney p304, p306).
- KDE/`plot.density` requires SciPy; a histogram's shape is a function of `bins`, so report the bin count (McKinney p327-328).
- Seaborn bars are group means with a default 95% CI line, so each group needs more than one observation or the error bar is meaningless (McKinney
  p325); filter impossible values before faceting (`tip_pct < 1`, then `< 0.5`) (McKinney p332-334).

## Reporting conventions
One `RESULT` line per figure (panel counts, tick labels, legend rows, bins, pixel size); caption below the figure, units on every axis label, and say
what the mark encodes - "bars are mean tip_pct, black line = 95% CI" (McKinney p310, p325).

## Out of scope
Choosing *which* graph suits a variable pair and reading it (graphical-summaries skill); the inference behind an error bar (confidence-interval
skills); time-series plotting and resampling (McKinney p319); interactive web graphics - Altair, Bokeh, Plotly - because matplotlib builds static
figures for print or web (McKinney p335).

## Conventions
- Always headless: `matplotlib.use("Agg")` before importing pyplot, output only to a `tempfile.mkdtemp()` dir, `plt.close("all")` at the end;
  import convention `import matplotlib.pyplot as plt` (McKinney p300).
- Pass an explicit `ax=` (or a fresh figure) to every call so a leftover current axes cannot swallow the plot.
- Demo data is seeded with `numpy.random.default_rng(42)`; the book's example files (`tips.csv`, `macrodata.csv`, `spx.csv`) are not distributed, so
  their printed tables are rebuilt inline from the cited values (McKinney p312, p323, p329).
- `cartopy`/geographic mapping is not installed and Ch.9 gives no basemap API, so no map figures here; per-category counts use
  `value_counts().plot.bar()` instead (McKinney p322).
- Library drift: pandas 3 rejects the `sort_columns` keyword of Table 9-4 - order the columns yourself (McKinney p319).
- Persistent defaults go in the `matplotlibrc` file (as `~/.matplotlibrc`, loaded every run); seaborn themes drive that same configuration system (McKinney p315-316).
- Never invent a value - compute it and cite the source page for the method. Source: McKinney, *Python for Data Analysis*, 3rd ed. (O'Reilly, 2022) Ch.9, pp.299-336.
