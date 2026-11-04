---
name: maa-forest-plots
description: Draw publication-quality meta-analysis forest plots in pure Python (numpy/matplotlib), replacing the unavailable PythonMeta and zEpid ForestPlot renderers - study-label gutter, squares whose AREA is proportional to the supplied weight by sqrt sizing, 95% CI whiskers, dashed line of no effect, weight bars with FE/RE percent read-off, fixed- and random-effects pooled diamonds, prediction-interval band, subgroup/sensitivity band headers, log x-axis for RR/OR versus linear for RD/MD/SMD, and a headless PNG in a temp dir with geometry, artist-count and file-size asserts. Consumes ready-made estimates, CIs, weights and pooled rows; it computes no statistics of its own. Use when asked to draw, restyle, headless-render or self-verify a forest plot for RR/OR/SMD, choose log-versus-linear axis and null-line placement, size study marks by weight, add a pooled diamond or prediction band, or reproduce Fig 2-4 of the Masoumi & Shahraz 2022 Python meta-analysis tutorial without PythonMeta or zEpid.
---

# Forest Plots for Meta-Analysis (Python)

## Purpose
Draw the forest plot itself - per-study labels, weight-scaled marks, CI whiskers, no-effect line, weight column and pooled diamond(s) - the graphic the tutorial prints as its Fig 2, 3 and 4 (Masoumi & Shahraz 2022 p3-5). It renders numbers it is handed and estimates none of them.

## Inputs
- `labels`: `author+year` strings in top-to-bottom display order; the 17 haloperidol-versus-placebo trials are embedded as literals (Masoumi & Shahraz 2022 p7) so the file runs standalone.
- `est`, `lo`, `hi`: per-study point estimate and 95% CI bounds on the NATURAL scale; ratio bounds must be strictly positive, difference bounds may be negative (Masoumi & Shahraz 2022 p2).
- `weight`: percent weights per study; the fixed (MH) and random (DL) columns each total 100 (Masoumi & Shahraz 2022 p7). One column sizes the marks, both are printed.
- pooled rows: `(name, est, lo, hi)` per model, one diamond each - the tutorial's Fig 2 stacks a fixed and a random diamond (Masoumi & Shahraz 2022 p3).
- `pi=(lo, hi)`: optional prediction-interval band, supplied by the heterogeneity step, not computed here (Masoumi & Shahraz 2022 p3).
- `ratio`: `True` (RR, OR) -> log axis with null line at 1; `False` (RD, MD, SMD) -> linear axis, null at 0.

## Algorithms (all pure numpy/scipy)
### Core forest plot: log axis, sqrt-sized squares, whiskers, weight bars, two diamonds (Masoumi & Shahraz 2022 p3-4, p7)
```python
import numpy as np, matplotlib, os, tempfile
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
# RR (95% CI) plus fixed-effect / random-effect percent weights, Python column of Table 2 (Masoumi & Shahraz 2022 p7)
S = [("Arvanitis 1997",1.42,0.89,2.25,18.86,14.66),("Beasley 1996",1.05,0.73,1.50,31.22,16.46),
     ("Bechelli 1983",6.21,1.52,25.35,2.05,4.48),("Borison 1992",7.00,0.40,122.45,0.49,1.30),
     ("Chouinard 1993",3.49,1.11,10.95,3.10,6.10),("Durost 1964",8.68,1.26,59.95,1.09,2.65),
     ("Garry 1962",1.75,0.58,5.24,3.37,6.46),("Howard 1974",2.04,0.67,6.21,3.27,6.33),
     ("Marder 1994",1.36,0.75,2.47,11.37,12.40),("Nishikawa 1982",3.00,0.14,65.91,0.42,1.13),
     ("Nishikawa 1984",9.20,0.58,145.77,0.53,1.39),("Reschke 1974",3.79,1.06,13.60,2.48,5.20),
     ("Selman 1976",1.48,0.94,2.35,19.11,14.71),("Serafetinides 1972",8.40,0.50,142.28,0.51,1.33),
     ("Simpson 1967",2.35,0.13,43.53,0.48,1.26),("Spencer 1992",11.00,1.67,72.40,1.14,2.77),
     ("Vichaiya 1971",19.00,1.16,311.97,0.52,1.36)]
lab = [r[0] for r in S]; est,lo,hi,wfe,wre = (np.array([r[i] for r in S]) for i in (1,2,3,4,5))
k = len(S); side = 17*np.sqrt(wre/wre.max())      # sqrt sizing: linear mark size -> drawn AREA proportional to weight
fig, ax = plt.subplots(figsize=(8.2,0.34*k+2.0)); ax.set_xscale("log")
ax.axvline(1.0,color="k",lw=0.9,ls="--"); ax.hlines(np.arange(k),lo,hi,color="k",lw=0.9)
ax.scatter(est,np.arange(k),marker="s",s=side**2,c="0.25",edgecolors="k",linewidths=0.6,zorder=3)
for i in range(k):   # labels in the left gutter, weight bar and FE/RE text in the right gutter
    ax.text(-0.02,i,lab[i],transform=ax.get_yaxis_transform(),ha="right",va="center",fontsize=7.5)
    ax.barh(i,wfe[i]/100*0.13,left=1.03,height=0.55,color="0.72",transform=ax.get_yaxis_transform())
    ax.text(1.18,i,f"{wfe[i]:.2f}/{wre[i]:.2f}",transform=ax.get_yaxis_transform(),ha="left",va="center",fontsize=6.8)
def diamond(yc,e,a,b,txt,fc="0.55"):
    ax.add_patch(plt.Polygon([[a,yc],[e,yc+0.30],[b,yc],[e,yc-0.30]],closed=True,fc=fc,ec="k",lw=0.8))
    ax.text(-0.02,yc,txt,transform=ax.get_yaxis_transform(),ha="right",va="center",fontsize=7.5,style="italic")
    ax.text(1.18,yc,f"{e:.2f} ({a:.2f}, {b:.2f})",transform=ax.get_yaxis_transform(),ha="left",va="center",fontsize=7.2)
diamond(k+1.0,1.57,1.28,1.92,"Fixed (Mantel-Haenszel)"); diamond(k+2.2,2.09,1.49,2.92,"Random (DerSimonian-Laird)","0.35")
ax.hlines(k+3.4,0.73,7.17,color="0.4",ls=":"); ax.vlines([0.73,7.17],k+3.2,k+3.6,color="0.4")   # prediction band, given
ax.set_ylim(k+4.3,-0.9); ax.set_yticks([]); ax.set_xlim(0.09,420)
xt=[0.1,0.2,0.5,1,2,5,10,20,50,100,200]; ax.set_xticks(xt); ax.set_xticklabels([str(v) for v in xt],fontsize=7.5)
ax.text(0.5,1.015,"Risk ratio (95% CI)",transform=ax.transAxes,ha="center",fontsize=8)
ax.text(1.18,1.015,"Weight %  FE/RE",transform=ax.transAxes,ha="left",fontsize=8)
ax.set_xlabel("Risk ratio, haloperidol vs placebo (log scale)",fontsize=8); ax.set_title("Forest plot: 17 trials, two pooled models",fontsize=9)
fig.subplots_adjust(left=0.20,right=0.80,top=0.92,bottom=0.10)
td=tempfile.TemporaryDirectory(); fp=os.path.join(td.name,"forest_rr.png"); fig.savefig(fp,dpi=300); fig.canvas.draw()
size=os.path.getsize(fp); pix=lambda x: float(ax.transData.transform((x,0.0))[0])
back=float(ax.transData.inverted().transform((pix(1.0),0.0))[0]); sym=float(np.max(np.abs(np.sqrt(lo*hi)/est-1)))
cross=int(((lo<1)&(hi>1)).sum())
assert ax.get_xscale()=="log" and size>8000 and len(ax.patches)==k+2 and len(ax.collections)==4 and len(ax.lines)==1
assert abs(wfe.sum()-100)<0.05 and abs(wre.sum()-100)<0.05 and sym<0.02 and cross+int((lo>1).sum())==k
assert 0<pix(1.0)<Image.open(fp).size[0] and abs(back-1.0)<1e-9 and np.isclose((side**2).max()/(side**2).min(),wre.max()/wre.min())
plt.close(fig)
print(f"RESULT studies={k} fixed=1.57(1.28-1.92) random=2.09(1.49-2.92) pi=0.73-7.17 ci_cross_null={cross} wsum={wfe.sum():.2f},{wre.sum():.2f} scale={ax.get_xscale()} ci_symmetry_err={sym:.4f} mark_area_ratio={(side**2).max()/(side**2).min():.2f} patches={len(ax.patches)} null_px={pix(1.0):.1f} png_kb={size/1024:.1f}")
```
Squares are sized by `sqrt(w/wmax)`, so the mark area is proportional to the weight and the largest:smallest area ratio is exactly `16.46/1.13 = 14.57` (Masoumi & Shahraz 2022 p7). A log axis makes each CI visually symmetric about its mark: `sqrt(lo*hi)` matches the printed RR within 1.26% on all 17 rows, the residue of the 2-dp table. Both diamonds sit right of RR = 1 while the 0.73-7.17 prediction band straddles it - the p3 reading that "future studies may not approve haloperidol's superior efficacy" is a property of the band, never of the diamonds (Masoumi & Shahraz 2022 p3). The pixel checks (`null_px` inside the PNG, data->display->data round-trip ~1e-16) are what let a headless figure be trusted without a human viewing it.

### Band headers, subgroup stacks and the sensitivity-summary strip (Fig 3 and Fig 4 shapes, Masoumi & Shahraz 2022 p3, p5)
```python
import numpy as np, matplotlib, os, tempfile
matplotlib.use("Agg")
import matplotlib.pyplot as plt
BAND,POINT,DIA = 0,1,2      # row kinds: bold header rule / bare mark / pooled diamond
ROWS = [(BAND,"Missing-data imputation scenarios (Fig 4)",None,None,None),(POINT,"Worst case ICA-w",1.97,None,None),
        (POINT,"Best case ICA-b",2.71,None,None),(BAND,"Main models (Table 2)",None,None,None),
        (DIA,"Fixed MH",1.57,1.28,1.92),(DIA,"Random DL",2.09,1.49,2.92)]
fig, ax = plt.subplots(figsize=(8.0,0.42*len(ROWS)+1.3)); ax.set_xscale("log"); ax.axvline(1.0,color="k",lw=0.9,ls="--")
nb=npt=nd=0
for yc,(kind,txt,e,a,b) in enumerate(ROWS):
    ax.text(-0.02,yc,txt,transform=ax.get_yaxis_transform(),ha="right",va="center",fontsize=7.6,fontweight="bold" if kind==BAND else "normal")
    if kind==BAND: ax.hlines(yc,-0.02,1.0,transform=ax.get_yaxis_transform(),color="0.6",lw=0.7); nb+=1
    elif kind==POINT: ax.scatter([e],[yc],marker="s",s=60,c="0.45",edgecolors="k"); npt+=1
    else:
        ax.add_patch(plt.Polygon([[a,yc],[e,yc+0.24],[b,yc],[e,yc-0.24]],closed=True,fc="0.5",ec="k",lw=0.8))
        ax.hlines(yc,a,b,color="k",lw=0.9); nd+=1
    if kind: ax.text(1.02,yc,(f"{e:.2f}" if a is None else f"{e:.2f} ({a:.2f}, {b:.2f})"),transform=ax.get_yaxis_transform(),ha="left",va="center",fontsize=7.4)
ax.set_ylim(len(ROWS)-0.4,-1.0); ax.set_yticks([]); ax.set_xlim(0.9,3.4); ax.set_xticks([1,1.5,2,2.5,3])
ax.set_xticklabels(["1","1.5","2","2.5","3"],fontsize=7.5); ax.set_xlabel("Summary risk ratio (log scale)",fontsize=8)
ax.set_title("Summary-effect strip with band headers",fontsize=9); fig.subplots_adjust(left=0.26,right=0.88,top=0.86,bottom=0.14)
td=tempfile.TemporaryDirectory(); fp=os.path.join(td.name,"forest_strip.png"); fig.savefig(fp,dpi=260); size=os.path.getsize(fp)
pts=[r for r in ROWS if r[0]==POINT]; dias=[r for r in ROWS if r[0]==DIA]; cross=sum(1 for r in dias if r[3]<1.0<r[4])
assert size>8000 and ax.get_xscale()=="log" and (nb,npt,nd)==(2,2,2) and len(ax.patches)==nd and len(ax.collections)==nb+npt+nd
assert cross==0 and all(r[3]>1.0 for r in dias) and pts[0][2]<pts[1][2]
plt.close(fig)
print(f"RESULT rows={npt+nd} bands={nb} diamonds={nd} whiskers={nd} marker_only={npt} rr_worst={pts[0][2]} rr_best={pts[1][2]} spread_ratio={pts[1][2]/pts[0][2]:.4f} null_crossings={cross} scale={ax.get_xscale()} png_kb={size/1024:.1f}")
```
A band row is a bold header over a full-width rule, which is how the missing/non-missing split (Fig 3) and the imputation scenarios (Fig 4) are stacked (Masoumi & Shahraz 2022 p3-4). `lo=None` is legal and draws a bare mark: p3 prints the scenario RRs as a range only - 1.97 worst case to 2.71 best case - without CI bounds, so whiskers are omitted rather than invented (Masoumi & Shahraz 2022 p3). `spread_ratio = 1.3756` is the plot's own measure of how far the missing-data assumptions move the summary.

### Ratio versus difference: linear axis and null at 0 for MD/SMD (Masoumi & Shahraz 2022 p2)
```python
import numpy as np, matplotlib, os, tempfile
matplotlib.use("Agg")
import matplotlib.pyplot as plt
rng=np.random.default_rng(42); k=6                      # demo rows: the plotter draws, a sibling computes these
est=rng.normal(0.15,0.35,k); se=np.abs(rng.normal(0.22,0.06,k)); lo,hi=est-1.96*se,est+1.96*se
wt=rng.uniform(5,30,k); lab=[f"Trial {i+1}" for i in range(k)]
def pick_scale(ratio,lo):
    if ratio and np.any(lo<=0.0): raise ValueError("log axis cannot hold a non-positive CI bound")
    return ("log",1.0) if ratio else ("linear",0.0)
scale,null=pick_scale(False,lo); guard=False
try: pick_scale(True,lo)
except ValueError: guard=True
fig,ax=plt.subplots(figsize=(7.4,0.34*k+1.8)); ax.set_xscale(scale); ax.axvline(null,color="k",lw=0.9,ls="--")
side=14*np.sqrt(wt/wt.max()); ax.hlines(np.arange(k),lo,hi,color="k",lw=0.9)
ax.scatter(est,np.arange(k),marker="s",s=side**2,c="0.25",edgecolors="k",linewidths=0.6,zorder=3)
for i in range(k): ax.text(-0.02,i,f"{lab[i]}  {wt[i]:.1f}%",transform=ax.get_yaxis_transform(),ha="right",va="center",fontsize=7.5)
pe,pa,pb=0.12,-0.05,0.29                                 # pooled row supplied as an input, never computed here
ax.add_patch(plt.Polygon([[pa,k+0.8],[pe,k+1.1],[pb,k+0.8],[pe,k+0.5]],closed=True,fc="0.5",ec="k",lw=0.8))
ax.text(-0.02,k+0.8,"Pooled (supplied input)",transform=ax.get_yaxis_transform(),ha="right",va="center",fontsize=7.5,style="italic")
ax.text(1.02,k+0.8,f"{pe:.2f} ({pa:.2f}, {pb:.2f})",transform=ax.get_yaxis_transform(),ha="left",va="center",fontsize=7.4)
ax.set_ylim(k+1.6,-0.8); ax.set_yticks([]); ax.set_xlabel("Standardised mean difference (linear scale)",fontsize=8)
ax.set_title("Forest plot on a linear axis: MD/SMD measures",fontsize=9); fig.subplots_adjust(left=0.26,right=0.88,top=0.88,bottom=0.15)
td=tempfile.TemporaryDirectory(); fp=os.path.join(td.name,"forest_smd.png"); fig.savefig(fp,dpi=260); size=os.path.getsize(fp)
crosses=bool(pa<null<pb); neg=int((lo<0).sum())
assert ax.get_xscale()=="linear" and size>8000 and guard and len(ax.patches)==1 and neg>0 and crosses
assert float(ax.get_lines()[0].get_xdata()[0])==0.0 and all(np.diff(ax.get_xticks())>0)
plt.close(fig)
print(f"RESULT studies={k} scale={ax.get_xscale()} null={null} pooled={pe:.2f}({pa:.2f},{pb:.2f}) ci_left_of_null={neg} diamond_crosses_null={crosses} ratio_guard_trips={guard} png_kb={size/1024:.1f}")
```
`pick_scale` is the only decision the plotter makes about statistics: ratios (RR, OR) go on a log axis with the no-effect line at 1, differences (RD, MD, SMD) on a linear axis with the line at 0, and the call raises if a difference measure is pushed onto the log axis, since 5 of these 6 demo CI bounds are <= 0 and have no log coordinate (Masoumi & Shahraz 2022 p2).

## Output contract
One PNG per block, written only inside that block's own `tempfile.TemporaryDirectory()` (`forest_rr.png`, `forest_strip.png`, `forest_smd.png`), asserted to exist and exceed 8 kB, plus a `RESULT` line carrying the plotted content: `studies`, `fixed=1.57(1.28-1.92)`, `random=2.09(1.49-2.92)`, `pi`, `ci_cross_null`, `wsum`, `scale`, `mark_area_ratio`, `null_px`, `png_kb` (block 1); `rows`, `bands`, `diamonds`, `marker_only`, `rr_worst=1.97`, `rr_best=2.71`, `spread_ratio`, `null_crossings` (block 2); `scale`, `null`, `pooled`, `ci_left_of_null`, `ratio_guard_trips` (block 3). Artist counts are asserted too: 19 patches (17 weight bars + 2 diamonds) with 4 collections (CI whiskers, scatter, prediction band and its caps) and 1 axvline in block 1, so a silently dropped row fails the block instead of shipping.

## Reporting conventions
- Say in the caption what each mark encodes: square = study point estimate, area proportional to the weight column named, horizontal line = 95% CI, diamond = pooled effect with its CI as the long axis (Masoumi & Shahraz 2022 p3-4).
- Name rows versus studies (band headers inflate rows), the axis scale, and the null value; caption goes below the figure, units and scale on the axis label (Masoumi & Shahraz 2022 p4-6).
- Label both model rows with their estimator - MH for fixed, DL for random - because the tutorial chose them explicitly (Masoumi & Shahraz 2022 p2).
- Quote read-offs in words: 11 of 17 study CIs cross RR = 1 and 6 lie wholly to the right; the diamonds clear the no-effect bar, the prediction band does not (Masoumi & Shahraz 2022 p3).
- Echo the supplied pooled tuple at the source's 2-dp precision; never print a pooled estimate the plot invented (Masoumi & Shahraz 2022 p7).

## Conventions
- `pythonmeta` (PythonMeta 1.23) and `zepid` (zEpid ForestPlot) are NOT installed and are never installed here; both the Fig 2 forest plot and the zEpid missing-data forest plot are rebuilt from scratch on `matplotlib` + `numpy` (Masoumi & Shahraz 2022 p2-3).
- Always headless: `matplotlib.use("Agg")` before importing pyplot, no `show()`, `plt.close(fig)` when saved; add `plt.style.use("grayscale")` for print - the tutorial's figures are black and white.
- Explicit log ticks written as plain strings (0.1 ... 200), never `10^n`; keep `xlim` wider than the widest bound so whisker caps on Borison 1992 (0.40-122.45) and Vichaiya 1971 (1.16-311.97) are not clipped (Masoumi & Shahraz 2022 p7).
- Gutters, not gridlines: study labels and weight text use `ax.get_yaxis_transform()` so the log axis stays readable; `fig.subplots_adjust(left=, right=)` reserves the space they draw in.
- Demo data is either the literal Table 2 rows or `numpy.random.default_rng(42)` draws; nothing is read from a file.
- Take the weight column as supplied: neither `1/se^2` nor `1/sqrt(se)` percentages of the printed CIs reproduce Table 2's FE column (Arvanitis 18.86 versus 35.45 and 33.95 recomputed), because MH and DL weight studies differently - re-deriving weights is the pooling skill's job (Masoumi & Shahraz 2022 p7).
- Kept honest, not fudged: p3 narrates Fig 2 as fixed 2.09 (1.69, 2.59) and random 2.28 (1.54, 3.37) with Q = 35.18, I-squared = 54.51%, while Table 2 (p7) prints fixed 1.57 (1.28-1.92), random 2.09 (1.49-2.92), Q = 27.29, I-squared = 41.37%. These blocks plot the Table 2 pair, the only one consistent with the shipped table - weighting the 17 printed RRs by Table 2's own columns returns 1.568 and 2.086 - and the p3 pair cannot be reproduced from it.

## Out of scope
- Study effect sizes, variances and CI bounds (resp.h/resp.p/T.h/T.p -> RR, OR, SMD) belong to the effect-size skill (Masoumi & Shahraz 2022 p2).
- MH/IV/Peto versus DL pooling, model choice and the pooled estimates/weights that size the marks: pooling skill (Masoumi & Shahraz 2022 p2).
- Q, I-squared, tau-squared and the bounds of the prediction band drawn here: heterogeneity skill (Masoumi & Shahraz 2022 p3).
- The missing/non-missing subgroup test and the ACS/ICA-0/ICA-1/ICA-b/ICA-w imputations whose summary RRs are plotted: subgroup and sensitivity skill (Masoumi & Shahraz 2022 p3).
- Funnel and contour-enhanced funnel plots, Egger regression, Begg test, trim-and-fill: publication-bias skill (Masoumi & Shahraz 2022 p5-6).
- Agreement checks of these graphics against R and STATA tables: validation skill (Masoumi & Shahraz 2022 p6-7).
