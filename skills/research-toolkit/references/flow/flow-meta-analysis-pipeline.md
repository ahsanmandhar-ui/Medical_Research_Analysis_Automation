---
name: flow-meta-analysis-pipeline
description: End-to-end meta-analysis / evidence pooling in pure Python (numpy/scipy/statsmodels/matplotlib) - study-level intake, per-study effect sizes (log RR/OR/RD, MD, Hedges g), fixed-effect inverse-variance / Mantel-Haenszel / Peto pooling, DerSimonian-Laird / Paule-Mandel / REML random effects, Cochran Q, I^2, tau^2, Higgins-Thompson prediction interval, forest and contour-enhanced funnel plots, Egger / Begg / trim-and-fill small-study-effect checks and reportable RESULT lines. Reimplements the pythonmeta and zEpid numerics. Use when asked to pool studies, run a meta-analysis or systematic-review synthesis, reproduce a Cochrane pooled estimate, test heterogeneity or publication bias, or draw a forest/funnel plot from a study-level table.
---

# Meta-Analysis Pipeline (Python)

## Purpose
Chain one dense workflow: study-level intake -> per-study effect size -> fixed & random pooling ->
heterogeneity -> forest/funnel plots -> small-study-effect tests -> reportable numbers; no external meta
package is required (Masoumi & Shahraz 2022 p2, p6-7).

## Inputs
- Binary per study: `events_t,n_t,events_c,n_c`; or Cochrane dropout form `resp_t,fail_t,drop_t,resp_c,fail_c,drop_c`.
- Continuous per study: `mean_t,sd_t,n_t,mean_c,sd_c,n_c`.
- Estimate+CI per study: `rr,lo,hi` -> `y=log(rr)`, `se=(log(hi)-log(lo))/(2*Z)` on log scales (Masoumi & Shahraz 2022 p7).
- `MEASURE in {rr,or,rd,md,smd}`, `ALPHA=0.05`, `Z=norm.ppf(0.975)`; optional `group` column for subgroups.

## Algorithms (all pure numpy/scipy)

### 1. Effect sizes and variances (feeds stage 2)
```python
import numpy as np
from scipy import stats
Z = float(stats.norm.ppf(0.975))
ev_t,n_t,ev_c,n_c = (np.array([32.,41,12,55]), np.array([100.,120,60,90]),
                     np.array([20.,30,7,40]),  np.array([100.,118,62,95]))
a,b,c,d = ev_t, n_t-ev_t, ev_c, n_c-ev_c
cc = (np.vstack([a,b,c,d])==0).any(0)                       # zero-cell studies
a,b,c,d = [x + 0.5*cc for x in (a,b,c,d)]                  # +0.5 to ALL four cells (Masoumi p2)
n1,n0 = a+b, c+d; p1,p0 = a/n1, c/n0
log_rr = np.log(p1/p0); var_rr = (1/a-1/n1)+(1/c-1/n0)     # log RR, Woolf variance
log_or = np.log((a*d)/(b*c)); var_or = 1/a+1/b+1/c+1/d     # log OR, Woolf variance
rd = p1-p0; var_rd = p1*(1-p1)/n1 + p0*(1-p0)/n0           # risk difference
m1,s1,M0,s0,N1,N0 = 5.2,1.1,4.4,1.3,50.,45.                # continuous example -> Hedges g
sp = np.sqrt(((N1-1)*s1**2+(N0-1)*s0**2)/(N1+N0-2)); dv=(m1-M0)/sp
J = 1-3/(4*(N1+N0-2)-1); g = J*dv; var_g = J**2*((N1+N0)/((N1-1)*(N0-1))+dv**2/(2*(N1+N0-2)))
print(f"RESULT log_rr={np.round(log_rr,4).tolist()} se_rr={np.round(np.sqrt(var_rr),4).tolist()}")
print(f"RESULT zero_cell_corrected={int(cc.sum())} hedges_g={g:.4f} var_g={var_g:.4f}")
```

### 2. Fixed/random pooling, heterogeneity, prediction interval (worked haloperidol RR)
```python
import numpy as np
from scipy import stats
Z = float(stats.norm.ppf(0.975))
rr = np.array([1.42,1.05,6.21,7.0,3.49,8.68,1.75,2.04,1.36,3.0,9.2,3.79,1.48,8.4,2.35,11.,19.])
lo = np.array([.89,.73,1.52,.40,1.11,1.26,.58,.67,.75,.14,.58,1.06,.94,.50,.13,1.67,1.16])
hi = np.array([2.25,1.5,25.35,122.45,10.95,59.95,5.24,6.21,2.47,65.91,145.77,13.6,2.35,142.28,43.53,72.4,311.97])
y = np.log(rr); se = (np.log(hi)-np.log(lo))/(2*Z); v = se**2; w = 1/v          # effect size from CI
mfe = (w*y).sum()/w.sum(); sfe = np.sqrt(1/w.sum())                             # fixed (inverse-variance)
df = len(y)-1; q = (w*(y-mfe)**2).sum()                                         # Cochran Q
tau2 = max(0., (q-df)/(w.sum()-(w**2).sum()/w.sum()))                           # DerSimonian-Laird tau^2
wr = 1/(v+tau2); mre = (wr*y).sum()/wr.sum(); sere = np.sqrt(1/wr.sum())        # random effects
I2 = max(0., (q-df)/q)*100                                                     # Higgins heterogeneity
tcrit = float(stats.t.ppf(0.975, df-1)); half = tcrit*np.sqrt(tau2+sere**2)     # Higgins-Thompson PI
print(f"RESULT FE_RR={np.exp(mfe):.4f} [{np.exp(mfe-Z*sfe):.3f},{np.exp(mfe+Z*sfe):.3f}]")
print(f"RESULT RE_RR={np.exp(mre):.4f} [{np.exp(mre-Z*sere):.3f},{np.exp(mre+Z*sere):.3f}]")
print(f"RESULT tau2={tau2:.4f} I2={I2:.2f} Q={q:.2f} df={df} p={stats.chi2.sf(q,df):.4f}")
print(f"RESULT PI_RR={np.exp(mre-half):.2f}-{np.exp(mre+half):.2f} crosses_null={bool(mre-half<0<mre+half)}")
assert abs(np.exp(mfe)-1.57)<0.01 and abs(np.exp(mre)-2.09)<0.01 and abs(tau2-0.146)<0.001
```
Prints `FE_RR=1.57 (1.28-1.92)`, `RE_RR=2.09 (1.49-2.92)`, `tau2=0.146`, `I2=41.3`, `Q=27.3` on 16 df
(p=0.039), prediction interval `RR 0.85-5.09` crossing the null (Masoumi & Shahraz 2022 p3, p6 Table 1).
`Paule-Mandel`/`REML` (solve `sum w(t)(y-ybar)^2=df`, or minimise the REML likelihood) also give tau^2 ~0.15 (Masoumi p2).

### 3. Mantel-Haenszel and Peto fixed-effect pooling
```python
import numpy as np
from scipy import stats
Z = float(stats.norm.ppf(0.975))
rng = np.random.default_rng(42); k = 10
n1 = rng.integers(60,240,k).astype(float); n0 = n1.copy()          # balanced arms
a = np.round(0.06*n1); c = np.round(0.03*n0); b,d = n1-a, n0-c     # rare events (Peto regime)
n = n1+n0; p1,p0 = a/n1, c/n0
y = np.log(p1/p0); v = (1/a-1/n1)+(1/c-1/n0); w = 1/v
IV = np.exp((w*y).sum()/w.sum())                                   # inverse-variance fixed
R,S = a*n0/n, c*n1/n; mh = R.sum()/S.sum()                         # Mantel-Haenszel RR
vr = ((n0/n)**2*a*(1-p1)).sum(); vs = ((n1/n)**2*c*(1-p0)).sum()
se_mh = np.sqrt(vr/R.sum()**2 + vs/S.sum()**2)                     # Greenland-Robins variance
E = (a+c)*n1/n; V = n1*n0*(a+c)*(b+d)/(n**2*(n-1))                 # Peto O-E and hypergeometric var
peto = np.exp((a-E).sum()/V.sum()); se_peto = np.sqrt(1/V.sum())
print(f"RESULT IV_RR={IV:.4f} MH_RR={mh:.4f} [{mh*np.exp(-Z*se_mh):.3f},{mh*np.exp(Z*se_mh):.3f}]")
print(f"RESULT Peto_OR={peto:.4f} [{peto*np.exp(-Z*se_peto):.3f},{peto*np.exp(Z*se_peto):.3f}]")
```
On homogeneous rare-event data IV, MH and Peto agree to ~0.003 (expect RR~2.07, MH~2.08, Peto OR~2.07);
MH is the fixed default and Peto needs rare events plus balanced arms (Masoumi & Shahraz 2022 p2).

### 4. Forest + contour-enhanced funnel plots, Egger / Begg / trim-and-fill (headless, k>=10)
```python
import numpy as np, tempfile, pathlib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
Z = float(stats.norm.ppf(0.975))
rr = np.array([1.42,1.05,6.21,7.0,3.49,8.68,1.75,2.04,1.36,3.0,9.2,3.79,1.48,8.4,2.35,11.,19.])
lo = np.array([.89,.73,1.52,.40,1.11,1.26,.58,.67,.75,.14,.58,1.06,.94,.50,.13,1.67,1.16])
hi = np.array([2.25,1.5,25.35,122.45,10.95,59.95,5.24,6.21,2.47,65.91,145.77,13.6,2.35,142.28,43.53,72.4,311.97])
y = np.log(rr); se = (np.log(hi)-np.log(lo))/(2*Z); v = se**2; w = 1/v
mfe = (w*y).sum()/w.sum(); sfe = np.sqrt(1/w.sum()); df = len(y)-1
q = (w*(y-mfe)**2).sum(); tau2 = max(0.,(q-df)/(w.sum()-(w**2).sum()/w.sum()))
wr = 1/(v+tau2); mre = (wr*y).sum()/wr.sum(); sere = np.sqrt(1/wr.sum())
out = pathlib.Path(tempfile.mkdtemp(prefix="meta_"))
fig, ax = plt.subplots(figsize=(7, 0.3*len(y)+2))                 # forest plot
for i in range(len(y))[::-1]:
    ax.plot([y[i]-Z*se[i], y[i]+Z*se[i]], [i, i], color="0.25", lw=1.2); ax.plot(y[i], i, "s", ms=5, color="0.1")
for lvl,(m,s,col) in enumerate([(mre,sere,"tab:blue"),(mfe,sfe,"tab:red")]):
    yy = len(y)+0.5+lvl*0.8; ax.fill([m-s,m,m+s,m,m-s],[yy,yy+.26,yy,yy-.26,yy],color=col)
ax.axvline(0, ls="--", color="0.4"); ax.set_xscale("log"); ax.set_title("Forest: haloperidol RR")
fig.tight_layout(); fig.savefig(out/"forest.png", dpi=110); plt.close(fig)
fig, ax = plt.subplots(figsize=(6,5)); grid = np.linspace(0, se.max()*1.05, 60)   # contour funnel
for mult,col in [(Z,"#ffd9d9"),(2.576,"#ffe9b3"),(3.29,"#c6f7c6")]:
    ax.fill_between(grid, -mult*grid, mult*grid, color=col)
ax.scatter(se, y, s=26, color="tab:blue"); ax.plot(grid, mre+Z*grid, "k--", lw=1); ax.plot(grid, mre-Z*grid, "k--", lw=1)
ax.axhline(0, color="0.4", ls="-."); ax.invert_xaxis(); ax.set_xlabel("SE"); ax.set_ylabel("log RR")
fig.tight_layout(); fig.savefig(out/"funnel.png", dpi=110); plt.close(fig)
assert len(y) >= 10, "Egger/Begg/trim-and-fill are underpowered below k=10"       # validity gate
egg = sm.WLS(y, sm.add_constant(se), weights=1/v).fit()            # Egger: slope on se = bias coefficient
btau, bp = stats.kendalltau((y-mre)/np.sqrt(np.maximum(v-sere**2,1e-12)), v)      # Begg rank correlation
r = stats.rankdata(np.abs(y-mre)); T = r[(y-mre)>0].sum()                          # Duval-Tweedie L0
k0 = int(min(max(0, round((4*T - len(y)*(len(y)+1))/(2*len(y)-1))), len(y)-2))
print(f"RESULT forest={(out/'forest.png').exists()} funnel={(out/'funnel.png').exists()}")
print(f"RESULT egger_bias={egg.params[1]:.4f} egger_p={egg.pvalues[1]:.5f} begg_tau={btau:.4f} begg_p={bp:.4f}")
print(f"RESULT trim_fill_k0={k0} adjusted_logRR={egg.params[0]:.4f}")
assert abs(egg.params[1]-1.789)<0.01 and abs(btau-0.427)<0.01
```
Judge the **bias coefficient** row (Egger slope 1.789, p<0.001; Begg tau 0.427, p 0.017), not the adjusted
intercept; trim-and-fill is a sensitivity estimate of suppressed studies (Masoumi & Shahraz 2022 p6 Table 1).

## Workflow
Intake -> effect sizes (#1) -> fixed & random pooling (#2) + MH/Peto (#3) -> read heterogeneity before
interpreting -> inspect the forest plot weights/diamond (#4) -> small-study battery only when k>=10 -> report
every blank from a `RESULT` line. If `I2>50%` or Q is significant prefer random effects and report the
prediction interval alongside the summary CI (Masoumi & Shahraz 2022 p3-4, p6).

## Validity conditions
- Pool only studies sharing population, exposure contrast, outcome and horizon; never mix RR with OR or log-
  scale with raw-scale effects (Masoumi p1-2). On log scales the no-effect value is RR 1 (log 0); state the
  direction of benefit explicitly (Masoumi p2-3).
- Significant I2 is heterogeneity, not proof of bias; asymmetry can also come from heterogeneity,
  non-availability or a wrong measure (Masoumi p4-6). Zero cells get +0.5 in all four cells per study; MH/Peto
  are fixed-only and Peto needs rare, balanced events (Masoumi p2).
- Do not run Egger/Begg/trim-and-fill below k~10 (underpowered); compare against R/Stata only at the published decimals.
- Clinical heterogeneity is never detectable statistically - read the studies (Masoumi p6).

## Output contract
Per `RESULT` line: `log_rr/se_rr`, `hedges_g/var_g` (#1); `FE_RR/RE_RR` with `lo/hi`, `tau2`, `I2`, `Q`, `df`, `p`,
`PI_RR`, `crosses_null` (#2); `IV_RR`, `MH_RR`, `Peto_OR` (#3); `egger_bias/egger_p`, `begg_tau/begg_p`,
`trim_fill_k0` (#4). Plots written to a fresh temp dir as `forest.png` and `funnel.png` (contour-enhanced);
subgroup table adds per-group random estimates plus `between_group_chi2, df, p, tau2_between`.

## Conventions
- Constants: `ALPHA=0.05`, `Z=norm.ppf(0.975)`, `CONF=0.95`; seeded demo data only via `numpy.random.default_rng(42)`; prediction interval uses `t` on k-2 df.
- Plots are headless: `matplotlib.use("Agg")`, save only into a `tempfile.mkdtemp()` dir, then close the fig.
- `pythonmeta` and `zEpid` are NOT installed and must not be pip-installed; the numerics are reimplemented here on numpy/scipy/statsmodels (Masoumi & Shahraz 2022 p2-6).
- Never invent a value - always compute it and cite the source page for the method.

## Out of scope
Single-study inference, network and meta-regression modelling belong to siblings; this owns the one-review
end-to-end flow and the shared stage contract.
