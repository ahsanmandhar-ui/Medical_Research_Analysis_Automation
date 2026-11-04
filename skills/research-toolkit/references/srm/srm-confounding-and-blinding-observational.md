---
name: srm-confounding-and-blinding-observational
description: Confounding and blinding toolkit for observational (non-randomised) studies, after Dunn ch.8 pp.131-140, in pure Python (numpy/scipy/pandas/statsmodels). Builds a baseline group-comparison table with standardised mean differences, computes blocked/stratified-mean estimates, crude vs OLS or logistic covariate-adjusted estimates with a change-in-estimate flag, demonstrates collider/M-bias and the residual bias left by unmeasured confounders, and quantifies Hawthorne/observer effects from overt-vs-covert counts (Wu 78% vs 55%) and blinding shares (Kardish 33/248). Use when asked about confounding in cohort, cross-sectional or case-control data, baseline imbalance tables, crude vs adjusted numbers, blocking or restricting a variable, residual confounding, Hawthorne or observer bias, or limitations wording for observational studies.
---

# Confounding and Blinding - Observational Studies (Python)

## Purpose
Nothing is imposed or allocated, so random allocation is impossible and confounding is *always* a potential threat to internal validity; the surviving levers are restriction, blocking and analysis after measurement, plus blinding of participants and measurers against the Hawthorne and observer effects (Dunn pp131-133).

## Inputs
- `x` (0/1 self-selected exposure), `y` (numeric or 0/1 outcome), recorded covariates `c1, c2`; baseline mode takes a group column plus characteristics; observation mode takes overt/covert counts and at-risk/blinded study counts.
- `SMD_THRESH = 0.2`, `CIE_THRESH = 10.0` (%), `ALPHA = 0.05`; carry proportions as 0-1, not percentages.

## Threat catalogue (Dunn p132)
- **Confounding** - always possible; restrict (with a justified reason, Dunn p132), block, or analyse (Dunn pp132-133).
- **Carry-over** - only where exposure came in A-then-B sequences; observe both orders and compare, washout cannot be imposed (Dunn p135).
- **Hawthorne** - participants may know they are watched; observe covertly where ethical, blind them to the purpose (Dunn p136).
- **Observer** - the measurer knows group membership; measure first, record group second, double-blind where feasible (Dunn pp136-137).
- **Placebo** - not directly relevant, no treatment imposed; a control group is still legitimate but unallocated (Dunn pp134, 137).

## Algorithms (all pure numpy/scipy)

### Baseline group comparison with standardised mean differences (Dunn p134, Table 8.1)
`SMD = (m1 - m2)/sqrt((s1^2 + s2^2)/2)` flagged at `|SMD| > 0.2`; categorical contrasts in percentage points. Replays Woolf et al. 2009 (iron, active vs sedentary women, n = 28 each: 20 vs 24 yr, 169 vs 166 cm, 68 vs 62 kg, 13% vs 11% hormonal contraceptives) - such imbalance means any outcome difference may track the groups, not the exposure (Dunn p134).
```python
import numpy as np
rng = np.random.default_rng(42); names = ("age_yrs","height_cm","weight_kg","hormonal_contraceptives")
specs = {"Active": [(20,2),(169,5),(68,7),.13], "Sedentary": [(24,2),(166,5),(62,7),.11]}   # (mean,sd) or share
smp = {g: [rng.normal(mu,sd,28) for mu,sd in v[:3]] + [(rng.random(28) < v[3]).astype(float)] for g,v in specs.items()}
out, flagged = [], 0
for i, nm in enumerate(names):
    x1, x2 = smp["Active"][i], smp["Sedentary"][i]
    if i == 3:                                                    # categorical -> difference of shares
        out.append(f"{nm} {100*x1.mean():.1f}%-vs-{100*x2.mean():.1f}% = {100*(x1.mean()-x2.mean()):+.1f} pp")
    else:
        smd = (x1.mean()-x2.mean())/np.sqrt((x1.var(ddof=1)+x2.var(ddof=1))/2)
        flagged += abs(smd) > 0.2; out.append(f"{nm} {x1.mean():.1f}-vs-{x2.mean():.1f} SMD={smd:+.2f}")
print("RESULT " + " | ".join(out))
print(f"RESULT n_characteristics=4 n_imbalanced={flagged} (Woolf shape: age -2.16, height +0.99, weight +0.81)")
```

### Blocking: stratified means pooled by stratum size (Dunn p133)
`blocked = sum_k w_k*(mean_y[x=1,k] - mean_y[x=0,k])`, `w_k = n_k/sum_k n_k` (Mantel-Haenszel flavour); it needs the confounder on hand and coarse bands leave within-stratum residual confounding. Seeded scenario throughout: `x` self-selected, measured confounders `c1, c2`, unmeasured `u`, mediator `m`, truth 2.63.
```python
import numpy as np, pandas as pd
rng = np.random.default_rng(42); n = 1200
x = (rng.random(n) < .4).astype(float); u = .6*x + rng.normal(0,1,n)
c1 = 12 + 1.2*x + rng.normal(0,1.5,n); c2 = ((.30 + .25*x) > rng.random(n)).astype(float)
m = .9*x + rng.normal(0,1,n); y = 2.0*x + .7*m + .5*c1 + 1.2*c2 + .8*u + rng.normal(0,1,n)
d = pd.DataFrame({"x":x,"y":y,"c1":c1,"c2":c2}); cut = d.c1 > d.c1.median(); num = den = 0.0
for lev, s in ((False, d[~cut]), (True, d[cut])):
    diff = s[s.x==1].y.mean() - s[s.x==0].y.mean(); num += diff*len(s); den += len(s)
    print(f"  stratum={'high' if lev else 'low'}-c1 n={len(s)} mean1={s[s.x==1].y.mean():.3f} mean0={s[s.x==0].y.mean():.3f} diff={diff:+.3f}")
print(f"RESULT crude={d[d.x==1].y.mean()-d[d.x==0].y.mean():.3f} blocked={num/den:.3f} (expect 4.030 -> 3.631)")
```

### Crude vs adjusted: OLS covariate adjustment and change-in-estimate (Dunn pp132-134)
`pct_change = (b_adj - b_crude)/|b_crude| * 100`; `>= 10%` is confounding-suspect (working convention). Froud et al. 2018 (2599 kiwifruit orchards) is the real pattern: elevation and fungicide use were important confounders, organic status was not (Dunn p133).
```python
import numpy as np, pandas as pd, statsmodels.formula.api as smf
rng = np.random.default_rng(42); n = 1200
x = (rng.random(n) < .4).astype(float); u = .6*x + rng.normal(0,1,n)
c1 = 12 + 1.2*x + rng.normal(0,1.5,n); c2 = ((.30 + .25*x) > rng.random(n)).astype(float)
m = .9*x + rng.normal(0,1,n); y = 2.0*x + .7*m + .5*c1 + 1.2*c2 + .8*u + rng.normal(0,1,n)
d = pd.DataFrame({"x":x,"y":y,"c1":c1,"c2":c2,"m":m,"u":u}); g = lambda f: float(smf.ols(f, d).fit().params["x"])
crude, adj, med, oracle = g("y ~ x"), g("y ~ x + c1 + c2"), g("y ~ x + c1 + c2 + m"), g("y ~ x + c1 + c2 + u")
pct = (adj-crude)/abs(crude)*100
print(f"RESULT crude={crude:.3f} adjusted={adj:.3f} plus_mediator={med:.3f} oracle_with_u={oracle:.3f} (truth total effect 2.63)")
print(f"RESULT pct_change={pct:+.1f} confounding_flag={'yes' if abs(pct)>=10 else 'no'} residual_bias={adj-oracle:+.3f}")
```

### Logistic adjustment for a binary outcome (Dunn pp132-133)
Report the adjusted odds ratio, but odds ratios are non-collapsible: part of the crude-vs-adjusted gap is the scale, not confounding - check a risk-difference scale when the outcome is common.
```python
import numpy as np, pandas as pd, statsmodels.formula.api as smf
rng = np.random.default_rng(42); n = 1200
x = (rng.random(n) < .4).astype(float); u = .6*x + rng.normal(0,1,n)
c1 = 12 + 1.2*x + rng.normal(0,1.5,n); c2 = ((.30 + .25*x) > rng.random(n)).astype(float)
m = .9*x + rng.normal(0,1,n); lp = -1.6 + .85*x + .45*m + .55*((c1-12)/1.5) + .90*c2 + .60*u
y = (rng.random(n) < 1/(1+np.exp(-lp))).astype(float)                 # truth total effect 1.255 log-odds
d = pd.DataFrame({"x":x,"y":y,"c1":c1,"c2":c2,"m":m,"u":u}); L = lambda f: float(smf.logit(f, d).fit(disp=0).params["x"])
crude, adj, med, oracle = L("y ~ x"), L("y ~ x + c1 + c2"), L("y ~ x + c1 + c2 + m"), L("y ~ x + c1 + c2 + u")
print(f"RESULT logit_crude={crude:.3f} logit_adjusted={adj:.3f} adjusted_OR={np.exp(adj):.2f} plus_mediator={med:.3f} oracle_with_u={oracle:.3f}")
print(f"RESULT pct_change={(adj-crude)/crude*100:+.1f} residual_bias={adj-oracle:+.3f} (expect 1.934 -> 1.439, -25.6%)")
```

### Collider / M-bias: adjustment that *creates* confounding
`x <- u1 -> M <- u2 -> y`: `M` is a collider, `x` has no effect on `y`, and conditioning on `M` opens the path. Adjust pre-exposure confounders only; a mediator row (3.11 -> 2.51) answers a direct-effect question instead - analyst rule extending Dunn's baseline-only examples (Dunn p133).
```python
import numpy as np, pandas as pd, statsmodels.formula.api as smf
rng = np.random.default_rng(7); n = 20000
u1, u2 = rng.normal(0,1,n), rng.normal(0,1,n)
x = .8*u1 + rng.normal(0,1,n); y = .8*u2 + rng.normal(0,1,n)          # truth: no x -> y effect
M = .9*u1 + .9*u2 + rng.normal(0,.5,n)                                # "control variable" caused by both
d = pd.DataFrame({"x":x,"y":y,"M":M})
b0, b1 = float(smf.ols("y ~ x", d).fit().params["x"]), float(smf.ols("y ~ x + M", d).fit().params["x"])
print(f"RESULT truth=0.000 crude={b0:+.3f} collider_adjusted={b1:+.3f} verdict={'adjusting induced bias' if abs(b1)>abs(b0)+0.05 else 'no induced bias'}")
```

### Unmeasured confounding: the hard limit of adjustment (Dunn pp133-135)
With `u` never recorded the fitted slope is `b + g*a` and no model term can reach `b`: Doll and Hill's smokers may have had other unrecorded risky behaviours (Dunn p134), and Gunnarsson et al. 2017 STEMI helicopter cohorts (327 physician-staffed vs 71 standard, of 398) matched on age, sex and smoking yet two hospitals and care teams stayed a potential confounder (Dunn pp134-135).
```python
import numpy as np, pandas as pd, statsmodels.formula.api as smf
rng = np.random.default_rng(42); n = 4000
x = (rng.random(n) < .4).astype(float)                                # exposure self-selected, not allocated
u = .6*x + rng.normal(0,1,n)                                          # a = 0.6: never recorded
y = 2.0*x + .8*u + rng.normal(0,1,n)                                  # truth b = 2.0, g = 0.8
d = pd.DataFrame({"x":x,"u":u,"y":y})
b_naive, b_if_u = float(smf.ols("y ~ x", d).fit().params["x"]), float(smf.ols("y ~ x + u", d).fit().params["x"])
print(f"RESULT truth=2.000 as_analysed={b_naive:.3f} if_u_recorded={b_if_u:.3f} bias={b_naive-2:+.3f} predicted_bias_gxa={.8*.6:+.3f}")
```

### Hawthorne and observer effects: overt vs covert, blinding share (Dunn pp136-137)
Wu et al. 2018 hand hygiene: 78% compliance observed overtly vs 55% covertly; Kardish et al. 2015: 248 of 492 ecology articles (50.4%) at risk of observer bias, only 33 of those (13.3%) stating blind observation. RD CI `(p1-p2) +/- z*sqrt(p1q1/n1 + p2q2/n2)`, Katz log CI for the ratio, Wilson CI for the share. Partial blinding still counts: measure blood pressure before asking smoking status - imperfect but easy (Dunn p136).
```python
import numpy as np
from statsmodels.stats.proportion import confint_proportions_2indep, proportion_confint, proportions_ztest
k1,n1,k2,n2 = 156,200,110,200; p1,p2 = k1/n1, k2/n2; rd, rr = p1-p2, p1/p2
rdlo,rdhi = confint_proportions_2indep(k1,n1,k2,n2,method="wald")
rrlo,rrhi = confint_proportions_2indep(k1,n1,k2,n2,method="log",compare="ratio")
z,pv = proportions_ztest([k1,k2],[n1,n2]); kb,nb = 33,248
wlo,whi = proportion_confint(kb,nb,alpha=0.05,method="wilson")
print(f"RESULT overt={p1:.3f} covert={p2:.3f} risk_difference={rd:+.3f}[{rdlo:+.3f},{rdhi:+.3f}] risk_ratio={rr:.3f}[{rrlo:.3f},{rrhi:.3f}] p={float(pv):.2e}")
print(f"RESULT blinded_pct={100*kb/nb:.1f}[{100*float(wlo):.1f},{100*float(whi):.1f}] at_risk_share={100*248/492:.1f} observation_effect_flag={'yes' if pv<0.05 and abs(rd)>=0.10 else 'no'}")
```

## Output contract
`RESULT name=value` lines only: `n_characteristics`, `n_imbalanced`, per-characteristic `SMD`/`pp`, `crude`, `blocked`, `adjusted`, `pct_change`, `confounding_flag`, `logit_adjusted`/`adjusted_OR`, `plus_mediator`, `oracle_with_u`, `residual_bias`, `risk_difference`/`risk_ratio` with CIs, `p`, `blinded_pct`, `observation_effect_flag`. Always report the crude and the adjusted estimate together, never the adjusted one alone.

## Validity conditions
- Adjustment only reaches what was measured: record everything likely important about the individuals **and** the circumstances of the study, since analysis is usually the only lever left (Dunn p133).
- Participants were not allocated by the researcher, so residual confounding remains and the estimate is an association (Dunn pp131-132, 134-135); similar baselines neither prove nor exclude confounding (Dunn p135).
- Restriction needs a justified reason, as with Doll and Hill's males under 35 (Dunn p132); blocking needs the variable recorded at a fine enough grain (Dunn p133).
- Carry-over needs exposure sequences (Dunn p135); placebo needs a treatment-like intervention (Dunn p137).
- Hawthorne and observer bias are data-generation problems: no covariate removes behaviour that changed because people were watched (Dunn pp136-137).

## Reporting conventions
- Baseline: "Groups were compared on k characteristics; j showed meaningful imbalance (|SMD| > 0.2), notably <variable> (<g1> vs <g2>)" - Table 8.1 model (Dunn p134).
- Effect: "The crude difference was <crude>; after adjustment for <covariates> it was <adjusted>, a <pct>% change in estimate, consistent with confounding by the recorded variables (Dunn pp132-134)."
- Limits: name plausible unmeasured confounders, downgrade to association, and state the direction of bias in the Delaney wording - findings "may be an underestimation of the magnitude of the issues" (Dunn p139). Phrase uncertainty as residual-bias risk, never as absence of confounding.

## Out of scope
`srm-confounding-and-blinding-experimental` owns randomised-experiment confounding and blinding (random and block allocation, SRM checks, washout periods, placebo controls in trials); estimation mechanics belong to the regression and hypothesis-test skills. This skill audits internal validity, not model fit.

## Conventions
- `ALPHA = 0.05`; imbalance flag `|SMD| > 0.2` (0.1 for high-stakes claims); confounding flag `|change-in-estimate| >= 10%` - both thresholds are this skill's working conventions, not Dunn's.
- Seeded demos only (`default_rng(42)`, collider demo `default_rng(7)`); never hardcode a result - compute it and print the `RESULT` lines. No package installs.
- Stack: `statsmodels` OLS/logit for adjustment, `numpy`/`scipy` for SMDs, `statsmodels.stats.proportion` for overt-vs-covert and Wilson intervals, `pandas` for strata; headless `matplotlib` (`Agg`) only if asked.
- DAG screening needs no extra package; a causal-epidemiology helper (`zepid`, not installed) would only automate the collider/M-bias check above.
- Source: Dunn, *Scientific Research and Methodology* (Springer, 2024) ch.8, pp.131-140.
