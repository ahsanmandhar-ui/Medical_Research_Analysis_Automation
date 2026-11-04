---
name: srm-graphical-summaries
description: Pick and build the correct graph for a variable combination and render it headless with matplotlib/seaborn - histograms, stem-and-leaf, dot charts, boxplots, bar/pie, stacked & side-by-side bars, scatter, time and case-profile plots - plus the graph-construction rules and a worked number per chart. Use when asked to choose or draw a graph to summarise data, decide bar-vs-histogram or 2-D-vs-3-D, whether a pie chart is valid, or whether a graph is even needed.
---

# Graphical Summaries of Data (Python)

## Purpose
Build the clearest, simplest graph per variable combination; the aim is reader understanding, not decoration (Dunn p171).

## Inputs
- One or two columns; each is quantitative (discrete/continuous) or qualitative (nominal/ordinal).
- For two variables put the response on the vertical, the explanatory on the horizontal (Dunn p188, p204).

## Algorithms (all pure numpy/scipy)
### Which graph to choose (Dunn p172-204, Table 12.8)
| Variable(s) | Recommended graph |
|---|---|
| one quantitative | histogram (moderate/large n), stem-and-leaf (small n), dot chart |
| one qualitative | bar chart, dot chart; pie only for parts of a whole |
| qualitative + quantitative | boxplot, back-to-back stem (2 levels), 2-D dot chart |
| quantitative + quantitative | scatterplot |
| qualitative + qualitative | stacked / side-by-side bar, dot chart |

### Histogram of one quantitative variable (Dunn p175-176)
Baby birthweights (n = 44); a boundary value falls in the higher box, so 17 babies are 3.0-3.5 kg (Dunn p175).
```python
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, tempfile, os
w = np.array([1.7,2.1,2.2,2.2,2.4,2.6,2.6,2.8,2.9,3.0,3.1,3.2,3.2,3.2,3.3,3.3,3.3,3.3,3.3,3.4,3.4,3.4,
3.4,3.4,3.4,3.4,3.5,3.5,3.5,3.5,3.5,3.5,3.6,3.6,3.6,3.7,3.7,3.7,3.8,3.8,3.8,3.9,3.9,4.2])
counts,_ = np.histogram(w, bins=[1.5,2.0,2.5,3.0,3.5,4.0,4.5])
c30 = counts[3]                                     # the [3.0,3.5) box
fig,ax = plt.subplots(); _,_,patches = ax.hist(w, bins=[1.5,2.0,2.5,3.0,3.5,4.0,4.5])
ax.set_title("Histogram of 44 baby birthweights"); ax.set_xlabel("kg"); ax.set_ylabel("Number")
d=tempfile.mkdtemp(); fp=os.path.join(d,"hist.png"); fig.savefig(fp); plt.close(fig)
assert os.path.exists(fp) and len(patches)==6 and c30==17
print(f"RESULT n={len(w)} count_3.0_to_3.5={c30} bars={len(patches)} saved={os.path.exists(fp)}")
```

### Stem-and-leaf plot (Dunn p172-173)
Whole kg = stem, first decimal = leaf; leaves ordered so patterns show.
```python
import numpy as np
from collections import defaultdict
w = [1.7,2.1,2.2,2.2,2.4,2.6,2.6,2.8,2.9,3.0,3.1,3.2,3.2,3.2,3.3,3.3,3.3,3.3,3.3,3.4,3.4,3.4,
3.4,3.4,3.4,3.4,3.5,3.5,3.5,3.5,3.5,3.5,3.6,3.6,3.6,3.7,3.7,3.7,3.8,3.8,3.8,3.9,3.9,4.2]
stem = defaultdict(list)
for v in w:
    s = int(np.floor(v)); stem[s].append(int(round((v-s)*10)))
rows = {s:''.join(map(str,sorted(stem[s]))) for s in sorted(stem)}
assert {s:len(stem[s]) for s in stem} == {1:1,2:8,3:34,4:1}
print(f"RESULT stem3_leaves={rows[3]} n_in_stem3={len(stem[3])} (34 of 44 are 3.something kg)")
```

### Boxplot for qualitative + quantitative (Dunn p184-186)
Five lines: min, Q1, median (Q2), Q3, max; software flags outliers as separate points.
```python
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, tempfile, os
before=[12.1,11.0,14.1,13.8,15.5,12.2,12.8,9.9,10.8,12.7]   # house insulation, MWh (Dunn p197)
after =[12.0,10.6,13.4,11.2,15.3,13.6,12.6,8.8,9.6,12.4]
fig,ax = plt.subplots(); bp = ax.boxplot([before,after], tick_labels=["Before","After"])
ax.set_ylabel("Energy (MWh)"); ax.set_title("Insulation energy before vs after")
d=tempfile.mkdtemp(); fp=os.path.join(d,"box.png"); fig.savefig(fp); plt.close(fig)
assert os.path.exists(fp) and len(bp["boxes"])==2 and len(bp["whiskers"])==4
print(f"RESULT boxes={len(bp['boxes'])} med_before={np.median(before):.2f} med_after={np.median(after):.2f}")
```

### Case-profile plot and histogram of differences (paired data, Dunn p196-197)
Each line joins one unit's before to its after; the difference histogram is a cleaner alternative.
```python
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, tempfile, os
before=np.array([12.1,11.0,14.1,13.8,15.5,12.2,12.8,9.9,10.8,12.7])
after =np.array([12.0,10.6,13.4,11.2,15.3,13.6,12.6,8.8,9.6,12.4])
diff = before-after
fig,(a1,a2) = plt.subplots(1,2)
for b,a in zip(before,after): a1.plot([0,1],[b,a],marker="o")
a1.set_xticks([0,1]); a1.set_xticklabels(["Before","After"]); a1.set_ylabel("MWh")
a2.hist(diff, bins=np.arange(-2,3,1)); a2.set_xlabel("Energy saving (MWh)")
d=tempfile.mkdtemp(); fp=os.path.join(d,"case.png"); fig.savefig(fp); plt.close(fig)
assert os.path.exists(fp) and len(a1.lines)==10
print(f"RESULT lines={len(a1.lines)} mean_saving={diff.mean():.2f} houses_that_increased={int((diff<0).sum())}")
```

### Stacked bar for two qualitative variables (Dunn p190-191)
```python
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, tempfile, os
seasons=["Aut","Win","Spr","Sum"]
U=np.array([75,74,71,58]); M=np.array([25,27,10,10]); F=np.array([21,22,18,12]); tot=U+M+F
fig,ax=plt.subplots(); ax.bar(seasons,U,label="Unknown"); ax.bar(seasons,M,bottom=U,label="M")
ax.bar(seasons,F,bottom=U+M,label="F"); ax.legend(); ax.set_ylabel("Possums")
d=tempfile.mkdtemp(); fp=os.path.join(d,"stack.png"); fig.savefig(fp); plt.close(fig)
aut_unk = U[0]/tot[0]*100
assert os.path.exists(fp) and len(ax.patches)==12
print(f"RESULT bars={len(ax.patches)} autumn_total={tot[0]} autumn_pct_unknown={aut_unk:.1f} (source 62.0, Dunn p191)")
```

### Pie chart only for parts of a whole (Dunn p180)
Valid when exhaustive and mutually exclusive; here 69% less severe, 31% more severe.
```python
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, tempfile, os
fig,ax=plt.subplots(); wedges=ax.pie([69,31],labels=["Less severe","More severe"],autopct="%1.0f%%")[0]
ax.set_title("Severity of diagnoses")
d=tempfile.mkdtemp(); fp=os.path.join(d,"pie.png"); fig.savefig(fp); plt.close(fig)
assert os.path.exists(fp) and len(wedges)==2
print(f"RESULT wedges={len(wedges)} less_severe=69 (a bar/dot chart or plain text is usually clearer)")
```

### Scatter (two quantitative) and dot chart (Dunn p175, p188)
```python
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, tempfile, os
rng = np.random.default_rng(42)
x = rng.normal(60,15,40); y = 0.9*x + rng.normal(0,8,40)
w = np.array([1.7,2.1,2.2,2.2,2.4,2.6,2.6,2.8,2.9,3.0,3.1,3.2,3.2,3.2,3.3,3.3,3.3,3.3,3.3,3.4,3.4,3.4,
3.4,3.4,3.4,3.4,3.5,3.5,3.5,3.5,3.5,3.5,3.6,3.6,3.6,3.7,3.7,3.7,3.8,3.8,3.8,3.9,3.9,4.2])
fig,(a1,a2)=plt.subplots(1,2)
sc=a1.scatter(x,y); a1.set_xlabel("Explanatory"); a1.set_ylabel("Response")
a2.plot(w,np.zeros(len(w)),"o"); a2.set_yticks([]); a2.set_xlabel("Baby weight (kg)")
d=tempfile.mkdtemp(); fp=os.path.join(d,"scatter.png"); fig.savefig(fp); plt.close(fig)
pts=len(sc.get_offsets())
assert os.path.exists(fp) and pts==40 and len(w)==44
print(f"RESULT scatter_pts={pts} dot_pts={len(w)} r={np.corrcoef(x,y)[0,1]:.3f}")
```

### Time plot (many time points, Dunn p172, p199)
```python
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, tempfile, os
mins=[5,64,78,115,177,245,247,262,271,428]; wt=[3.8,3.3,3.6,3.8,3.6,2.2,1.7,2.8,3.2,3.5]
fig,ax=plt.subplots(); ln,=ax.plot(mins,wt,marker="o")
ax.set_xlabel("Minutes after midnight"); ax.set_ylabel("kg"); ax.set_title("Time plot of birthweights")
d=tempfile.mkdtemp(); fp=os.path.join(d,"time.png"); fig.savefig(fp); plt.close(fig)
assert os.path.exists(fp) and len(ln.get_xdata())==10
print(f"RESULT points={len(ln.get_xdata())} first={wt[0]} last={wt[-1]}")
```

## Output contract
Each chart asserts its artist count (`bars`/`boxes`/`lines`/`wedges`/`pts`/`points`) and saves one PNG into a
fresh `tempfile` dir (assert it exists). Print `RESULT name=value` per figure plus the reproduced source number
(`count_3.0_to_3.5=17`, `autumn_pct_unknown=62.0`, `mean_saving=0.54`).

## Validity conditions
- A pie chart is meaningless unless categories are parts of one whole, exhaustive and mutually exclusive; browser
  use or iPhone ownership across states sum over 100%, so a bar/dot chart fits instead (Dunn p180-181).
- Describe a quantitative distribution as average, variation, shape (skew/symmetric/bimodal), outliers, and always
  explain what the graph says in words (Dunn p171, p177).
- Never truncate a bar-chart axis below zero and never add an artificial 3rd dimension; both mislead (Dunn p193, p200).
- Sometimes no graph is needed: a fact like "69% less severe", a bar/dot chart, or a small table often reads better (Dunn p183).

## Conventions
- matplotlib is headless (`matplotlib.use("Agg")`); every figure is saved only inside `tempfile.mkdtemp()`.
- Random demo data (scatter) seeded with `numpy.random.default_rng(42)`; never invent a number - compute it,
  print the `RESULT` line and cite the source page for the method.
- Seaborn is an acceptable wrapper; no interactive plotly fallback is installed, so every chart stays on headless matplotlib.
- Source: Dunn, *Scientific Research and Methodology* (Springer, 2024) Ch.12, pp.171-208.
