# Research Automation Toolkit

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TimesFM](https://img.shields.io/badge/TimesFM-1.0-200M-purple.svg)](https://github.com/google-research/timesfm)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen.svg)](https://github.com/ahsanmandhar-ui/research-automation-toolkit)

> Narrative: Data analysis will be automated in the future.
> This repository bundles a complete research-and-data-analysis toolkit with 61 specialist method guides, a TimesFM-1.0-200M zero-shot forecasting pipeline, and Mermaid skill-graph diagrams — all structured for reproducible, AI-augmented research automation.
> Copyright © Ahsan Mandhar.

---

## TimesFM Forecast: Meta-Analysis Publications (2015-2031)

Zero-shot foundation forecast under AI-automation structural shift, generated with TimesFM-1.0-200M (CPU-only, bfloat16, <2GB memory).

![Meta-Analysis Publications Forecast](data/meta_analysis_forecast_2027_2031.png)

> Run `python analysis/timesfm_forecast_v2.py` to generate the forecast CSV, table, and PNG. Outputs land in `data/`.

---
### Skills Overview

### SRM — Statistical Research Methods (22 skills)

| Skill ID | Skill Name | Role |
|----------|-----------|------|
| s1 | srm-research-question-design | Turn a topic into an answerable quantitative research question; Extract POCI elements: Population, Outcome, Comparison, Intervention; Identify response and explanatory variables |
| s2 | srm-study-design-selection | Pick and encode the right quantitative design from POCI components; Descriptive vs analytical vs experimental design; Draft a study-design protocol |
| s3 | srm-sampling-methods | Draw and audit research samples using numpy.random and pandas.sample; Simple random, systematic, stratified, cluster sampling; Assess whether a sample can represent its population |
| s4 | srm-variable-types-encoding | Classify every variable as quantitative or qualitative; Nominal vs ordinal, discrete vs continuous; Encode into correct pandas dtype |
| s5 | srm-data-collection-protocols | Study measurement and data capture protocols; Data collection planning and execution |
| s6 | srm-numerical-summaries-quantitative | Numerically describe one quantitative variable or many by group; Mean vs median, range, sample standard deviation, variance; IQR, percentiles, standardise to z-score, flag outliers |
| s7 | srm-numerical-summaries-qualitative | Numerically summarise qualitative categorical data; Overall, row and column proportions and percentages; Odds or odds ratio from a contingency table |
| s8 | srm-graphical-summaries | Visualise distributions and relationships graphically; Publication-quality figures from pandas frames |
| s9 | srm-distributions-normal-model | Normal distribution model in pure Python; Probability density, z-scores, standard normal table |
| s10 | srm-confidence-intervals-single | Confidence interval for ONE population proportion or mean; 90/95/99% CI, margin of error, standard error; Sample size determination |
| s11 | srm-confidence-intervals-comparisons | Confidence intervals for comparisons between two groups; Paired before/after mean differences; Two independent means: Welch vs pooled; Odds ratio from four cell counts |
| s12 | srm-hypothesis-testing-framework | Hypothesis testing framework: null vs alternative; P-values, Type I and Type II errors, statistical power; Decision logic for accepting or rejecting hypotheses |
| s13 | srm-hypothesis-tests-means | t-tests: one-sample, paired/dependent, two-sample/Welch; Equal vs unequal variance assumptions; Turn a t-statistic into a P-value |
| s14 | srm-hypothesis-tests-categorical | Chi-square tests for categorical data; Independence and goodness-of-fit tests; Expected counts below 5 rule |
| s15 | srm-correlation | Scatterplot reading: form, direction, variation, outliers; Pearson correlation coefficient r and R-squared; Influential points detection |
| s16 | srm-linear-regression | Fit simple linear regression y_hat = b0 + b1*x; Least-squares OLS via pure Python; Interpret slope and intercept, test slope significance |
| s17 | srm-probability-and-decision-logic | Probability from three approaches: classical, frequentist, subjective; Independence of events, conditional probability; Expected number of outcomes |
| s18 | srm-confounding-and-blinding-experimental | Protect internal validity of an experiment treatment vs control; Balance groups, check randomisation, SRM check; A/B test and clinical trial design audit |
| s19 | srm-confounding-and-blinding-observational | Confounding and blinding for observational non-randomised studies; Cohort, cross-sectional, case-control designs; Baseline imbalance tables, crude vs adjusted numbers |
| s20 | srm-statistical-reference | Lookup reference layer for Dunn Scientific Research and Methodology; Notation dictionary, parameter vs statistic vs standard error; Verify code against book printed answers |
| s21 | srm-reporting-research | Build and QC the written scientific report following Dunn structure; AIMRaD and STARD-compliant report assembly; Results section, tables, figures, critical reading |
| s22 | srm-correlation-linear-integration | Integrate correlation and regression results; Combine r, R-squared, and regression output; Report strength of evidence and effect size |

---
### MAA — Meta-Analysis (7 skills)

| Skill ID | Skill Name | Role |
|----------|-----------|------|
| m1 | maa-effect-sizes | Compute study-level effect sizes for meta-analytic input table; Risk ratio RR, odds ratio OR on log-OR scale, risk difference RD; Variance, standard error, CI behind one effect size; Handle zero cell or dropouts |
| m2 | maa-pooling-models | Fixed- and random-effects pooling of meta-analytic evidence; Inverse-variance fixed-effect weights; Mantel-Haenszel and Peto methods; Study weights and pooled result |
| m3 | maa-heterogeneity | Heterogeneity statistics computed from first formulas; Cochran's Q with chi-square P value; I-squared, tau-squared, H-squared; Prediction interval |
| m4 | maa-forest-plots | Draw publication-quality meta-analysis forest plots; Replace unavailable PythonMeta and zEpid ForestPlot; Log versus linear axis, null-line placement; Size study marks by precision |
| m5 | maa-publication-bias | Detect publication bias and small-study effects; Contour-enhanced funnel plot with matplotlib zones; Egger and Begg P-values; Assess asymmetry in funnel plot |
| m6 | maa-subgroup-meta-regression | Subgroup contrasts and meta-regression; Sensitivity and robustness analysis; Test whether study characteristics explain pooled effect; Drop-one sensitivity analysis |
| m7 | maa-validation-vs-r-stata | Cross-validate Python meta-analysis against R meta/metafor or Stata; Write reference path for comparison; Choose cross-tool tolerance; Explain small discrepancies between tools |

---

### PDA — Python Data Analysis (13 skills)

| Skill ID | Skill Name | Role |
|----------|-----------|------|
| p1 | pda-python-language-core | Core Python language semantics for data analysis; Standard library: re, itertools, collections, pathlib, os, sys, tempfile; No third-party dependencies; Raw-Python data prep, grouping, counting with dict/set |
| p2 | pda-numpy-fundamentals | NumPy ndarray toolkit for array-oriented data work; Build, subset, aggregate NumPy arrays; Select rows by boolean mask or integer index array; Compute sum/mean along axis, sort, unique |
| p3 | pda-numpy-advanced | Advanced NumPy internals: shape, strides, flags, dtype hierarchy; Zero-copy strided views, np.issubdtype; Fix operands could not be broadcast together errors; Handle arrays too large for RAM |
| p4 | pda-pandas-foundations | Build and inspect core pandas objects: Series, DataFrame, Index; Creation from arrays, lists, dicts; Index and name attributes; Select rows/columns with loc/iloc, filter with mask |
| p5 | pda-data-loading-io | Load data into pandas DataFrame from every source; CSV, Excel, JSON, database query into DataFrame; scikit-learn toy datasets, seeded simulated generators |
| p6 | pda-data-cleaning | Clean data end to end in pure Python pandas/numpy; Drop duplicate rows, filter rows on condition; Handle NaN values, join two frames |
| p7 | pda-data-wrangling-reshape | Wrangle pandas DataFrame end to end; Creation, description head/tail/shape/describe; loc vs iloc and conditional selection; Count unique classes, group by category |
| p8 | pda-groupby-aggregation | Split-apply-combine in pure pandas/numpy; Group by array, column, dict, Series, callable, MultiIndex level; Optimized or custom aggregation functions; Per-group stats, top-n per group, pivot tables |
| p9 | pda-time-series | Date/timestamp APIs, date_range and frequency aliases; Shifting, resampling down/up-sample, OHLC; Rolling/expanding windows with min_periods, EWMA; Parse dates, handle time zones |
| p10 | pda-plotting-visualization | Build publication figures from pandas frames with matplotlib and seaborn; Plot Series or DataFrame, histogram, KDE; Subplots/panels, legend, error bars; Save figure headless |
| p11 | pda-modeling-libraries | Fit statistical and ML models from pandas frame; Patsy formula design matrices: dmatrices, standardize/center; C(), I(), key1:key2 interactions; statsmodels, patsy, sklearn integration |
| p12 | pda-ipython-jupyter-workflow | Pure-Python reproduction of IPython/Jupyter analysis workflow; Magic commands, timing, profiling, code introspection; Structure notebook for reproducibility |
| p13 | pda-analysis-examples | Analysis examples and worked demonstrations; End-to-end analysis walkthroughs; Reference implementations |

---

### MLK — Machine Learning (13 skills)

| Skill ID | Skill Name | Role |
|----------|-----------|------|
| l1 | mlk-vectors-matrices-arrays | Linear algebra for ML: vectors, matrices, tensors; Dot/inner products, norms, projections; Matrix multiply, inverse, solve; Eigenvalues, SVD, PCA mechanics, loss functions |
| l2 | mlk-data-loading | Load data into training-ready pandas/sklearn frame; scikit-learn toy datasets, seeded simulated generators; Import CSV, Excel, JSON; Query database into DataFrame |
| l3 | mlk-data-wrangling | Wrangle pandas DataFrame end to end in pure Python; Creation, description, head/tail/shape/describe; Drop duplicates, filter rows, group by category; Handle NaN, join two frames |
| l4 | mlk-datetime-features | Parse date strings and build ML-ready temporal features; to_datetime with explicit format; Break timestamp into component features; Encode day-of-week, compute time differences |
| l5 | mlk-categorical-features | Encode categorical string features into numeric ML inputs; Label and ordinal reduction, one-hot/dummy encoding; Target-encode a group, handle high-cardinality features |
| l6 | mlk-text-features | Text features for machine learning in pure Python; Strip whitespace and punctuation; CountVectorizer, TfidfVectorizer pipelines; Stemming, stop-word lists |
| l7 | mlk-numerical-features | Build ML-ready numeric features from quantitative columns; Scale, normalise, standardise numeric features; Polynomial and interaction terms; Log/sqrt transform, outlier detection, binning |
| l8 | mlk-dim-reduction-extraction | Reduce original features to smaller projected components; PCA with variance target, ICA, NMF, kernel PCA; t-SNE, TruncatedSVD; How many components keep 95% or 99% variance |
| l9 | mlk-naive-bayes | Train Gaussian, multinomial, Bernoulli naive Bayes classifiers; sklearn.naive_bayes with numpy; Compare NB variants for different data types |
| l10 | mlk-svm | Train, inspect and tune support vector machines; sklearn.svm LinearSVC, SVC, SVR; Max-margin hyperplane, kernel trick; C/gamma values, class probabilities |
| l11 | mlk-trees-forests | Train, prune, export and interpret decision trees, random forests, boosted trees; sklearn.tree, sklearn.ensemble, numpy; Feature importance, OOB score, gradient boosting |
| l12 | mlk-neural-networks | Build, train and diagnose feedforward neural networks; Hand-written one-hidden-layer numpy MLP; Forward propagation, overfitting diagnostics; Keras-style DNN/MLP fundamentals |
| l13 | mlk-clustering | Partition observations with no target unsupervised clustering; sklearn.cluster + sklearn.metrics; How many clusters, silhouette analysis; K-means, hierarchical, DBSCAN |

---

### FLOW — Workflow (6 skills)

| Skill ID | Skill Name | Role |
|----------|-----------|------|
| f1 | flow-py-analysis-env | Python analysis environment setup; Environment pinning, reproducibility gates |
| f2 | flow-unit-hygiene | Unit hygiene and provenance gates; Audit trails, reproducibility checks |
| f3 | flow-supervisor | Organise parallel work across many skills; Fleet supervision, task coordination |
| f4 | flow-analysis-pipeline-dunn | Run and gate a whole quantitative study in Python; Research question -> design -> describe -> infer -> report; Following Dunn's six basic steps |
| f5 | flow-meta-analysis-pipeline | End-to-end meta-analysis / evidence pooling; Study-level intake, per-study effect sizes; Reproduce Cochrane pooled estimate |
| f6 | flow-corpus-extraction | Re-extract or extend PDF document corpus with PyMuPDF; Inclusive page-range Markdown rendering; Monospace code detection by per-line font mix; Map printed folios to PDF pages |

---

### Complete Skill List — All 61 Skills by Family

| Family | Skill ID | Skill Name |
|--------|----------|----------|
| **SRM** (22) | s1 | srm-research-question-design |
| | s2 | srm-study-design-selection |
| | s3 | srm-sampling-methods |
| | s4 | srm-variable-types-encoding |
| | s5 | srm-data-collection-protocols |
| | s6 | srm-numerical-summaries-quantitative |
| | s7 | srm-numerical-summaries-qualitative |
| | s8 | srm-graphical-summaries |
| | s9 | srm-distributions-normal-model |
| | s10 | srm-confidence-intervals-single |
| | s11 | srm-confidence-intervals-comparisons |
| | s12 | srm-hypothesis-testing-framework |
| | s13 | srm-hypothesis-tests-means |
| | s14 | srm-hypothesis-tests-categorical |
| | s15 | srm-correlation |
| | s16 | srm-linear-regression |
| | s17 | srm-probability-and-decision-logic |
| | s18 | srm-confounding-and-blinding-experimental |
| | s19 | srm-confounding-and-blinding-observational |
| | s20 | srm-statistical-reference |
| | s21 | srm-reporting-research |
| | s22 | srm-correlation-linear-integration |
| **MAA** (7) | m1 | maa-effect-sizes |
| | m2 | maa-pooling-models |
| | m3 | maa-heterogeneity |
| | m4 | maa-forest-plots |
| | m5 | maa-publication-bias |
| | m6 | maa-subgroup-meta-regression |
| | m7 | maa-validation-vs-r-stata |
| **PDA** (13) | p1 | pda-python-language-core |
| | p2 | pda-numpy-fundamentals |
| | p3 | pda-numpy-advanced |
| | p4 | pda-pandas-foundations |
| | p5 | pda-data-loading-io |
| | p6 | pda-data-cleaning |
| | p7 | pda-data-wrangling-reshape |
| | p8 | pda-groupby-aggregation |
| | p9 | pda-time-series |
| | p10 | pda-plotting-visualization |
| | p11 | pda-modeling-libraries |
| | p12 | pda-ipython-jupyter-workflow |
| | p13 | pda-analysis-examples |
| **MLK** (13) | l1 | mlk-vectors-matrices-arrays |
| | l2 | mlk-data-loading |
| | l3 | mlk-data-wrangling |
| | l4 | mlk-datetime-features |
| | l5 | mlk-categorical-features |
| | l6 | mlk-text-features |
| | l7 | mlk-numerical-features |
| | l8 | mlk-dim-reduction-extraction |
| | l9 | mlk-naive-bayes |
| | l10 | mlk-svm |
| | l11 | mlk-trees-forests |
| | l12 | mlk-neural-networks |
| | l13 | mlk-clustering |
| **FLOW** (6) | f1 | flow-py-analysis-env |
| | f2 | flow-unit-hygiene |
| | f3 | flow-supervisor |
| | f4 | flow-analysis-pipeline-dunn |
| | f5 | flow-meta-analysis-pipeline |
| | f6 | flow-corpus-extraction |

---

## Roadmap: Data Analysis Automation

- [x] Research toolkit: 61 method guides (SRM, MAA, PDA, MLK, FLOW)
- [x] TimesFM-1.0-200M zero-shot forecast pipeline
- [x] Mermaid skill-graph diagrams for all families
- [ ] Automated validity-condition checking per dataset
- [ ] Scheduled forecast re-runs (GitHub Actions)
- [ ] Auto-generated research reports from templates
- [ ] Integration with additional foundation models

**Built with** TimesFM · Mermaid · Python

### Security — What Gets Pushed

Only essential, human-reviewed files are tracked. Excluded via `.gitignore`:

| Category | Excluded |
|----------|----------|
| Python cache | `__pycache__/`, `*.pyc`, `*.pyo` |
| Environments | `.venv/`, `env/`, `.env`, `*.log` |
| IDE | `.vscode/`, `.idea/`, `.oxlintrc.json` |
| OS junk | `.DS_Store`, `Thumbs.db` |
| Archives | `*.zip`, `*.tar.gz`, `*.gz` |
| AI/cache | `tmp/`, `.cache/`, `hf_timesfm_*/`, `.huggingface/` |
| AI-generated | `*_ai_generated*`, `*_generated.py` |
| Jupyter | `.ipynb_checkpoints/` |

**Tracked files only:** `README.md`, `.gitignore`, `LICENSE`, `requirements.txt`, `data/*.csv`, `analysis/*.py`, `skills/**/*.md`, `skills/mermaid/*.mmd`, `docs/*.mmd`.

### SRM — Statistical Research Methods (22 skills)

```mermaid
flowchart TD
    classDef default fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#01579b,font-size:11px
    classDef group fill:#c5e1a5,stroke:#33691e,stroke-width:3px,color:#33691e,font-size:12px,font-weight:bold

    subgraph SRM["SRM - Statistical Research Methods (22 skills)"]
        direction TB
        s1[srm-research-question-design] --> s2[srm-study-design-selection] --> s3[srm-sampling-methods] --> s4[srm-variable-types-encoding] --> s5[srm-data-collection-protocols] --> s6[srm-numerical-summaries-quantitative] --> s7[srm-numerical-summaries-qualitative] --> s8[srm-graphical-summaries] --> s9[srm-distributions-normal-model] --> s10[srm-confidence-intervals-single] --> s11[srm-confidence-intervals-comparisons] --> s12[srm-hypothesis-testing-framework] --> s13[srm-hypothesis-tests-means] --> s14[srm-hypothesis-tests-categorical] --> s15[srm-correlation] --> s16[srm-linear-regression] --> s17[srm-probability-and-decision-logic] --> s18[srm-confounding-and-blinding-experimental] --> s19[srm-confounding-and-blinding-observational] --> s20[srm-statistical-reference] --> s21[srm-reporting-research] --> s22[srm-correlation-linear-integration]
    end
    class SRM group

## Mermaid Diagram 2: MAA — Meta-Analysis

```markdown
### MAA — Meta-Analysis (7 skills)

```mermaid
flowchart TD
    classDef default fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#4a148c,font-size:11px
    classDef group fill:#c5e1a5,stroke:#33691e,stroke-width:3px,color:#33691e,font-size:12px,font-weight:bold

    subgraph MAA["MAA - Meta-Analysis (7 skills)"]
        direction TB
        m1[maa-effect-sizes] --> m2[maa-pooling-models] --> m3[maa-heterogeneity] --> m4[maa-forest-plots] --> m5[maa-publication-bias] --> m6[maa-subgroup-meta-regression] --> m7[maa-validation-vs-r-stata]
    end
    class MAA group

## Mermaid Diagram 3: PDA — Python Data Analysis

```markdown
### PDA — Python Data Analysis (13 skills)

```mermaid
flowchart TD
    classDef default fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px,color:#1b5e20,font-size:11px
    classDef group fill:#c5e1a5,stroke:#33691e,stroke-width:3px,color:#33691e,font-size:12px,font-weight:bold

    subgraph PDA["PDA - Python Data Analysis (13 skills)"]
        direction TB
        p1[pda-python-language-core] --> p2[pda-numpy-fundamentals] --> p3[pda-numpy-advanced] --> p4[pda-pandas-foundations] --> p5[pda-data-loading-io] --> p6[pda-data-cleaning] --> p7[pda-data-wrangling-reshape] --> p8[pda-groupby-aggregation] --> p9[pda-time-series] --> p10[pda-plotting-visualization] --> p11[pda-modeling-libraries] --> p12[pda-ipython-jupyter-workflow] --> p13[pda-analysis-examples]
    end
    class PDA group

## Mermaid Diagram 4: MLK — Machine Learning

```markdown
### MLK — Machine Learning (13 skills)

```mermaid
flowchart TD
    classDef default fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#e65100,font-size:11px
    classDef group fill:#c5e1a5,stroke:#33691e,stroke-width:3px,color:#33691e,font-size:12px,font-weight:bold

    subgraph MLK["MLK - Machine Learning (13 skills)"]
        direction TB
        l1[mlk-vectors-matrices-arrays] --> l2[mlk-data-loading] --> l3[mlk-data-wrangling] --> l4[mlk-datetime-features] --> l5[mlk-categorical-features] --> l6[mlk-text-features] --> l7[mlk-numerical-features] --> l8[mlk-dim-reduction-extraction] --> l9[mlk-naive-bayes] --> l10[mlk-svm] --> l11[mlk-trees-forests] --> l12[mlk-neural-networks] --> l13[mlk-clustering]
    end
    class MLK group

## Mermaid Diagram 5: FLOW — Workflow

```markdown
### FLOW — Workflow (6 skills)

```mermaid
flowchart TD
    classDef default fill:#fce4ec,stroke:#880e4f,stroke-width:2px,color:#880e4f,font-size:11px
    classDef group fill:#c5e1a5,stroke:#33691e,stroke-width:3px,color:#33691e,font-size:12px,font-weight:bold

    subgraph FLOW["FLOW - Workflow (6 skills)"]
        direction TB
        f1[flow-py-analysis-env] --> f2[flow-unit-hygiene] --> f3[flow-supervisor] --> f4[flow-analysis-pipeline-dunn] --> f5[flow-meta-analysis-pipeline] --> f6[flow-corpus-extraction]
    end
    class FLOW group

## Mermaid Diagram 6: Full Skill Graph — All 61 Skills

```markdown
### Full Skill Graph — All 61 Skills

```mermaid
flowchart TB
    classDef family fill:#e1f5fe,stroke:#01579b,stroke-width:3px,color:#01579b,font-size:13px,font-weight:bold

    subgraph SRM["SRM - Statistical Research Methods (22 skills)"]
        direction TB
        s1[Study Design & Question] --> s2[Sampling Methods] --> s3[Variable Types & Encoding] --> s4[Data Collection] --> s5[Distributions] --> s6[Graphical Summaries] --> s7[CI Single] --> s8[CI Comparisons] --> s9[Hypothesis Framework] --> s10[t-Tests Means] --> s11[Chi-Square Categorical] --> s12[Correlation] --> s13[Linear Regression] --> s14[Probability & Decision] --> s15[Confounding Experimental] --> s16[Confounding Observational] --> s17[Statistical Reference] --> s18[Reporting Research] --> s19[Correlation-Linear Integration]
    end

    subgraph MAA["MAA - Meta-Analysis (7 skills)"]
        direction TB
        m1[Effect Sizes] --> m2[Pooling Models] --> m3[Heterogeneity] --> m4[Forest Plots] --> m5[Publication Bias] --> m6[Subgroup Meta-Regression] --> m7[Validation vs R Stata]
    end

    subgraph PDA["PDA - Python Data Analysis (13 skills)"]
        direction TB
        p1[Python Language Core] --> p2[NumPy Fundamentals] --> p3[NumPy Advanced] --> p4[Pandas Foundations] --> p5[Data Loading IO] --> p6[Data Cleaning] --> p7[Data Wrangling Reshape] --> p8[GroupBy Aggregation] --> p9[Time Series] --> p10[Plotting Visualization] --> p11[Modeling Libraries] --> p12[IPython Jupyter Workflow] --> p13[Analysis Examples]
    end

    subgraph MLK["MLK - Machine Learning (13 skills)"]
        direction TB
        l1[Vectors Matrices Arrays] --> l2[Data Loading] --> l3[Data Wrangling] --> l4[Datetime Features] --> l5[Categorical Features] --> l6[Text Features] --> l7[Numerical Features] --> l8[Dimensionality Reduction] --> l9[Naive Bayes] --> l10[SVM] --> l11[Trees Forests] --> l12[Neural Networks] --> l13[Clustering]
    end

    subgraph FLOW["FLOW - Workflow (6 skills)"]
        direction TB
        f1[Python Analysis Env] --> f2[Unit Hygiene] --> f3[Supervisor] --> f4[Analysis Pipeline Dunn] --> f5[Meta-Analysis Pipeline] --> f6[Corpus Extraction]
    end

    SRM --> MAA
    MAA --> PDA
    PDA --> MLK
    MLK --> FLOW

    class SRM,MAA,PDA,MLK,FLOW family
    linkStyle default stroke-width:2px,fill:none,stroke:#0288d1


