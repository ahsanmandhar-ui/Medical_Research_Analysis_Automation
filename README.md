# Research Automation Toolkit

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TimesFM](https://img.shields.io/badge/TimesFM-1.0--200M-purple.svg)](https://github.com/google-research/timesfm)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen.svg)](https://github.com/Ahsan543/research-automation-toolkit)

> **Narrative:** Data analysis will be automated in the future.  
> This repository bundles a complete research-and-data-analysis toolkit with 61 specialist method guides, a TimesFM-1.0-200M zero-shot forecasting pipeline, and Mermaid skill-graph diagrams — all structured for reproducible, AI-augmented research automation.

---

## 📊 TimesFM Forecast: Meta-Analysis Publications (2015–2031)

Zero-shot foundation forecast under AI-automation structural shift, generated with **TimesFM-1.0-200M** (CPU-only, bfloat16, <2GB memory).

![Meta-Analysis Publications Forecast](data/meta_analysis_forecast_2027_2031.png)

### Forecast Table

---

## 🎯 Skills Overview

### Skill Families (61 total references)

| Family | Prefix | Count | Focus Area |
|--------|--------|-------|------------|
| **SRM** — Statistical Research Methods | `srm-` | 28 | Study design, sampling, descriptive stats, CI, hypothesis tests (t, chi-square), correlation, regression, confounding & blinding, reporting |

### Mermaid Skill Graphs

<details>
<summary><b>SRM — Statistical Research Methods (28 skills)</b></summary>

```mermaid
%%{init: {'theme': 'base'}}%%
flowchart TD
    classDef default fill:#f9f,stroke:#333,stroke-width:2px
    classDef entry fill:#bbf,stroke:#f66,stroke-width:2px
    classDef analysis fill:#bfb,stroke:#f66,stroke-width:2px

    A[Study Design & Question] --> B[Sampling Methods]
    B --> C[Variable Types & Encoding]
    C --> D[Data Collection Protocols]
    D --> E[Distributions: Normal Model]
    E --> F[Graphical Summaries]
    F --> G[Numerical Summaries: Quantitative]
    G --> H[Numerical Summaries: Qualitative]
    H --> I[Confidence Intervals: Single]
    I --> J[Confidence Intervals: Comparisons]
    J --> K[Hypothesis Testing Framework]
    K --> L[Hypothesis Tests: Means]
    L --> M[Hypothesis Tests: Categorical]
    M --> N[Correlation]
    N --> O[Linear Regression]
    O --> P[Probability & Decision Logic]
    P --> Q[Confounding & Blinding: Experimental]
    Q --> R[Confounding & Blinding: Observational]
    R --> S[Statistical Reference]
    S --> T[Reporting Research]

    A:::entry

<details>
<summary><b>MAA — Meta-Analysis (7 skills)</b></summary>

```mermaid
%%{init: {'theme': 'base'}}%%
flowchart TD
    classDef default fill:#f9f,stroke:#333,stroke-width:2px
    classDef entry fill:#bbf,stroke:#f66,stroke-width:2px
    classDef analysis fill:#bfb,stroke:#f66,stroke-width:2px

    A[Effect Sizes] --> B[Pooling Models]
    B --> C[Heterogeneity]
    C --> D[Forest Plots]
    D --> E[Publication Bias]
    E --> F[Subgroup & Meta-Regression]
    F --> G[Validation vs R/Stata]

    A:::entry

<details>
<summary><b>PDA — Python Data Analysis (13 skills)</b></summary>

```mermaid
%%{init: {'theme': 'base'}}%%
flowchart TD
    classDef default fill:#f9f,stroke:#333,stroke-width:2px
    classDef entry fill:#bbf,stroke:#f66,stroke-width:2px
    classDef analysis fill:#bfb,stroke:#f66,stroke-width:2px

    A[Python Language Core] --> B[NumPy Fundamentals]
    B --> C[NumPy Advanced]
    C --> D[Pandas Foundations]
    D --> E[Data Loading / IO]
    E --> F[Data Cleaning]
    F --> G[Data Wrangling / Reshape]
    G --> H[GroupBy & Aggregation]
    H --> I[Time Series]
    I --> J[Plotting & Visualization]
    J --> K[Modeling Libraries]
    K --> L[IPython / Jupyter Workflow]
    L --> M[Analysis Examples]

    A:::entry

<details>
<summary><b>MLK — Machine Learning (13 skills)</b></summary>

```mermaid
%%{init: {'theme': 'base'}}%%
flowchart TD
    classDef default fill:#f9f,stroke:#333,stroke-width:2px
    classDef entry fill:#bbf,stroke:#f66,stroke-width:2px
    classDef analysis fill:#bfb,stroke:#f66,stroke-width:2px

    A[Vectors / Matrices / Arrays] --> B[Data Loading]
    B --> C[Data Wrangling]
    C --> D[Datetime Features]
    D --> E[Categorical Features]
    E --> F[Text Features]
    F --> G[Numerical Features]
    G --> H[Dimensionality Reduction & Extraction]
    H --> I[Naive Bayes]
    I --> J[SVM]
    J --> K[Trees & Forests]
    K --> L[Neural Networks]
    L --> M[Clustering]

    A:::entry

<details>
<summary><b>FLOW — Workflow (6 skills)</b></summary>

```mermaid
%%{init: {'theme': 'base'}}%%
flowchart TD
    classDef default fill:#f9f,stroke:#333,stroke-width:2px
    classDef entry fill:#bbf,stroke:#f66,stroke-width:2px
    classDef analysis fill:#bfb,stroke:#f66,stroke-width:2px

    A[Python Analysis Env] --> B[Unit Hygiene]
    B --> C[Supervisor]
    C --> D[Analysis Pipeline: Dunn]
    D --> E[Meta-Analysis Pipeline]
    E --> F[Corpus Extraction]

    A:::entry
    F:::analysis

    linkStyle default stroke-width:2px,fill:none,stroke:#666
```
</details>

<details>
<summary><b>Full Skill Graph — All 61 Skills</b></summary>

```mermaid
%%{init: {'theme': 'base'}}%%
flowchart TB
    subgraph SRM["SRM — Statistical Research Methods (28)"]
        direction TB
        s1[Study Design] --> s2[Sampling] --> s3[Variables] --> s4[Distributions] --> s5[CI & Hypothesis] --> s6[Regression] --> s7[Reporting]
    end

    subgraph MAA["MAA — Meta-Analysis (7)"]
        direction TB
        m1[Effect Sizes] --> m2[Pooling] --> m3[Heterogeneity] --> m4[Forest Plots] --> m5[Publication Bias] --> m6[Subgroup] --> m7[Validation]
    end

    subgraph PDA["PDA — Python Data Analysis (13)"]
        direction TB
        p1[Python Core] --> p2[NumPy] --> p3[Pandas] --> p4[Data IO] --> p5[Cleaning] --> p6[Wrangling] --> p7[GroupBy] --> p8[Time Series] --> p9[Plotting] --> p10[Modeling] --> p11[Jupyter] --> p12[Examples]
    end

    subgraph MLK["MLK — Machine Learning (13)"]
        direction TB
        l1[Linear Algebra] --> l2[Data Loading] --> l3[Features] --> l4[Dimensionality] --> l5[Classification] --> l6[Trees] --> l7[Neural Nets] --> l8[Clustering]
    end

    subgraph FLOW["FLOW — Workflow (6)"]
        direction TB

---

## 🏗️ Architecture

```mermaid
%%{init: {'theme': 'base'}}%%
flowchart TD
    classDef data fill:#bbf,stroke:#333,stroke-width:2px
    classDef process fill:#bfb,stroke:#333,stroke-width:2px
    classDef model fill:#fbb,stroke:#333,stroke-width:2px
    classDef output fill:#bfb,stroke:#333,stroke-width:2px

    A[CSV: meta_analysis_publications_2015_2026.csv]:::data
    B[timesfm_forecast_v2.py]:::process
    C[TimesFM 1.0-200M Model]:::model
    D[Forecast CSV 2027-2031]:::output
    E[Forecast PNG]:::output
    F[Research Toolkit Skills (61)]:::process
    G[Mermaid Diagrams]:::output
    H[README.md UI]:::output

    A --> B
    B --> C
    C --> D
    C --> E
    F --> G
    B --> H
    G --> H
    D --> H
    E --> H

    linkStyle default stroke-width:2px,fill:none,stroke:#666
```

---

## 🚀 Quick Start

### 1. Clone

```bash
git clone https://github.com/Ahsan543/research-automation-toolkit.git
cd research-automation-toolkit
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the TimesFM Forecast

```bash
python analysis/timesfm_forecast_v2.py

---

## 🔒 Security — What Gets Pushed

Only essential, human-reviewed files are tracked in this repository. The following are **excluded** via `.gitignore`:

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

---

## 📂 Repository Structure

```
research-automation-toolkit/
├── README.md                          # This file — UI with badges, narrative, embedded PNG
├── .gitignore                         # Security: excludes AI junk, cache, env, etc.
├── LICENSE                            # MIT
├── requirements.txt                   # Python dependencies
├── data/
│   └── meta_analysis_publications_2015_2026.csv   # Historical data (2015–2026)
├── analysis/
│   └── timesfm_forecast_v2.py         # TimesFM 1.0-200M zero-shot forecast script
├── skills/
│   ├── research-toolkit/
│   │   ├── SKILL.md                   # Routing guide for 61 method references
│   │   └── references/
│   │       ├── srm/   (28 files)
│   │       ├── maa/   (7 files)
│   │       ├── pda/   (13 files)
│   │       ├── mlk/   (13 files)
│   │       └── flow/  (6 files)
│   └── mermaid/
│       ├── srm-skill-graph.mmd
│       ├── maa-skill-graph.mmd
│       ├── pda-skill-graph.mmd
│       ├── mlk-skill-graph.mmd
│       ├── flow-skill-graph.mmd
│       └── full-skill-graph.mmd
└── docs/
    └── architecture.mmd               # High-level automation architecture
```

---

## 🔮 Roadmap: Data Analysis Automation

- [x] Research toolkit: 61 method guides (SRM, MAA, PDA, MLK, FLOW)
- [x] TimesFM-1.0-200M zero-shot forecast pipeline
- [x] Mermaid skill-graph diagrams for all families
- [ ] Automated validity-condition checking per dataset
- [ ] Scheduled forecast re-runs (GitHub Actions)
- [ ] Auto-generated research reports from templates
- [ ] Integration with additional foundation models

---

**Built with** [TimesFM](https://github.com/google-research/timesfm) · [Mermaid](https://mermaid.js.org/) · [Python](https://www.python.org/)
```

This generates:
- `data/meta_analysis_forecast_2027_2031.csv` — forecast values
- `data/meta_analysis_forecast_2027_2031.png` — forecast chart
- `data/forecast_table.md` — forecast table markdown

### 4. Explore the Research Toolkit

```bash
ls skills/research-toolkit/references/srm/
ls skills/research-toolkit/references/maa/
ls skills/research-toolkit/references/pda/
ls skills/research-toolkit/references/mlk/
ls skills/research-toolkit/references/flow/
```

Read `skills/research-toolkit/SKILL.md` for the full routing guide.
        f1[Env] --> f2[Hygiene] --> f3[Supervisor] --> f4[Pipeline] --> f5[Meta-Analysis Pipeline] --> f6[Corpus Extraction]
    end

    SRM --> MAA
    MAA --> PDA
    PDA --> MLK
    MLK --> FLOW

    classDef family fill:#f9f,stroke:#333,stroke-width:3px
    class SRM,MAA,PDA,MLK,FLOW family

    linkStyle default stroke-width:2px,fill:none,stroke:#666
```
</details>
    M:::analysis

    linkStyle default stroke-width:2px,fill:none,stroke:#666
```
</details>
    M:::analysis

    linkStyle default stroke-width:2px,fill:none,stroke:#666
```
</details>
    G:::analysis

    linkStyle default stroke-width:2px,fill:none,stroke:#666
```
</details>
    S:::analysis
    T:::analysis

    linkStyle default stroke-width:2px,fill:none,stroke:#666
```
</details>
| **MAA** — Meta-Analysis | `maa-` | 7 | Effect sizes, forest plots, heterogeneity, pooling models, publication bias, subgroup meta-regression, validation vs R/Stata |
| **PDA** — Python Data Analysis | `pda-` | 13 | Data loading/IO, cleaning, wrangling/reshape, groupby, time series, plotting, NumPy, pandas, Jupyter |
| **MLK** — Machine Learning | `mlk-` | 13 | Categorical/text/numerical features, clustering, dim reduction (PCA/t-SNE), naive Bayes, SVM, trees/forests, neural networks |
| **FLOW** — Workflow | `flow-` | 6 | Analysis pipeline, corpus extraction, meta-analysis pipeline, Python env, supervisor, unit hygiene |

| Year | Point Forecast (mean) | 10th pct (worst-case decline) | 90th pct (best-case stability) |
|------|----------------------|-------------------------------|--------------------------------|
| 2027 | — | — | — |
| 2028 | — | — | — |
| 2029 | — | — | — |
| 2030 | — | — | — |
| 2031 | — | — | — |

> Run `python analysis/timesfm_forecast_v2.py` to generate the forecast CSV, table, and PNG. Outputs land in `data/`.