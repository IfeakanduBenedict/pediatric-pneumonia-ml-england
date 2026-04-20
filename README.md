# Predicting Pediatric Pneumonia Emergency Hospitalisation Rates Across English Local Authorities Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![SHAP](https://img.shields.io/badge/SHAP-Interpretability-green.svg)
![Licence](https://img.shields.io/badge/Licence-Academic-lightgrey.svg)

MSc Data Science project applying interpretable machine learning to predict pediatric pneumonia emergency hospitalisation rates across 281 English Local Authorities using NHS Fingertips data (2023/24).

---

## Overview

Pediatric pneumonia is a major driver of emergency hospital admissions in England, with substantial geographic variation across Local Authorities. This project builds and validates a predictive modelling framework using regularised regression and tree-based methods to:

- Quantify geographic inequalities in paediatric pneumonia burden
- Identify the most important socioeconomic, demographic, and healthcare-utilisation drivers
- Provide NHS England with an interpretable tool for evidence-based resource allocation

The final model (**Lasso Regression**) explains **72% of the geographic variance** in pediatric pneumonia rates (Test R² = 0.7172) while meeting all standard regression assumptions.

---

## Key Findings

| Metric | Value |
|---|---|
| Local Authorities analysed | 281 |
| Features used | 18 (reduced to 9 by Lasso) |
| Best model | Lasso Regression (alpha = 5.0) |
| Test R² | **0.7172** (71.72% variance explained) |
| Test RMSE | 144.54 per 100,000 population |
| Top predictor | Emergency admissions, ages 0–4 (r = 0.79) |
| Deprivation gradient | 22.3% higher rates in most vs least deprived |
| Feature importance method agreement | Spearman ρ = 1.000 |

**Headline insight:** Healthcare utilisation patterns dominate as predictors (88% of model importance), but socioeconomic deprivation operates indirectly through healthcare-access pathways rather than disappearing as a factor.

---

## Research Questions

**Primary:** To what extent can machine learning models predict emergency pneumonia hospitalisation rates among children and young people under 19 across English Local Authorities using socioeconomic, demographic, and healthcare-utilisation indicators?

**Secondary:** Which socioeconomic, demographic, and healthcare-related factors contribute most to geographic inequalities in pneumonia hospitalisation rates among children and young people under 19 across England?

---

## Dataset

- **Source:** [NHS Fingertips](https://fingertips.phe.org.uk/) (UKHSA / NHS England)
- **Reference period:** 2023/24
- **Sample:** 281 English Local Authorities (District and Unitary Authority level)
- **Target:** Emergency pneumonia hospitalisation rate per 100,000 population, under-19s
- **Features:** 11 raw predictors expanded to 18 after feature engineering, spanning four domains:
  - Socioeconomic deprivation (IMD, child poverty, fuel poverty, free school meals)
  - Demographic vulnerability (age structure, ethnicity, birth rate)
  - Housing & geography (population density, overcrowding)
  - Healthcare system utilisation (infant mortality, emergency admissions)

Data are aggregated at area level with no individual-level identifiers. Formal ethical approval was not required as the data are publicly available.

---

## Methodology

The analysis follows a structured machine learning pipeline:

1. **Data preparation** — 282 Local Authorities imported, one observation with missing outcome removed (n = 281), log-transformation applied to skewed population density
2. **Exploratory data analysis** — univariate, bivariate, correlation heatmaps, deprivation gradient, urban–rural stratification
3. **Feature engineering** — IMD quintile dummies, urban binary indicator, interaction terms
4. **Multicollinearity assessment** — Variance Inflation Factor (mean VIF = 103.39, 15/18 features with VIF > 10)
5. **Modelling** — three models compared on 80/20 train–test split:
   - Linear Regression (baseline)
   - **Lasso Regression** (L1 regularisation) — selected as final model
   - Gradient Boosting (ensemble benchmark)
6. **Hyperparameter optimisation** — GridSearchCV for Lasso (10-fold CV), RandomizedSearchCV for Gradient Boosting (5-fold CV)
7. **Regression diagnostics** — Shapiro–Wilk, Breusch–Pagan, Durbin–Watson (all assumptions satisfied)
8. **Feature importance triangulation** — Lasso coefficients, SHAP values, permutation importance (all methods agreed with Spearman ρ = 1.000)
9. **Sensitivity analysis (Model B)** — re-ran Lasso excluding healthcare-utilisation features to isolate upstream predictors (Test R² = 0.49 with only demographic features retained)

---

## Results Summary

### Model Comparison

| Model | Train R² | Test R² | CV R² (± SD) | Test RMSE | Overfitting Gap |
|---|---|---|---|---|---|
| Linear Regression | 0.6931 | 0.6896 | 0.6156 (±0.17) | 151.42 | 0.0034 |
| **Lasso Regression**  | 0.6755 | **0.7172** | 0.6241 (±0.16) | **144.54** | -0.0417 |
| Gradient Boosting | 0.9121 | 0.6267 | 0.6094 (±0.17) | 166.07 | 0.2854 |

The Gradient Boosting loss curves revealed classic overfitting — training loss fell to 5,019 while test loss plateaued at 27,578 (5.5× gap), confirming Lasso's suitability given the modest sample size.

### Feature Importance by Category

| Category | Importance |
|---|---|
| Healthcare Utilisation | 88.25% |
| Demographics | 11.28% |
| Socioeconomic Deprivation | 0.47% |
| Housing & Geography | 0.00% |

All three importance methods (Lasso coefficients, SHAP, permutation) converged on the same ranking with Spearman ρ = 1.000.
---

## How to Reproduce

### Requirements

```bash
python >= 3.10
pandas
numpy
scikit-learn
statsmodels
matplotlib
seaborn
scipy
shap
joblib
```

Install via:

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn scipy shap joblib
```

### Running the Notebook

1. Clone the repository:
   ```bash
   git clone https://github.com/IfeakanduBenedict/pediatric-pneumonia-ml-england.git
   cd pediatric-pneumonia-ml-england
   ```

2. Open the notebook in Google Colab or Jupyter:
   ```bash
   jupyter notebook Pediatric_Pneumonia_Analysis.ipynb
   ```

3. Run all cells in order. The notebook reproduces the full pipeline end-to-end.

A fixed random seed (`RANDOM_STATE = 42`) is used throughout to ensure reproducibility.

---

## Key Visualisations

The notebook generates the following figures, all referenced in the dissertation:

- Figure 4.1 — Target variable distribution (histogram, box plot, Q-Q plot)
- Figure 4.2 — Correlation matrix heatmap
- Figure 4.3 — Pneumonia rates by IMD quintile
- Figure 4.4 — VIF multicollinearity assessment
- Figure 4.5 — Lasso actual vs predicted (with 95% prediction interval)
- Figure 4.6 — Four-panel model performance comparison
- Figure 4.7 — **Gradient Boosting loss curves (overfitting diagnostic)**
- Figure 4.8 — Lasso regression diagnostic plots
- Figure 4.9 — Top 10 features by Lasso coefficient
- Figure 4.10 — SHAP beeswarm and bar plots
- Figure 4.11 — Permutation importance rankings
- Figure 4.12 — Feature importance by category

---

## Policy Implications

The study supports a **dual strategy** for addressing paediatric pneumonia inequalities:

1. **Strengthening primary care access** in deprived areas, particularly for families with young children — high emergency admission rates likely reflect blocked GP pathways rather than unavoidable clinical need
2. **Long-term social determinants** interventions aligned with NHS England's Core20PLUS5 framework and the Marmot Review's recommendations

The validated predictive model offers NHS England a practical tool for evidence-based geographic risk stratification and resource allocation using routinely collected data.

---

## Limitations

- Cross-sectional ecological design precludes causal inference
- Area-level aggregation means findings inform planning, not individual risk
- Unmeasured confounders (air quality, vaccination coverage, GP accessibility) likely explain much of the 28% unexplained variance
- Sample size of 281 limits complex model fitting and stability of estimates
- Standard k-fold cross-validation does not fully account for spatial autocorrelation (though Durbin–Watson = 1.92 suggested minimal impact)

---

## Citation

If you use this work, please cite:

> Uzoegwu, I. (2026). *Predicting Paediatric Pneumonia Emergency Hospitalisation Rates Across English Local Authorities Using Machine Learning*. MSc Data Science Dissertation, University of Hertfordshire.

---

## Author

**Ifeakandu Uzoegwu**
MSc Data Science — University of Hertfordshire
Student Number: 23068196

**Supervisor:** Hyungrok Kim
**Module Supervisors:** Darshan Kakkad, Carolyn Devereux

**Department of Physics, Astronomy and Mathematics**
**School of Physics, Engineering and Computer Science**

---

## Acknowledgements

This project used data from NHS Fingertips (UKHSA / NHS England), an open public health data repository. Thanks to the open-source community for the libraries that made this analysis possible: scikit-learn, SHAP, pandas, statsmodels, and matplotlib.

---

## Licence

This repository is released for academic and non-commercial use. Please credit the author when reusing materials.
