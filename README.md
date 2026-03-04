# Predicting Pediatric Pneumonia Hospitalisation Rates in English Local Authorities

## Project Overview

This project focuses on the development of a machine learning system that predicts pediatric pneumonia hospitalization rates across English Local Authorities using routinely collected NHS Fingertips data. The system analyzes socioeconomic, demographic, and healthcare utilization patterns to identify high-risk areas, aiming to reduce geographic health inequalities and support evidence-based NHS resource allocation.

Three machine learning models—**Linear Regression**, **Lasso Regression**, and **Gradient Boosting**—were trained and evaluated, with performance assessed using R² score, RMSE, and comprehensive statistical validation including regression assumptions testing.

---

## Research Questions

### Primary Research Question
**How can machine learning models (Linear Regression, Lasso Regression, and Gradient Boosting) be utilized to predict emergency pneumonia hospitalisation rates among children and young people under 19 across English Local Authorities using socioeconomic deprivation, demographic vulnerability, and healthcare utilisation indicators?**

### Secondary Research Question
**Which socioeconomic, demographic, and healthcare-related factors contribute most to geographic inequalities in pneumonia hospitalisation rates among children and young people under 19 across England?**

---

## Data Ethics

This project exclusively utilizes publicly available, aggregated data from **NHS Fingertips Public Health Data**, managed by the UK Office for Health Improvement and Disparities (OHID). The dataset contains no personally identifiable information—all data are Local Authority-level aggregates complying with UK data protection regulations.

As the analysis uses only publicly accessible, anonymized aggregate statistics, no ethical approval was required. The research follows University of Hertfordshire ethical guidance and ensures transparency, reproducibility, and responsible use of machine learning in public health settings.

---

## Data Description

- **Source:** NHS Fingertips Public Health Data (Public Health England)
- **Format:** CSV
- **Samples:** 281 English Local Authorities
- **Time Period:** 2021-2024
- **Features:**
  - **Clinical/Healthcare:** Emergency admissions (0-4, under-18), respiratory admissions, A&E attendances
  - **Socioeconomic:** IMD score & quintiles, child poverty, fuel poverty, free school meals
  - **Demographic:** Population indices (0-4, 5-14, 15-19 years), ethnic minority percentage, birth rate, infant mortality
  - **Housing/Geography:** Overcrowded households, population density, urban/rural classification
- **Target:** Pneumonia hospitalization rates (ages 0-19) per 100,000 population

**Total Features:** 18 predictors across 4 categories

---

## Project Structure

1. **Data Collection:** NHS Fingertips API (14 indicator datasets merged)
2. **Data Preparation:**
   - Handling missing values (imputation/exclusion)
   - Feature engineering (log transformations, dummy variables, interaction terms)
   - VIF analysis for multicollinearity assessment
3. **Exploratory Data Analysis (EDA):**
   - Univariate and bivariate analysis
   - Correlation heatmaps and deprivation gradient visualization
   - Urban vs Rural comparison
4. **Model Development:**
   - Linear Regression (baseline)
   - Lasso Regression with L1 regularization (feature selection)
   - Gradient Boosting (ensemble method)
   - 80/20 train-test split with stratification
   - 10-fold cross-validation for hyperparameter optimization
5. **Evaluation:**
   - R² Score, RMSE, MAE, MAPE
   - Regression assumptions testing (Shapiro-Wilk, Breusch-Pagan, Durbin-Watson)
   - Residual diagnostics and scatter plots
6. **Feature Importance Analysis:**
   - Lasso coefficients
   - SHAP values for model-agnostic interpretability
   - Consensus ranking between methods

---

## Key Features

- **Comprehensive Feature Engineering:** Log transformations, dummy encoding, interaction terms (IMD × PopDensity, ChildPoverty × Urban)
- **Multicollinearity Management:** L1 regularization (Lasso) reduced feature space from 18 → 9 non-zero coefficients
- **SHAP Explainability:** Identifies top contributing factors (e.g., Emergency Admissions, Ethnic Minority %)
- **Statistical Rigor:** All regression assumptions validated (normality ✓, homoscedasticity ✓, independence ✓)
- **Model Tuning:** GridSearchCV and RandomizedSearchCV for optimal hyperparameters
- **Performance-Driven:** Lasso Regression selected based on R² = 0.72 and superior feature interpretability

---

## Results

### Model Performance Summary

| Model | Type | CV R² | Test R² | Test RMSE | Test MAE | Key Characteristics |
|-------|------|-------|---------|-----------|----------|---------------------|
| **Lasso Regression** ⭐ | Linear (L1) | 0.62 ± 0.04 | **0.72** | **144.54** | **107.70** | Automatic feature selection, best performance |
| Linear Regression | Baseline | 0.62 | 0.69 | 148.54 | 109.74 | Unregularized baseline |
| Gradient Boosting | Ensemble | 0.61 | 0.63 | 166.07 | 120.05 | Best tree-based model |

### Top 10 Predictive Factors

| Rank | Feature | Category | Lasso Coefficient | SHAP Importance | Effect Direction |
|------|---------|----------|-------------------|-----------------|------------------|
| 1 | Emergency Admissions (0-4 years) | Healthcare | +192.98 | 142.24 | Strong positive |
| 2 | Ethnic Minority Percentage | Demographics | -53.81 | 32.33 | Protective (negative) |
| 3 | Population (0-4 years) Index | Demographics | +45.46 | 29.95 | Positive |
| 4 | IMD Quintile 4 | Socioeconomic | -35.13 | 11.83 | Negative vs Q5 |
| 5 | IMD Quintile 3 | Socioeconomic | -27.68 | 9.35 | Negative vs Q5 |
| 6 | Overcrowded Households (%) | Housing | +19.42 | 7.20 | Positive |
| 7 | Emergency Admissions (Under-18) | Healthcare | +16.89 | 6.89 | Positive |
| 8 | Child Poverty (%) | Socioeconomic | +14.23 | 5.94 | Positive |
| 9 | IMD × PopDensity (interaction) | Composite | +12.47 | 4.88 | Positive |
| 10 | Birth Rate per 1,000 | Demographics | +8.91 | 3.67 | Positive |

### Category-Level Importance

- **Healthcare Utilization:** 60.1%
- **Demographics:** 26.3%
- **Socioeconomic Deprivation:** 13.6%
- **Housing & Geography:** 0.0% (eliminated by Lasso)

---

## Requirements

### Python Version
- Python 3.8+

### Libraries
```
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
scipy >= 1.7.0
statsmodels >= 0.13.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
shap >= 0.40.0
jupyter >= 1.0.0
```

---

### Prepare the Dataset
Ensure the NHS Fingertips dataset is available in the `/data` directory:
```
data/pediatric_pneumonia_england_LA_dataset_2023_2024.csv
```

### Run the Analysis
**Launch Jupyter Notebook:**
```bash
jupyter notebook
```

**Open the main notebook:**
- Navigate to `Pediatric_Pneumonia_AnalysisC.ipynb`
- Run: `Kernel` → `Restart & Run All`

**Expected Runtime:**
- Full notebook execution: ~5-10 minutes
- Hyperparameter tuning: 2-3 minutes
- Visualization generation: 1-2 minutes

### Loading Pre-Trained Models (Optional)
```python
import joblib

# Load saved Lasso model
lasso_model = joblib.load('models/lasso_best_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Make predictions
predictions = lasso_model.predict(X_test_scaled)
```

---

## Key Findings

### 1. Strong Predictive Capability
- Lasso model achieves **R² = 0.72** (72% variance explained)
- Test RMSE = 144.54 per 100,000 population
- All regression assumptions satisfied

### 2. Healthcare Utilization Dominates
- Emergency admissions account for **60.1%** of predictive importance
- Challenges conventional focus on socioeconomic deprivation alone
- Suggests healthcare access barriers as critical intervention points

### 3. Unexpected Protective Effects
- **Higher ethnic minority percentages associated with LOWER pneumonia rates** (coefficient = -53.81)
- Warrants investigation of cultural health practices or healthcare engagement patterns

### 4. Clear Deprivation Gradient
- Stepwise reduction from most deprived (Q5) to least deprived (Q1)
- But effect secondary to healthcare utilization patterns

### 5. Linear Models Superior
- Lasso (R² = 0.72) significantly outperforms Gradient Boosting (R² = 0.63)
- Validates linear relationships and appropriateness of regularization approach

---

## Future Work and Directions

- **Longitudinal Analysis:** Incorporate multi-year trends to establish causal pathways
- **Air Quality Data:** Include pollution metrics as additional predictor
- **Vaccination Coverage:** Integrate immunization rates to assess preventive factors
- **Spatial Regression:** Account for geographic clustering and neighborhood effects
- **Real-Time Monitoring:** Develop early warning system using weekly admission data
- **Clinical Validation:** Collaborate with NHS England for field testing and policy implementation
- **Extension to Other Conditions:** Apply methodology to asthma, bronchiolitis, other pediatric respiratory illnesses
- **Interactive Dashboard:** Deploy Streamlit/Dash interface for NHS commissioners

---

## Acknowledgments

I would like to thank:
- **Dr. Vid Irsic** (University of Hertfordshire) for academic supervision and guidance
- **NHS Fingertips** and **Public Health England** for providing open access to public health data
- **UK Office for Health Improvement and Disparities (OHID)** for maintaining data infrastructure
- **University of Hertfordshire** for academic support and resources
- **scikit-learn** and **SHAP** development teams for excellent machine learning tools
- The **open-source Python community** for foundational data science libraries

---

## Contact

For questions, feedback, or collaboration opportunities, please contact:

**Ifeakandu Uzoegwu**
- 📧 Email: [ultimateozzie@gmail.com]
- 💼 LinkedIn: [Ifeakandu Uzoegwu]

---

## Contributing

Pull requests and issue reports are welcome. Contributions that improve model performance, interpretability, documentation, or clinical applicability are especially encouraged.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
