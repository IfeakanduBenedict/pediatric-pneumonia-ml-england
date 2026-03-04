Predicting Pediatric Pneumonia Hospitalisation Rates in English Local Authorities
Project Overview
This project focuses on the development of a machine learning system that predicts pediatric pneumonia hospitalization rates across English Local Authorities using routinely collected NHS Fingertips data. The system analyzes socioeconomic, demographic, and healthcare utilization patterns to identify high-risk areas, aiming to reduce geographic health inequalities and support evidence-based NHS resource allocation.
Three machine learning models—Linear Regression, Lasso Regression, and Gradient Boosting—were trained and evaluated, with performance assessed using R² score, RMSE, and comprehensive statistical validation including regression assumptions testing.

Research Questions
Primary Research Question
How can machine learning models (Linear Regression, Lasso Regression, and Gradient Boosting) be utilized to predict emergency pneumonia hospitalisation rates among children and young people under 19 across English Local Authorities using socioeconomic deprivation, demographic vulnerability, and healthcare utilisation indicators?
Secondary Research Question
Which socioeconomic, demographic, and healthcare-related factors contribute most to geographic inequalities in pneumonia hospitalisation rates among children and young people under 19 across England?

Data Ethics
This project exclusively utilizes publicly available, aggregated data from NHS Fingertips Public Health Data, managed by the UK Office for Health Improvement and Disparities (OHID). The dataset contains no personally identifiable information—all data are Local Authority-level aggregates complying with UK data protection regulations.
As the analysis uses only publicly accessible, anonymized aggregate statistics, no ethical approval was required. The research follows University of Hertfordshire ethical guidance and ensures transparency, reproducibility, and responsible use of machine learning in public health settings.

Data Description

Source: NHS Fingertips Public Health Data (Public Health England)
Format: CSV
Samples: 281 English Local Authorities
Time Period: 2021-2024
Features:

Clinical/Healthcare: Emergency admissions (0-4, under-18), respiratory admissions, A&E attendances
Socioeconomic: IMD score & quintiles, child poverty, fuel poverty, free school meals
Demographic: Population indices (0-4, 5-14, 15-19 years), ethnic minority percentage, birth rate, infant mortality
Housing/Geography: Overcrowded households, population density, urban/rural classification


Target: Pneumonia hospitalization rates (ages 0-19) per 100,000 population

Total Features: 18 predictors across 4 categories

Project Structure

Data Collection: NHS Fingertips API (14 indicator datasets merged)
Data Preparation:

Handling missing values (imputation/exclusion)
Feature engineering (log transformations, dummy variables, interaction terms)
VIF analysis for multicollinearity assessment


Exploratory Data Analysis (EDA):

Univariate and bivariate analysis
Correlation heatmaps and deprivation gradient visualization
Urban vs Rural comparison


Model Development:

Linear Regression (baseline)
Lasso Regression with L1 regularization (feature selection)
Gradient Boosting (ensemble method)
80/20 train-test split with stratification
10-fold cross-validation for hyperparameter optimization


Evaluation:

R² Score, RMSE, MAE, MAPE
Regression assumptions testing (Shapiro-Wilk, Breusch-Pagan, Durbin-Watson)
Residual diagnostics and scatter plots


Feature Importance Analysis:

Lasso coefficients
SHAP values for model-agnostic interpretability
Consensus ranking between methods




Key Features

Comprehensive Feature Engineering: Log transformations, dummy encoding, interaction terms (IMD × PopDensity, ChildPoverty × Urban)
Multicollinearity Management: L1 regularization (Lasso) reduced feature space from 18 → 9 non-zero coefficients
SHAP Explainability: Identifies top contributing factors (e.g., Emergency Admissions, Ethnic Minority %)
Statistical Rigor: All regression assumptions validated (normality ✓, homoscedasticity ✓, independence ✓)
Model Tuning: GridSearchCV and RandomizedSearchCV for optimal hyperparameters
Performance-Driven: Lasso Regression selected based on R² = 0.72 and superior feature interpretability


Results
Model Performance Summary
ModelTypeCV R²Test R²Test RMSETest MAEKey CharacteristicsLasso Regression ⭐Linear (L1)0.62 ± 0.040.72144.54107.70Automatic feature selection, best performanceLinear RegressionBaseline0.620.69148.54109.74Unregularized baselineGradient BoostingEnsemble0.610.63166.07120.05Best tree-based model
Top 10 Predictive Factors
RankFeatureCategoryLasso CoefficientSHAP ImportanceEffect Direction1Emergency Admissions (0-4 years)Healthcare+192.98142.24Strong positive2Ethnic Minority PercentageDemographics-53.8132.33Protective (negative)3Population (0-4 years) IndexDemographics+45.4629.95Positive4IMD Quintile 4Socioeconomic-35.1311.83Negative vs Q55IMD Quintile 3Socioeconomic-27.689.35Negative vs Q56Overcrowded Households (%)Housing+19.427.20Positive7Emergency Admissions (Under-18)Healthcare+16.896.89Positive8Child Poverty (%)Socioeconomic+14.235.94Positive9IMD × PopDensity (interaction)Composite+12.474.88Positive10Birth Rate per 1,000Demographics+8.913.67Positive
Category-Level Importance

Healthcare Utilization: 60.1%
Demographics: 26.3%
Socioeconomic Deprivation: 13.6%
Housing & Geography: 0.0% (eliminated by Lasso)
