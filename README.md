<!-- Bio -->
<p align="center">
  <img src="https://github.com/Rudra-G-23/breast-cancer-prediction-app/blob/main/assets/rudra.png?raw=true" alt="rudra" width="300"/>
</p>

<!-- refer this for animation "https://github.com/DenverCoder1/readme-typing-svg?tab=readme-ov-file" -->
<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=30&duration=6000&pause=1000&color=5E17EB&center=true&vCenter=true&width=435&lines=Rudra+Prasad+Bhuyan;Data+Lover;Data+Science+Enthusiast" alt="Typing Effect" />
</p>

<!-- refer this for the badges "https://github.com/Ileriayo/markdown-badges#-developerforums" -->
<p align="center">
  <a href="https://github.com/Rudra-G-23">
    <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Badge"/>
  </a>
  <a href="https://www.linkedin.com/in/rudra-prasad-bhuyan-44a388235">
    <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn Badge"/>
  </a>
  <a href="https://www.kaggle.com/rudraprasadbhuyan">
    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="Kaggle Badge"/>
  </a>
</p>

---

<!-- Badges -->
<p align="center">
  <a href="https://pypi.org/project/numpy/">
    <img src="https://img.shields.io/badge/numpy-2.2.4-blue?style=for-the-badge&logo=numpy" alt="numpy">
  </a>
  <a href="https://pypi.org/project/pandas/">
    <img src="https://img.shields.io/badge/pandas-2.2.3-yellow?style=for-the-badge&logo=pandas" alt="pandas">
  </a>
  <a href="https://pypi.org/project/plotly/">
    <img src="https://img.shields.io/badge/plotly-6.0.1-orange?style=for-the-badge&logo=plotly" alt="plotly">
  </a>
  <a href="https://pypi.org/project/scikit-learn/">
    <img src="https://img.shields.io/badge/scikit--learn-1.6.1-green?style=for-the-badge&logo=scikit-learn" alt="scikit-learn">
  </a>
  <a href="https://pypi.org/project/streamlit/">
    <img src="https://img.shields.io/badge/streamlit-1.44.1-red?style=for-the-badge&logo=streamlit" alt="streamlit">
  </a>
  <a href="https://pypi.org/project/seaborn/">
    <img src="https://img.shields.io/badge/Seaborn-0.13.2-blue?style=for-the-badge&logo=seaborn" alt="Seaborn version">
  </a>
  <a href="https://pypi.org/project/scipy/">
    <img src="https://img.shields.io/badge/SciPy-1.15.2-blue?style=for-the-badge&logo=scipy" alt="SciPy version">
  </a>
  <a href="https://pypi.org/project/mlxtend/">
    <img src="https://img.shields.io/badge/mlxtend-0.23.4-blue?style=for-the-badge&logo=mlxtend" alt="mlxtend version">
  </a>
  <a href="https://pypi.org/project/matplotlib/">
    <img src="https://img.shields.io/badge/Matplotlib-3.10.1-blue?style=for-the-badge&logo=matplotlib" alt="Matplotlib version">
  </a>
  <a href="https://pypi.org/project/dtreeviz/">
    <img src="https://img.shields.io/badge/DTreeViz-2.2.2-blue?style=for-the-badge&logo=python" alt="DTreeViz version">
  </a>
</p>

---

## 📘 Table of Contents

- [3. Feature Engineering](#3-feature-engineering)
  - [3.1 🔧 Feature Transformation](#31--feature-transformation)
    - [3.1.1 📌 Prerequisite](#311--prerequisite)
    - [3.1.2 🔧 Encoding Categorical and Numerical Data](#312--encoding-categorical-and-numerical-data)
    - [3.1.3 📏 Feature Scaling](#313--feature-scaling)
    - [3.1.4 🧩 Handling Missing Data](#314--handling-missing-data)
    - [3.1.5🚨 Handling Outliers](#315-handling-outliers)
  - [3.2 🏗️ Feature Construction](#32-️-feature-construction)
  - [3.3 🔍 Feature Extraction](#33--feature-extraction)
- [4. 📊  Regression](#4---regression)
- [5. 🧑‍💻 Gradient Descent](#5--gradient-descent)
- [6. 👮 Regularization](#6--regularization)
- [7. 📘 Logistic Regression](#7--logistic-regression)
- [8. 🌴 Decision Tree](#8--decision-tree)
- [9.🌋 Voting Ensemble Learning](#9-voting-ensemble-learning)
- [10. 🛍️ Bagging Ensemble Learning](#10-️-bagging-ensemble-learning)

---
# 3. Feature Engineering  

## 3.1 🔧 Feature Transformation


### 3.1.1 📌 Prerequisite

| Topic                   | What You'll Learn                                | Notebook | Lecture |
|-------------------------|--------------------------------------------------|------|-----|
| What is Feature Engineering | –                                              | –    | [🔥](https://youtu.be/sluoVhT0ehg?si=AGLgJFKSC-2f-NFQ) |
| Column Transformer      | How to transform columns                         | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/003-feature-engineering/prerequisite/D028-column-transformer/d028-column-transformer.ipynb) | [🔥](https://youtu.be/5TVj6iEBR4I?si=mj1_85nzrGZm8vQ-) |
| Sklearn without Pipeline | Why avoiding pipelines can cause problems       | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/tree/main/003-feature-engineering/prerequisite/D029-sklearn-pipelines/d028-1-without-pipeline) | [🔥](https://youtu.be/xOccYkgRV4Q?si=3kGjRUE0I3YNu5Xk) |
| Sklearn with Pipeline   | How to implement sklearn pipelines effectively  | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/tree/main/003-feature-engineering/prerequisite/D029-sklearn-pipelines/d028-2-with-pipeline) | [🔥](https://youtu.be/xOccYkgRV4Q?si=3kGjRUE0I3YNu5Xk) |




### 3.1.2 🔧 Encoding Categorical and Numerical Data

| Topic                     | What You'll Learn                                                    | Notebook | Lecture |
|---------------------------|----------------------------------------------------------------------|------|-----|
| Ordinal Encoding          | Ordinal categorical data preprocessing using `OrdinalEncoder()`     | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/tree/main/003-feature-engineering/feature-transformation/encoding-numerical-and-categorical-features/D026-ordinal-encoding) | [🔥](https://youtu.be/w2GglmYHfmM?si=W0wBWKrHsJvS5fcn) |
| One Hot Encoding          | Nominal categorical data preprocessing using `OneHotEncoder()`      | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/tree/main/003-feature-engineering/feature-transformation/encoding-numerical-and-categorical-features/D027-one-hot-encoding) | [🔥](https://youtu.be/U5oCv3JKWKA?si=_5nasUH0Dwr6DcH3) |
| Function Transformer      | Log, reciprocal transformation using `FunctionTransformer()`        | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/tree/main/003-feature-engineering/feature-transformation/encoding-numerical-and-categorical-features/D030-function-transformer) | [🔥](https://youtu.be/cTjj3LE8E90?si=rJLBINYfwQulzmuu) |
| Power Transformer         | Square, square root transformation using `PowerTransformer()`       | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/tree/main/003-feature-engineering/feature-transformation/encoding-numerical-and-categorical-features/D031-power-transformer) | [🔥](https://youtu.be/lV_Z4HbNAx0?si=vASsNWI4cdGwxc5A) |
| Binarization              | Preprocessing with `Binarizer()`                                    | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/003-feature-engineering/feature-transformation/encoding-numerical-and-categorical-features/D032-binning-and-binarization/d032-binarization.ipynb) | [🔥](https://youtu.be/kKWsJGKcMvo?si=L8wzXb_FktlxLvlB) |
| Binning                   | Preprocessing with `KBinsDiscretizer()`                             | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/003-feature-engineering/feature-transformation/encoding-numerical-and-categorical-features/D032-binning-and-binarization/d032-binning.ipynb) | [🔥](https://youtu.be/kKWsJGKcMvo?si=L8wzXb_FktlxLvlB) |
| Handling Mixed Variables  | Processing datasets with both numerical & categorical features      | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/tree/main/003-feature-engineering/feature-transformation/encoding-numerical-and-categorical-features/D033-handling-mixed-variables) | [🔥](https://youtu.be/9xiX-I5_LQY?si=7_fsUHCnuplV04dS) |
| Handling Date & Time      | How to work with time and date columns                              | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/tree/main/003-feature-engineering/feature-transformation/encoding-numerical-and-categorical-features/D034-handling-date-and-time) | [🔥](https://youtu.be/J73mvgG9fFs?si=9eC5fZHX0_LUFS-A) |



### 3.1.3 📏 Feature Scaling

| Topic           | What You'll Learn                                | Notebook | Lecture |
|------------------|--------------------------------------------------|------|-----|
| Standardization  | Preprocessing using `StandardScaler()`           | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/tree/main/003-feature-engineering/feature-transformation/feature-scaling/D024-standardization) | [🔥](https://youtu.be/1Yw9sC0PNwY?si=eb12zqBs0EWNWa84) |
| Normalization    | Preprocessing using `MinMaxScaler()`             | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/tree/main/003-feature-engineering/feature-transformation/feature-scaling/D025-normalization) | [🔥](https://youtu.be/eBrGyuA2MIg?si=x2cqL2Fllo1x3Zfh) |




### 3.1.4 🧩 Handling Missing Data

| Topic                                     | What You'll Learn                                 | Notebook | Lecture |
|-------------------------------------------|---------------------------------------------------|------|-----|
| Complete Case Analysis                    | Remove `NaN` values                               | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/tree/main/003-feature-engineering/feature-transformation/handling-missing-data/D035-complete-case-analysis) | [🔥](https://youtu.be/aUnNWZorGmk?si=iBqblSFCAIuDtg2G) |
| Arbitrary Value Imputation (Numerical)    | Impute with arbitrary value using `SimpleImputer()` | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/003-feature-engineering/feature-transformation/handling-missing-data/D036-imputing-numerical-data/d036-arbitrary-value-imputation.ipynb) | [🔥](https://youtu.be/mCL2xLBDw8M?si=7Uk6LvbVPlpbvGr9) |
| Mean/Median Imputation (Numerical)        | Impute with mean/median using `SimpleImputer()`  | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/003-feature-engineering/feature-transformation/handling-missing-data/D036-imputing-numerical-data/d036-mean-median-imputation.ipynb) | [🔥](https://youtu.be/mCL2xLBDw8M?si=7Uk6LvbVPlpbvGr10) |
| Missing Category Imputation (Categorical) | Fill missing with a label using `SimpleImputer()` | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/003-feature-engineering/feature-transformation/handling-missing-data/D037-handling-missing-categorical-data/d037-missing-category-imputation.ipynb) | [🔥](https://youtu.be/l_Wip8bEDFQ?si=RSP9yS-FoPJK5LpQ) |
| Frequent Value Imputation (Categorical)   | Replace missing with most frequent value          | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/003-feature-engineering/feature-transformation/handling-missing-data/D037-handling-missing-categorical-data/d037-frequent-value-imputation.ipynb) | [🔥](https://youtu.be/l_Wip8bEDFQ?si=RSP9yS-FoPJK5LpQ) |
| Missing Indicator                         | Add binary flag for missing values (`MissingIndicator()`) | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/003-feature-engineering/feature-transformation/handling-missing-data/D038-missing-indicator/d038-missing-indicator.ipynb) | [🔥](https://youtu.be/Ratcir3p03w?si=Wrc6ueG9uEHOEWeq) |
| Auto Imputer Parameter Tuning             | Use `GridSearchCV()` to optimize imputer settings | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/003-feature-engineering/feature-transformation/handling-missing-data/D038-missing-indicator/d038-automatically-select-imputer-parameters.ipynb) | [🔥](https://youtu.be/Ratcir3p03w?si=qjEvlueDAqWxoJwI) |
| Random Sample Imputation                  | Fill missing values with random samples           | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/003-feature-engineering/feature-transformation/handling-missing-data/D038-missing-indicator/d038-random-sample-imputation.ipynb) | [🔥](https://youtu.be/Ratcir3p03w?si=5E0EGZcAta_zlTHQ) |
| KNN Imputer                               | Use K-Nearest Neighbors to fill missing values    | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/tree/main/003-feature-engineering/feature-transformation/handling-missing-data/D039-knn-imputer) | [🔥](https://youtu.be/-fK-xEev2I8?si=uII3A_rnQyOuHyXp) |
| Iterative Imputer                         | MICE-style multivariate imputation               | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/tree/main/003-feature-engineering/feature-transformation/handling-missing-data/D040-iterative-imputer) | [🔥](https://youtu.be/a38ehxv3kyk?si=mlhlu5njZqdzaNA7) |



### 3.1.5🚨 Handling Outliers

| Topic                         | What You'll Learn                              | Notebook | Lecture |
|-------------------------------|------------------------------------------------|------|-----|
| What is Outliers               | Introduction to outliers and their impact      | [👨‍💻](https://youtu.be/Lln1PKgGr_M?si=Fp98i508TjfOTWPl) | [🔥](https://youtu.be/Lln1PKgGr_M?si=Fp98i508TjfOTWPl) |
| Outlier Removal using Z-Score  | Removing outliers using Z-Score                | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/tree/main/003-feature-engineering/feature-transformation/handling-outliers/D042-outlier-removal-using-zscore) | [🔥](https://youtu.be/OnPE-Z8jtqM?si=Vl-xkzHMwRKKgBMg) |
| Outlier Removal using IQR      | Removing outliers using Interquartile Range (IQR) | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/tree/main/003-feature-engineering/feature-transformation/handling-outliers/D043-outlier-removal-using-iqr-method) | [🔥](https://youtu.be/Ccv1-W5ilak?si=DmqPmRfU__AZD3F7) |
| Outlier Removal using Percentiles | Removing outliers using Percentiles           | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/tree/main/003-feature-engineering/feature-transformation/handling-outliers/D044-outlier-detection-using-percentiles) | [🔥](https://youtu.be/bcXA4CqRXvM?si=Xyc3CNYAG-bfeWuh) |



## 3.2 🏗️ Feature Construction

| Topic                          | What You'll Learn                   | Notebook | Lecture |
|--------------------------------|-------------------------------------|------|-----|
| Feature Construction and Splitting | Extract useful data and split features | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/003-feature-engineering/feature-construction/D045-feature-construction-and-feature-splitting/d045-feature-construction-and-spliting.ipynb) | [🔥](https://youtu.be/ma-h30PoFms?si=W6SevsAJmczsq8gl) |



## 3.3 🔍 Feature Extraction

| Topic                                | What You'll Learn                           | Notebook | Lecture |
|--------------------------------------|---------------------------------------------|------|-----|
| Curse of Dimensionality             | Introduction to the "curse" of high dimensions | [👨‍💻](https://youtu.be/ToGuhynu-No?si=zherGBVvVowd28gA) | [🔥](https://youtu.be/ToGuhynu-No?si=zherGBVvVowd28gA) |
| PCA Geometric Intuition (PCA)       | Geometric understanding of PCA (Principal Component Analysis) | [👨‍💻](https://youtu.be/iRbsBi5W0-c?si=JKu4Zxsw5JIobucP) | [🔥](https://youtu.be/iRbsBi5W0-c?si=JKu4Zxsw5JIobucP) |
| PCA Problem Formulation & Solution  | Formulating and solving PCA problems        | [👨‍💻](https://youtu.be/tXXnxjj2wM4?si=ZhaMAhyJ7fmRGPVE) | [🔥](https://youtu.be/tXXnxjj2wM4?si=ZhaMAhyJ7fmRGPVE) |
| PCA Step by Step Implementation     | Implementing PCA step by step               | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/003-feature-engineering/feature-extraction/D047-pca/d048-pca-step-by-step.ipynb) | [🔥](https://youtu.be/tofVCUDrg4M?si=BNCdU1hioqkNG97f) |
| PCA + KNN (MNIST Dataset)           | Apply PCA and KNN on the MNIST dataset      | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/003-feature-engineering/feature-extraction/D047-pca/principal-component-analysis.ipynb) | [🔥](https://www.kaggle.com/code/rudraprasadbhuyan/principal-component-analysis-knn) |


---

# 4. 📊  Regression

| Topic                        | What You'll Learn                          | Notebook | Lecture |
|------------------------------|--------------------------------------------|------|-----|
| Simple LR from Scratch        | Code implementation from scratch           | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/004-Regression/D048-simple-linear-regression/d048-simple-lr-from-scratch.ipynb) | [🔥](https://youtu.be/dXHIDLPKdmA?si=iIJotZ1If_TIdVuP) |
| Sklearn LR                    | Using `LinearRegression()` from `sklearn`  | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/004-Regression/D048-simple-linear-regression/d048-sklearn-simple-linear-regression.ipynb) | [🔥](https://youtu.be/UZPfbG0jNec?si=bjXTEvU1qQqCxN6T) |
| Regression Metrics            | Understanding R² score, MSE, RMSE          | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/tree/main/004-Regression/D049-regression-metrics) | [🔥](https://youtu.be/Ti7c-Hz7GSM?si=ZvJ43nHhbPzOS0RV) |
| Geometric Intuition           | Understanding the geometric intuition of MLR | [👨‍💻](https://youtu.be/ashGekqstl8?si=hswrGO8OG0I41eo2) | [🔥](https://youtu.be/ashGekqstl8?si=hswrGO8OG0I41eo2) |
| Multiple LR from Scratch      | Code implementation from scratch           | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/004-Regression/D050-multiple-linear-regression/d050-code-from-scratch.ipynb) | [🔥](https://youtu.be/VmZWXzxmNrE?si=bL5HbAJaA-t9ijKa) |
| Mathematical Formulation Sklearn LR | Using `LinearRegression()` from `sklearn` | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/004-Regression/D050-multiple-linear-regression/d050-multiple-linear-regression.ipynb) | [🔥](https://youtu.be/NU37mF5q8VE?si=gARD4yKizqujXnF1) |
| Polynomial LR                 | Preprocessing and using `PolynomialFeatures()` | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/004-Regression/D053-polynomial-regression/d053-polynominal-regression.ipynb) | [🔥](https://youtu.be/BNWLf3cKdbQ?si=_E0LHxNmNEn-V_Re) |


---

# 5. 🧑‍💻 Gradient Descent

| Topic                        | What You'll Learn                        | Notebook | Lecture |
|------------------------------|------------------------------------------|------|-----|
| Gradient Descent              | Basic Introduction to Gradient Descent   | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/005-Gradient-Descent/d051-1-gradient-descent-basic.ipynb) | [🔥](https://youtu.be/ORyfPJypKuU?si=ufY-8HvDbjnAerf0) |
| Batch Simple GD               | Implementing Simple Batch GD from Scratch | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/005-Gradient-Descent/d051-2-simple-batch-gd-from-scratch.ipynb) | [🔥](https://youtu.be/ORyfPJypKuU?si=ufY-8HvDbjnAerf1) |
| Batch GD                      | Implementing Batch Gradient Descent from Scratch | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/005-Gradient-Descent/d052-3-batch-gradient-descent-from-scratch.ipynb) | [🔥](https://youtu.be/Jyo53pAyVAM?si=FZycRMeQfpvLIUaH) |
| Stochastic GD                 | Implementing Stochastic Gradient Descent from Scratch | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/005-Gradient-Descent/d052-4-stochastic-gradient-descent-from-scratch.ipynb) | [🔥](https://youtu.be/V7KBAa_gh4c?si=SFNLEfFo4DtkfU4T) |
| Mini Batch GD                 | Implementing Mini-Batch Gradient Descent from Scratch | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/005-Gradient-Descent/d052-5-mini-batch-gradient-descent-from-scratch.ipynb) | [🔥](https://youtu.be/_scscQ4HVTY?si=6eg8lAFkCKG84e9W) |


---

# 6. 👮 Regularization

| Topic | What You'll Learn | Notebook  | Lecture |
|:-----|:------------------|:--------------:|:------------:|
| Bias-Variance Trade-off | Understanding Underfitting & Overfitting | - | [ 🔥](https://youtu.be/74DU02Fyrhk?si=yynvkCEiZeuwzFgg) |
| Ridge Regression Geometric Intuition (Part 1) | Introduction to Regularized Linear Models | [ 👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/006-Regularization/D055-regularized-linear-models/d055-1-ridge-regularization.ipynb) | [ 🔥](https://youtu.be/aEow1QoTLo0?si=2deJS6Xgr2OQ0V0I) |
| Ridge Regression Mathematical Formulation (Part 2) | Scratch  for slope (m) and intercept (b) | [ 👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/006-Regularization/D055-regularized-linear-models/d055-2-ridge-regression-from-scratch-m-and-b.ipynb) | [ 🔥](https://youtu.be/oDlZBQjk_3A?si=aqYTS7Bijh4m6CGH) |
| Ridge Regression Mathematical Formulation (Part 2) | Full Scratch  Implementation | [ 👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/006-Regularization/D055-regularized-linear-models/d055-3-ridge-regression-from-scratch.ipynb) | [ 🔥](https://youtu.be/oDlZBQjk_3A?si=aqYTS7Bijh4m6CGH) |
| Ridge Regression (Part 3) | Gradient Descent Implementation | [ 👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/006-Regularization/D055-regularized-linear-models/d055-4-ridge-regression-gradient-descent.ipynb) | [ 🔥](https://youtu.be/Fci_wwMp8G8?si=NjBiXvQT_VfVkNdc) |
| 5 Key Points about Ridge Regression | Q&A, Effects, and Insights | [ 👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/006-Regularization/D055-regularized-linear-models/d055-5-ridge-regression-key-understandings.ipynb) | [ 🔥](https://youtu.be/8osKeShYVRQ?si=7BuDXc0k5GtwLUs5) |
| Lasso Regression | Full  Implementation | [ 👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/006-Regularization/D056-lasso-regression/d056-1-lasso-regression.ipynb) | [ 🔥](https://youtu.be/HLF4bFbBgwk?si=U3Opb8ukrlvNzfQU) |
| Why Lasso Regression Creates Sparsity | Understanding Sparsity Effect | [ 👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/006-Regularization/D056-lasso-regression/d056-2-lasso-regression-key-points.ipynb) | [ 🔥](https://youtu.be/FN4aZPIAfI4?si=Ew8Jag9JI-0cjvxT) |
| ElasticNet Regression | Comparison and Effects | [ 👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/006-Regularization/D057-elasticnet-regression/d057-elastic-net-regression.ipynb) | [ 🔥](https://youtu.be/2g2DBkFhTTY?si=EKe9wHb4dNt8GFCi) |

---

# 7. 📘 Logistic Regression

| Topic | What You'll Learn | Notebook | Lecture |
|-------|-------------------|----------------|--------------|
| LR 1 - Perceptron Trick | Why to use it, transformations, region concept | - | [🔥 ](https://youtu.be/XNXzVfItWGY?si=V4IK0OnMLi-loBmq) |
| LR 2 - Perceptron Trick Code | Math to algorithm conversion | [👨‍💻  ](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/007-Logistic-Regression/D058-60-logistic-regression/d058-1-perceptron-trick.ipynb) | [🔥 ](https://youtu.be/tLezwPKvPK4?si=BmBqY5r5Yw1FMUD7) |
| LR 3 - Sigmoid Function | How the sigmoid function helps to find the error line | [👨‍💻  ](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/007-Logistic-Regression/D058-60-logistic-regression/d058-2-perceptron-trick-sigmoid.ipynb) | [🔥 ](https://youtu.be/ehO0-6i9qD4?si=G5MPcuqNU7q1aHRM) |
| LR 4 - Math Behind Optimal Line | Maximum likelihood, binary cross-entropy, gradient descent | - | [🔥 ](https://youtu.be/6bXOo0sxY5c?si=dIJouZmqGSly5LY9) |
| Extra - Derivative of Sigmoid | Helps derive matrix form from loss function | - | [🔥 ](https://youtu.be/awjXaFR1jOM?si=DFoNIZXKWkxoNMMM) |
| LR 5 - Logistic Regression (Gradient Descent) | Scratch   implementation | [👨‍💻  ](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/007-Logistic-Regression/D058-60-logistic-regression/d058-3-gradient-descent.ipynb) | [🔥 ](https://youtu.be/ABrrSwMYWSg?si=qLVeSllhEMaO3Z64) |
| LR 6 - Multinomial Logistic Regression | Softmax regression | [👨‍💻  ](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/007-Logistic-Regression/D058-60-logistic-regression/d060-4-softmax-regression-multinomial.ipynb) | [🔥 ](https://youtu.be/Z8noL_0M4tw?si=YT6bOk373RWTdBa9) |
| LR 7 - Non-Linear Regression | Polynomial features | [👨‍💻  ](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/007-Logistic-Regression/D058-60-logistic-regression/d060-5-polynomial-logistic-regression.ipynb) | [🔥 ](https://youtu.be/WnBYW_DX3sM?si=3OwkTs282U-JXyk0) |
| LR 8 - Hyperparameter | Sklearn documentation and hyperparameter tuning | - | [🔥 ](https://youtu.be/ay_OcblJasE?si=jxUuC_Mp8J6CJWUy) |
| P1 Classification Metrics | Accuracy, confusion matrix, Type I & II errors, binary vs. multi-class | [👨‍💻  ](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/007-Logistic-Regression/D059-classification-metrics/d059-1-classification-metrics-binary.ipynb) | [🔥 ](https://youtu.be/c09drtuCS3c?si=DLNZwtnk52s1kKFl) |
| P2 Classification Metrics Binary | Precision, recall & F1 score (binary) | [👨‍💻  ](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/007-Logistic-Regression/D059-classification-metrics/d059-2-classification-metrics-multi-iris.ipynb) | [🔥 ](https://youtu.be/iK-kdhJ-7yI?si=8q_tIFuS9gCAoOgc) |
| P2 Classification Metrics Multi-Class | Precision, recall & F1 score (multi-class) | [👨‍💻  ](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/007-Logistic-Regression/D059-classification-metrics/d059-3-classification-metrics-multi-mnist.ipynb) | [🔥 ](https://youtu.be/iK-kdhJ-7yI?si=8q_tIFuS9gCAoOgc) |

---

# 8. 🌴 Decision Tree

| Topic | What You'll Learn | Notebook | Lecture |
|-------|-------------------|----------------|--------------|
| D1 - Decision Tree Geometric Intuition | Entropy, Gini Impurity, Information Gain | -  | [🔥](https://youtu.be/IZnno-dKgVQ?si=5tmUP56R95itdTmN)                                         |
| D2 - Hyperparameters            | Overfitting and Underfitting            | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/008-Decision-Tree/d061-1-decision-trees-classification.ipynb) | [🔥](https://youtu.be/mDEV0Iucwz0?si=1AXH7wuJy9TciuUv)                                         |
| D3 - Regression Trees         | Numerical Points                        | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/008-Decision-Tree/d061-2-decision-tree-regression.ipynb) | [🔥](https://youtu.be/RANHxyAvtM4?si=tWatJitWY0fmdT0N)                                         |
| D4 - Awesome Decision Tree     | `dtreeviz` Library                      | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/008-Decision-Tree/d061-3-dtreeviz_demo.ipynb) | [🔥](https://youtu.be/RANHxyAvtM4?si=speQ_QjsBCKEy9yk)                                         |

---

# 9.🌋 Voting Ensemble Learning

| Topic                          | What You'll Learn                  | Notebook                                                                                    | Lecture                                                                                       |
|--------------------------------|------------------------------------|------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| Intro to Ensemble Learning     | Ensemble techniques in ML          | -                                                                                                    | [🔥](https://youtu.be/bHK1fE_BUms?si=uGyvIKSgT6ZwKvX3)                                        |
| VE1 - Voting Ensemble          | Code overview                      | -                                                                                                    | [🔥](https://youtu.be/_W1i-c_6rOk?si=wS6uzuGfgjSGdpB3)                                        |
| VE2 - Voting Classifier        | Hard vs Soft voting                | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/009-Voting-Ensmble/d063-01-voting-classifier-iris.ipynb) | [🔥](https://youtu.be/pGQnNYdPTvY?si=im2srmpXhHDIBPO_)            |
| VE3 - Voting Ensemble Regression | Ensemble for regression tasks      | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/009-Voting-Ensmble/d063-02-voting-regressor.ipynb)     | [🔥](https://youtu.be/ut4vh59rGkw?si=WjYF4t9Tf224R7Hd)    |


---


# 10. 🛍️ Bagging Ensemble Learning

| Topic                 | What You'll Learn         | Notebook                                                                                         | Lecture                                                                                    |
|-----------------------|---------------------------|----------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| BE1 - Introduction     | Basics of bagging         | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/010-Bagging-Ensemble/d064-01-bagging-learning.ipynb)      | [🔥](https://youtu.be/LUiBOAy7x6Y?si=27LJ6X3X_LdjVsAy)                                       |
| BE2 - Bagging Classifiers | Bagging for classification | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/010-Bagging-Ensemble/d064-02-bagging-classifiers.ipynb)   | [🔥](https://youtu.be/-1T54G_E-ys?si=LPNYI-kthEb558zk)                                       |
| BE3 - Bagging Regressor   | Bagging for regression    | [👨‍💻](https://github.com/Rudra-G-23/100-Days-of-ML/blob/main/010-Bagging-Ensemble/d064-03-bagging-regression.ipynb)     | [🔥](https://youtu.be/HYVzrETXbkE?si=Y5qlzBc4AwyrehtA)                                       |
