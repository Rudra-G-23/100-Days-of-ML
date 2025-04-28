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
