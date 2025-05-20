## 0. Project Title - Predictive Modeling for Diabetes Risk Assessment

## 1. Executive Summary

### 1.1 Business Problem
Diabetes, especially Type 2, is a critical global health issue, leading to severe complications such as heart disease, kidney failure, and increased mortality rates. Early diagnosis and intervention are essential for improving patients' quality of life and reducing healthcare costs. Type 2 diabetes is closely linked to lifestyle factors like poor diet, obesity, and lack of physical activity.

The **Pima Indians Diabetes** dataset provides clinical data about individuals with a high prevalence of Type 2 diabetes, making it an ideal resource for predictive modeling. This study leverages machine learning techniques to predict the likelihood of diabetes occurrence based on health-related factors such as age, BMI, blood pressure, and glucose levels.

By comparing the performance of various machine learning models, the goal is to identify the most effective approach for early detection of diabetes, which can help healthcare providers and insurers reduce long-term costs and improve patient outcomes.

## 2 Rationale - Importance of the Study
### 2.1 Public Health Significance
Early detection of Type 2 diabetes can help reduce long-term healthcare costs and improve patient outcomes through earlier interventions, such as lifestyle changes and medication. This study aims to develop a predictive tool that helps healthcare providers identify individuals at high risk of diabetes before it progresses to more severe stages.

### 2.2 Challenges
Healthcare providers and insurance companies face rising costs due to the increasing prevalence of diabetes and its complications. By accurately predicting diabetes risk, this study offers a potential solution for reducing costs associated with treating advanced diabetes-related complications. Healthcare providers can offer targeted prevention and treatment plans, while insurance companies can focus on early intervention, improving the overall health of the population and reducing claims related to chronic diabetes conditions.

### 2.3 Impact of Unanswered Question
If this question remains unanswered, individuals at high risk of diabetes may not receive timely intervention or monitoring, leading to worsening health outcomes and increased medical expenses. The lack of predictive tools makes it difficult for healthcare providers and insurers to manage the growing burden of diabetes on the healthcare system.

### 2.4 Benefit of the Analysis
By providing a model that predicts diabetes risk, healthcare providers can offer more personalized care, focusing on prevention for high-risk individuals, thus reducing the overall cost burden.
Insurance companies can use the predictive model to identify high-risk individuals and offer preventive care plans, potentially reducing the number of claims for expensive diabetic complications.
This analysis will also serve as a foundation for creating more targeted health interventions, allowing healthcare systems to allocate resources more efficiently and improve patient outcomes.

## 3. Research Question
Can machine learning models predict the likelihood of Type 2 diabetes in Pima Indian women based on clinical features such as age, BMI, blood pressure, and glucose levels?

## 4. Data Source
The dataset used in this study is the [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database), which is publicly available at the UCI Machine Learning Repository.

This dataset contains the following attributes:

1. **Pregnancies**: Number of times the patient has been pregnant.
2. **Glucose**: Plasma glucose concentration after a 2-hour oral glucose tolerance test.
3. **BloodPressure**: Diastolic blood pressure (mm Hg).
4. **SkinThickness**: Triceps skin fold thickness (mm).
5. **Insulin**: 2-Hour serum insulin (mu U/ml).
6. **BMI**: Body mass index (kg/m²).
7. **DiabetesPedigreeFunction**: A function that scores the likelihood of diabetes based on family history.
8. **Age**: Age of the individual (years).
9. **Outcome**: 1 indicates diabetes, 0 indicates no diabetes.

## 5. Methodology

### 5.0 Load & Understand the Data
Import the dataset into our environment, using pandas library. This allows us to manipulate and explore the data, preparing it for analysis and model training. 

### 5.1 Exploratory Data Analysis (EDA)
Perform Data Preprocessing & Visual Exploration to identify data quality issues such as missing values, duplicates, skewness, scaling issues and fix them. Perform an initial visual distribution using Count Plots

The dataset is slightly imbalanced, with more non-diabetic cases. Did some Fetaure Engineering and Feature scaling and then Trained a Logistic Regression model as the baseline on this dataset and found the accuracy score to be 73.4%

### 5.2 Balanace the datset and run the EDA
We balanced the dataset and performed Visual and statistical exploration that help identify trends, relationships, and potential outliers. Some key plots are shown below.
+ **Count plots** - To visualize the distribution of diabetes outcomes
![Image](https://github.com/user-attachments/assets/f52612df-1cf0-4d04-9173-4a781b8b203b)
  
+ **Correlation heatmaps** - To understand the relationships between variables
 ![Image](https://github.com/user-attachments/assets/63eb84f7-9067-4044-b9e7-23d2d01479b0)

+ **KDE plots** - To visualize feature distributions, such as age, by outcome class
![Image](https://github.com/user-attachments/assets/96565f33-48ad-43a4-ad1b-ca00d922db1e)
  

### 5.2 Feature Engineering
Based on insights from the Exploratory Data Analysis, new features are created to better capture underlying patterns in the data and improve model performance. We will also perform some light feature engineering to include - 
+ Creating age categories (e.g., Young Adult, Adult, Senior)
+ Adding a derived feature: Glucose-to-Insulin ratio (to reflect insulin sensitivity)
+ Dropping features with very low variance or weak correlation (after we did EDA)

These engineered features aim to provide additional context that raw variables may not fully express. One-hot encoding is used for categorical variables to make them compatible with machine learning algorithms.

### 5.3 Feature Scaling
Features like Age, BMI, Glucose, etc., are on different scales and will be normalized to ensure each feature contributes equally to model performance. To ensure that no single feature disproportionately influences the model’s performance, continuous variables such as BMI, Age, and Glucose levels are standardized using StandardScaler. 
Algorithms like Logistic Regression, SVM, and KNN are sensitive to feature magnitudes. Scaling centers the data (mean = 0, std = 1), improving convergence and performance.

### 5.4 Machine Learning Models

#### 5.4.1 Baseline Model
We will use a simple, algorithm like Logistic Regression for the first baseline.

##### 5.4.1.1 Confusion Matrix

![Confusion Matrix - Logistic Regression](https://github.com/user-attachments/assets/12f3510c-580b-48bd-94e8-d97f5e0d0cc9)

##### 5.4.1.2 Classification Report

| Metric  | Class 0 (Non-Diabetic) | Class 1 (Diabetic)  | Notes |
| ------------- | ------------- | ------------- | ------------- |
| Precision | 0.78 | 0.64 | The model predicts diabetes correctly 64% of the time |
| Recall | 0.83 | 0.56 | It correctly identifies 56% of actual diabetic cases (misses 44%) |
| F1-score | 0.80 | 0.59 | Lower F1 for class 1 due to lower recall |
| Accuracy | 73.4% overall |  | Seems Balanced but favors non-diabetic predictions |

 ##### 5.4.1.3 Insights from Logistic Regression Model
+ The model does well predicting non-diabetic individuals.
+ But it misses many actual diabetic cases (high false negatives → recall = 0.56).
+ In healthcare, missing a diabetic case can be risky — recall for class 1 is critical here.
  
## 6. Results 

### 6.1 Exploratory Data Analysis (EDA) Results
The dataset contained 768 records with 8 clinical features and one target variable (Outcome) indicating diabetes diagnosis (1 = Diabetic, 0 = Non-Diabetic). Our EDA revealed several key insights:

#### 6.1.1 Class Distribution
+ 65.1% Non-Diabetic (0)
+ 34.9% Diabetic (1)
+ This indicates a moderately imbalanced dataset.

#### 6.1.2 Strongly Correlated Features with Diabetes
+ Glucose (correlation = 0.49) showed the strongest positive correlation with diabetes.
+ BMI (0.31), Age (0.24), and Pregnancies (0.22) also showed weak-to-moderate positive relationships.
+ Blood Pressure, Skin Thickness, and Insulin showed weak or near-zero correlation.

#### 6.1.3 Distribution Patterns
+ Diabetic individuals tend to have higher glucose, BMI, and age levels.
+ KDE and box plots visualized this difference effectively.

#### 6.1.4 New Features Created
+ AgeGroup: Categorical bins (e.g., ‘20s’, ‘30s’, etc.)
+ Glucose-to-Insulin Ratio: A derived metric for insulin sensitivity
+ Both showed notable separation between diabetic and non-diabetic groups.

### 6.2 Data Cleaning & Preprocessing Results
+ Replaced 0s with NaNs in critical columns (Glucose, BMI, etc.) and imputed them using median values.
+ Removed duplicate entries.
+ Applied feature scaling using StandardScaler to normalize continuous features.
+ Used one-hot encoding for the new AgeGroup feature.

### 6.3 Baseline Machine Learning Model
Trained a Logistic Regression model as the baseline using the processed features. 

#### 6.3.1 Model Performance on Test Data

| Metric  | Class 0 (Non-Diabetic) | Class 1 (Diabetic)  | Notes |
| ------------- | ------------- | ------------- | ------------- |
| Precision | 0.78 | 0.64 | The model predicts diabetes correctly 64% of the time |
| Recall | 0.83 | 0.56 | It correctly identifies 56% of actual diabetic cases (misses 44%) |
| F1-score | 0.80 | 0.59 | Lower F1 for class 1 due to lower recall |
| Accuracy | 73.4% overall |  | Seems Balanced but favors non-diabetic predictions |

#### 6.3.2  Confusion Matrix

+ True Negatives (TN) = 83 → Non-diabetic predicted as non-diabetic ✔️
+ False Positives (FP) = 17 → Non-diabetic predicted as diabetic ❌
+ False Negatives (FN) = 24 → Diabetic predicted as non-diabetic ❌
+ True Positives (TP) = 30 → Diabetic predicted as diabetic ✔️
  
+ The model performs reasonably well but struggles more with identifying diabetic cases (Class 1) — due to class imbalance and overlapping feature distributions.

### 6.4 Summary of Key Findings

+ Glucose, BMI, and Age are the top predictors of diabetes in the dataset.
+ The Glucose-to-Insulin Ratio and AgeGroup improved class separability.
+ Logistic Regression achieved an accuracy of 73.4%, setting a reliable baseline for more complex models

## 7. Next Steps
1. Try Other Models
   + Random Forest
   + XGBoost
   + SVM
   + k-NN
2. Visualizations to explore further
3. Compare models
4. Hyper parameters tuning
5. Suggest best models

## 8. Outline of project

Jupyter Notebook - https://github.com/meenamurali2m/Capstone_Project/blob/main/Capstone_MM.ipynb

