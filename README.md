## 0. Project Title

**Predictive Modeling for Diabetes Risk Assessment**

**Author**

Meena Murali

## 1. Executive SUmmary

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

### 5.0 Load the Data
Import the dataset into our environment, using pandas library. This allows us to manipulate and explore the data, preparing it for analysis and model training.

### 5.1 Understand the Data - Data Preprocessing & Exploration
#### 5.1.0 Statistical Summary
To Identify data quality issues, Detect the need for scaling or normalization, Highlight features with high variance that may dominate models if not scaled, 
Justify feature engineering, such as categorizing age or deriving ratios

#### 5.1.0.1 **Insights from Summary**
+ Outliers are present in Insulin, SkinThickness, and Pregnancies.
+ Missing values if any will be encoded as 0s in features like Glucose and BMI and will be handled through median imputation.
+ Skewness in variables like Insulin and SkinThickness may influence the model types and will be normalized or log-transformed.
+ Standard deviations are high for Insulin and Glucose, indicating wide variation.
   
#### 5.1.1 Data Loading & Cleaning
Missing Values & Deduplication - We check for missing values or zero values for certain features. These values will be handled through imputation techniques or removal, depending on their distribution. Duplicate entries will be removed if found.

#### 5.1.2 Exploratory Data Analysis (EDA)
**Visualization:** - Perform Visual and statistical exploration that help identify trends, relationships, and potential outliers.
+ **Count plots** - To visualize the distribution of diabetes outcomes
+ **Histograms** - For feature distributions
+ **Correlation heatmaps** - To understand the relationships between variables
+ **Box plots** - To visualize the distribution of features like Glucose, BMI, etc., across diabetic and non-diabetic groups
+ **Violin plots** - To compare distributions of features like Glucose and Insulin between the two outcome classes
+ **KDE plots** - To visualize feature distributions, such as age, by outcome class
  
##### 5.1.2.1 Count Plot (Outcome Distribution) - 
To see how many individuals (%) in the dataset have diabetes (Outcome = 1) versus those who don't (Outcome = 0).

![Image](https://github.com/user-attachments/assets/8b6840a9-b788-4545-94b4-d9f3221113b1)

##### 5.1.2.1.1 **Insights from Count Plot** - 
We see that the dataset is slightly imbalanced, with more non-diabetic cases. 


##### 5.1.2.2 Histograms (Distribution of Features) - 
To visualize how features like Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,	DiabetesPedigreeFunction are distributed across the population

![Image](https://github.com/user-attachments/assets/0f4c2dfa-5d83-4bff-9e12-a08b2b74568c)

##### 5.1.2.2.1 **Insights from Histogram** - 
Some features (e.g., Insulin) are highly skewed, indicating outliers or irregular distribution. Others like Glucose show distinct peaks that may separate diabetic from non-diabetic individuals.


##### 5.1.2.3 Correlation Heatmap - 
To check how features are related to each other and to the Outcome

![Image](https://github.com/user-attachments/assets/efb13d74-b4ec-4a02-aa23-bb0e71a7b837)

##### 5.1.2.3.1 **Insights from Heatmap** - 
Glucose, BMI, and Age show moderate to strong positive correlation with Outcome, making them promising predictors.


##### 5.1.2.4 Boxplots Plots - 
To compare feature value distributions across diabetic and non-diabetic groups

![Image](https://github.com/user-attachments/assets/33cd501a-594b-452e-b5c5-5a4a4d725221)

##### 5.1.2.4.1 **Insights from Box PLot** - 
Diabetic individuals have higher median Glucose and BMI. 


##### 5.1.2.5 Violin Plots - 
To view both distribution shape and spread of each feature within each class.

![Violin Plot](https://github.com/user-attachments/assets/0a31d2ae-5852-4158-a644-904105bc918b)

##### 5.1.2.5.1 **Insights from Violin Plot** - 
Glucose and Insulin show clear separation in distribution shapes between diabetic and non-diabetic patients. This reinforces their predictive power.


##### 5.1.2.6 KDE Plot - 
To Directly compare feature distributions

![Image](https://github.com/user-attachments/assets/4843c154-561e-44ca-b220-337d15031b5a)

##### 5.1.2.6.1 ****Insights from KDE Plot**** - 
Glucose shows a clear shift in distribution — highly relevant to diabetes prediction, BMI and Age show moderate predictive value, Insulin might be less impactful on its own.


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

## 8. Outline of project

Jupyter Notebook - https://github.com/meenamurali2m/Capstone_Project/blob/main/Capstone_MM.ipynb

