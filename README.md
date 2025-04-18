## 1. Introduction
### Business Problem
Diabetes, especially Type 2, is a critical global health issue, leading to severe complications such as heart disease, kidney failure, and increased mortality rates. Early diagnosis and intervention are essential for improving patients' quality of life and reducing healthcare costs. Type 2 diabetes is closely linked to lifestyle factors like poor diet, obesity, and lack of physical activity.

The **Pima Indians Diabetes** dataset provides clinical data about individuals with a high prevalence of Type 2 diabetes, making it an ideal resource for predictive modeling. This study leverages machine learning techniques to predict the likelihood of diabetes occurrence based on health-related factors such as age, BMI, blood pressure, and glucose levels.

By comparing the performance of various machine learning models, the goal is to identify the most effective approach for early detection of diabetes, which can help healthcare providers and insurers reduce long-term costs and improve patient outcomes.

## 2. Research Question
Can machine learning models predict the likelihood of Type 2 diabetes in Pima Indian women based on clinical features such as age, BMI, blood pressure, and glucose levels?

## 3. Data Source
The dataset used in this study is the Pima Indians Diabetes dataset, which is publicly available at the UCI Machine Learning Repository:

[Pima Indians Diabetes Dataset]([url](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database))

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

## 4. Methodology
We load and explore the dataset with basic statistical summaries (using describe(), head(), and info()).

### 4.0 Load the Data
Import the dataset into our environment, using pandas library. This allows us to manipulate and explore the data, preparing it for analysis and model training.

### 4.1 Understand the Data - Data Preprocessing & Exploration
#### 4.1.0 Statistical Summary
To Identify data quality issues, Detect the need for scaling or normalization, Highlight features with high variance that may dominate models if not scaled, 
Justify feature engineering, such as categorizing age or deriving ratios

#### 4.1.0.1 **Insights from Summary**
+ Outliers are present in Insulin, SkinThickness, and Pregnancies.
+ Missing values if any will be encoded as 0s in features like Glucose and BMI and will be handled through median imputation.
+ Skewness in variables like Insulin and SkinThickness may influence the model types and will be normalized or log-transformed.
+ Standard deviations are high for Insulin and Glucose, indicating wide variation.
   
#### 4.1.1 Data Loading & Cleaning
Missing Values & Deduplication - We check for missing values or zero values for certain features. These values will be handled through imputation techniques or removal, depending on their distribution. Duplicate entries will be removed if found.

#### 4.1.2 Exploratory Data Analysis (EDA)
**Visualization:** - Perform Visual and statistical exploration that help identify trends, relationships, and potential outliers.
+ **Count plots** - To visualize the distribution of diabetes outcomes
+ **Histograms** - For feature distributions
+ **Correlation heatmaps** - To understand the relationships between variables
+ **Box plots** - To visualize the distribution of features like Glucose, BMI, etc., across diabetic and non-diabetic groups
+ **Violin plots** - To compare distributions of features like Glucose and Insulin between the two outcome classes
+ **KDE plots** - To visualize feature distributions, such as age, by outcome class
  
##### 4.1.2.1 Count Plot (Outcome Distribution) - 
To see how many individuals in the dataset have diabetes (Outcome = 1) versus those who don't (Outcome = 0).

![Image](https://github.com/user-attachments/assets/8b6840a9-b788-4545-94b4-d9f3221113b1)

##### 4.1.2.1.1 **Insights from Count Plot** - 
We see that the dataset is slightly imbalanced, with more non-diabetic cases. 

##### 4.1.2.2 Histograms (Distribution of Features) - 
To visualize how features like Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,	DiabetesPedigreeFunction are distributed across the population

![Image](https://github.com/user-attachments/assets/0f4c2dfa-5d83-4bff-9e12-a08b2b74568c)

##### 4.1.2.2.1 **Insights from Histogram** - 
Some features (e.g., Insulin) are highly skewed, indicating outliers or irregular distribution. Others like Glucose show distinct peaks that may separate diabetic from non-diabetic individuals

##### 4.1.2.3 Correlation Heatmap - 
To check how features are related to each other and to the Outcome

![Image](https://github.com/user-attachments/assets/efb13d74-b4ec-4a02-aa23-bb0e71a7b837)

##### 4.1.2.3.1 **Insights from Heatmap** - 
Glucose, BMI, and Age show moderate to strong positive correlation with Outcome, making them promising predictors.

##### 4.1.2.4 Boxplots Plots - 
To compare feature value distributions across diabetic and non-diabetic groups

![Image](https://github.com/user-attachments/assets/33cd501a-594b-452e-b5c5-5a4a4d725221)

##### 4.1.2.4.1 **Insights from Box PLot** - 
Diabetic individuals have higher median Glucose and BMI. 

##### 4.1.2.5 Violin Plots - 
To view both distribution shape and spread of each feature within each class.

![Violin Plot](https://github.com/user-attachments/assets/0a31d2ae-5852-4158-a644-904105bc918b)

##### 4.1.2.5.1 **Insights from Violin Plot** - 
Glucose and Insulin show clear separation in distribution shapes between diabetic and non-diabetic patients. This reinforces their predictive power.


##### 4.1.2.6 KDE Plot - 
To Directly compare feature distributions

![Image](https://github.com/user-attachments/assets/4843c154-561e-44ca-b220-337d15031b5a)

##### 4.1.2.6.1 ****Insights from KDE Plot**** - 
Glucose shows a clear shift in distribution — highly relevant to diabetes prediction, BMI and Age show moderate predictive value, Insulin might be less impactful on its own.


### 4.2 Feature Engineering
Based on insights from the Exploratory Data Analysis, new features are created to better capture underlying patterns in the data and improve model performance. We will also perform some light feature engineering to include - 
+ Creating age categories (e.g., Young Adult, Adult, Senior)
+ Adding a derived feature: Glucose-to-Insulin ratio (to reflect insulin sensitivity)
+ Dropping features with very low variance or weak correlation (after we did EDA)

These engineered features aim to provide additional context that raw variables may not fully express. One-hot encoding is used for categorical variables to make them compatible with machine learning algorithms.

### 4.3 Feature Scaling
Features like Age, BMI, Glucose, etc., are on different scales and will be normalized to ensure each feature contributes equally to model performance. To ensure that no single feature disproportionately influences the model’s performance, continuous variables such as BMI, Age, and Glucose levels are standardized using StandardScaler. 
Algorithms like Logistic Regression, SVM, and KNN are sensitive to feature magnitudes. Scaling centers the data (mean = 0, std = 1), improving convergence and performance.

### 4.4 Machine Learning Models
+ Supervised Learning Models: Logistic Regression, Decision Trees, Random Forest, Support Vector Machines (SVM), and k-Nearest Neighbors (k-NN).
+ Model Tuning: Hyperparameter optimization using GridSearchCV or RandomizedSearchCV will be used to fine-tune the models.

#### 4.4.1 Baseline Model - We will use a simple, algorithm like Logistic Regression for the first baseline.

![Confusion Matrix - Logistic Regression](https://github.com/user-attachments/assets/12f3510c-580b-48bd-94e8-d97f5e0d0cc9)

#### 4.4.1.1 **Confusion Matrix Analysis**

+ True Negatives (TN) = 83 → Non-diabetic predicted as non-diabetic ✔️
+ False Positives (FP) = 17 → Non-diabetic predicted as diabetic ❌
+ False Negatives (FN) = 24 → Diabetic predicted as non-diabetic ❌
+ True Positives (TP) = 30 → Diabetic predicted as diabetic ✔️

#### 4.4.1.2 **Baseline Logistic Regression Model Summary**

| Metric  | Class 0 (Non-Diabetic) | Class 1 (Diabetic)  | Notes |
| ------------- | ------------- | ------------- | ------------- |
| Precision | 0.78 | 0.64 | When the model predicts diabetes, it's correct 64% of the time |
| Recall | 0.83 | 0.56 | It correctly identifies 56% of actual diabetic cases (misses 44%) |
| F1-score | 0.80 | 0.59 | Lower F1 for class 1 due to lower recall |
| Accuracy | 73.4% overall |  | Balanced but favors non-diabetic predictions |

  
#### 4.4.1.3 **Insights from Logistic Regression Model**
+ The model does well predicting non-diabetic individuals.
+ But it misses many actual diabetic cases (high false negatives → recall = 0.56).
+ In healthcare, missing a diabetic case can be risky — recall for class 1 is critical here.

#### 4.4.1.4 **Next Steps**
1. Try Other Models - Try models better suited for non-linear relationships or feature interactions:
+ Random Forest
+ XGBoost
+ SVM
+ k-NN
2. Handle Class Imbalance - The dataset is slightly imbalanced (approx. 65% non-diabetic, 35% diabetic). Might use class_weight='balanced' in LogisticRegression
3. Visualizations to Explore Further
   
****************************************************************************************************************************************************************************
**NOTE: Once we try the other models the below sections will be finalized**
****************************************************************************************************************************************************************************
### 4.5 Model Evaluation
Performance will be evaluated using metrics such as Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
The data will be split into training and testing sets, and possibly cross-validation will be used to avoid overfitting and ensure reliable results.

## 5. Expected Results
The model should provide a reliable prediction of diabetes likelihood based on health factors.
Key features (e.g., BMI, glucose levels, age) will be identified as significant predictors of diabetes risk.
The results will include performance metrics that validate the predictive ability of the selected models.
The best model will enable early detection of diabetes, leading to better prevention strategies.

## 6. Importance of the Study
### Public Health Significance:
Early detection of Type 2 diabetes can help reduce long-term healthcare costs and improve patient outcomes through earlier interventions, such as lifestyle changes and medication. This study aims to develop a predictive tool that helps healthcare providers identify individuals at high risk of diabetes before it progresses to more severe stages.

### Business Problem
Healthcare providers and insurance companies face rising costs due to the increasing prevalence of diabetes and its complications. By accurately predicting diabetes risk, this study offers a potential solution for reducing costs associated with treating advanced diabetes-related complications. Healthcare providers can offer targeted prevention and treatment plans, while insurance companies can focus on early intervention, improving the overall health of the population and reducing claims related to chronic diabetes conditions.

### Impact of Unanswered Question
If this question remains unanswered, individuals at high risk of diabetes may not receive timely intervention or monitoring, leading to worsening health outcomes and increased medical expenses. The lack of predictive tools makes it difficult for healthcare providers and insurers to manage the growing burden of diabetes on the healthcare system.

### Benefit of the Analysis
By providing a model that predicts diabetes risk, healthcare providers can offer more personalized care, focusing on prevention for high-risk individuals, thus reducing the overall cost burden.
Insurance companies can use the predictive model to identify high-risk individuals and offer preventive care plans, potentially reducing the number of claims for expensive diabetic complications.
This analysis will also serve as a foundation for creating more targeted health interventions, allowing healthcare systems to allocate resources more efficiently and improve patient outcomes.

## 7. Conclusion
This study will apply machine learning techniques to predict the likelihood of diabetes in Pima Indian women based on clinical features. The ultimate goal is to identify the most effective models for early detection. The findings could serve as a valuable resource for healthcare providers and insurers, enabling them to offer preventive care and reduce the long-term costs associated with diabetes complications. Through this analysis, we hope to contribute to better public health management and improved diabetes care.
