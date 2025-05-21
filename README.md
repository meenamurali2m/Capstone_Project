# Predictive Modeling for Diabetes Risk Assessment

## 1. Executive Summary

### 1.1 Business Problem

Diabetes, particularly Type 2, is a pressing global health issue associated with severe complications such as cardiovascular disease, kidney failure, and vision loss. Early diagnosis is essential for preventing long-term complications and reducing healthcare costs. This project aims to build a machine learning-based predictive model using the Pima Indians Diabetes dataset to identify individuals at high risk for diabetes.

### 1.2 Objective

To evaluate and compare the performance of various machine learning models in predicting the likelihood of Type 2 diabetes, with the goal of assisting healthcare providers and policymakers in early diagnosis and intervention strategies.

---

## 2. Rationale and Significance

### 2.1 Public Health Impact

Early identification of high-risk individuals allows timely lifestyle interventions and treatment, which can dramatically reduce diabetes-related complications and costs.

### 2.2 Challenges

The lack of accurate predictive tools makes it difficult to identify at-risk individuals early. This study addresses this gap using machine learning models to assess diabetes risk based on key health indicators.

### 2.3 Potential Benefits

* Improved early diagnosis and preventive care.
* Better allocation of healthcare resources.
* Reduced healthcare expenditures.

---

## 3. Research Question

Can machine learning models accurately predict the likelihood of Type 2 diabetes in Pima Indian women based on features such as glucose level, BMI, and age?

---

## 4. Dataset Description

**Source**: UCI Machine Learning Repository

**Features:**

* Pregnancies
* Glucose
* BloodPressure
* SkinThickness
* Insulin
* BMI
* DiabetesPedigreeFunction
* Age
* Outcome (Target Variable: 1 = Diabetic, 0 = Non-Diabetic)

---

## 5. Methodology

### 5.1 Data Loading & Cleaning

* Replaced zero values in key columns with NaNs.
* Imputed missing values using median imputation.
* Scaled continuous features using StandardScaler.

### 5.2 Exploratory Data Analysis

* Distribution plots and correlation heatmaps revealed strong positive correlation of diabetes with Glucose, BMI, and Age.
* Diabetic individuals showed higher glucose levels and BMI.

![Heat_Map_Balanced](https://github.com/user-attachments/assets/1614d48d-40de-4fff-b5f8-ee4cce0e7ee6)

### 5.3 Feature Engineering

* Created new features:

  * AgeGroup (categorized age bins)
  * Glucose-to-Insulin ratio
* Dropped weakly correlated features.
* One-hot encoding applied to categorical variables.

### 5.4 Model Building

Trained and evaluated the following models:

* Logistic Regression (Baseline)
* XGBoost (Default)
* XGBoost (GridSearchCV Tuned)
* Ensemble Model (Soft Voting) 

### 5.5 Hyperparameter Tuning

Used GridSearchCV for XGBoost, SVM, and k-NN to optimize performance metrics (primarily F1-score).

---

## 6. Model Evaluation

| Model                            | Accuracy  | Precision | Recall    | F1 Score  |
| -------------------------------- | --------- | --------- | --------- | --------- |
| Logistic Regression (Baseline)   | 0.740     | 0.750     | 0.720     | 0.735     |
| XGBoost (Default)                | 0.870     | 0.819     | 0.950     | 0.880     |
| XGBoost (GridSearchCV)           | 0.870     | 0.830     | 0.930     | 0.877     |
| **Ensemble Model (Soft Voting)** | **0.895** | **0.900** | **0.895** | **0.902** |


Best Model: The Ensemble Model (Soft Voting) outperformed others across all key metrics.


### 6.1 Model Performance Comparison

![Image](https://github.com/user-attachments/assets/16ac0f9c-7998-4738-b517-3058df8691b5)

### 6.2 Model Testing


Using the Model for Diabetes Prediction, we test a scenario to predict the outcome if the patient is diabetic or non-diabetic

---

## 7. Final Recommendations

### 7.1 Deployment

* The ensemble model can be deployed as a web service API for use in clinical systems.
* Input: Patient data (e.g., Glucose, BMI, Age)
* Output: Probability of diabetes risk and status (e.g., "Diabetic" / "Non-Diabetic")

### 7.2 Medical Recommendation Integration

* Flag high-risk patients for further diagnostic tests.
* Recommend lifestyle intervention programs based on risk score.

### 7.3 Limitations

* The dataset is limited to a specific population (Pima Indian women).
* More features like HbA1c, cholesterol, and family history could improve accuracy.

---

## 8. Conclusion

This project demonstrates the effectiveness of machine learning in identifying individuals at risk for diabetes. With a robust ensemble model achieving nearly 88% accuracy and strong F1-score, this solution has the potential to support early diagnosis efforts and reduce healthcare burdens. Future work may involve integrating additional patient data, real-time monitoring, and broader demographic data for generalization.


## 9. Outline of project

Jupyter Notebook - Capstone_MM - Final_Project.ipynb

