# Loan Eligibility Prediction using Decision Tree
![GitHub](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange)
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-green)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4%2B-yellow)

## Table of Contents
1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Dataset Description](#dataset-description)
4. [Objectives](#objectives)
5. [Methodology](#methodology)
   - [Data Preprocessing](#data-preprocessing)
   - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
   - [Feature Engineering](#feature-engineering)
   - [Model Selection and Training](#model-selection-and-training)
   - [Model Evaluation](#model-evaluation)
6. [Results](#results)
7. [Future Work](#future-work)

---

## Overview
In today's fast-paced world, financial institutions like Dream Housing Finance Company are increasingly relying on automation to streamline their processes. One such critical process is determining the eligibility of customers for home loans. Automating this process not only saves time but also ensures consistency and accuracy in decision-making.

---

## Problem Statement
The primary objective of this project is to build a predictive model that can automatically determine whether a customer is eligible for a home loan based on various parameters provided during the online application process. We will use a Decision Tree algorithm to achieve this. Decision Trees are a popular choice for classification tasks due to their simplicity, interpretability, and ability to handle both numerical and categorical data.

---

## Dataset Description
The dataset used in this project contains information about loan applicants and their eligibility status. Each record includes several features and a binary target variable indicating whether the applicant is eligible for a loan. The dataset includes the following features:

- **Gender**: The gender of the applicant (Male/Female).
- **Married**: Marital status of the applicant (Yes/No).
- **Dependents**: Number of dependents the applicant has.
- **Education**: Educational qualification of the applicant (Graduate/Not Graduate).
- **Self_Employed**: Whether the applicant is self-employed (Yes/No).
- **ApplicantIncome**: Income of the applicant.
- **LoanAmount**: The amount of loan requested by the applicant.
- **Loan_Amount_Term**: The term of the loan in months.
- **Credit_History**: Record of the applicant's past credit behavior (Good/Bad).
- **Property_Area**: Area where the property is located (Urban/Semiurban/Rural).
- **Loan_Status**: The target variable indicating whether the applicant is eligible for the loan (Y = Eligible, N = Not Eligible).

---

## Objectives
1. **Build a predictive model** to classify loan applicants as eligible or ineligible.
2. **Evaluate the model's performance** using metrics such as accuracy, precision, recall, and F1-score.
3. **Provide insights** into the primary factors influencing loan eligibility using the dataset.

---

## Methodology

### Data Preprocessing
- **Handling Missing Values:** Missing values were imputed using mode for categorical features and median for numerical features.
- **Outlier Treatment:** Outliers were capped using the Interquartile Range (IQR) method.
- **Balancing the Dataset:** The dataset was initially imbalanced, with more eligible applicants than ineligible ones. SMOTE (Synthetic Minority Oversampling Technique) was applied to balance the classes.
- **Encoding Categorical Variables:** Categorical variables were encoded using label encoding and one-hot encoding to make them suitable for machine learning algorithms.

### Exploratory Data Analysis (EDA)
- **Distribution Analysis:** Histograms and boxplots were used to visualize the distribution of features.
- **Correlation Analysis:** A correlation heatmap was generated to understand the relationships between features.
- **Target Variable Analysis:** The distribution of the target variable (`Loan_Status`) was analyzed to identify class imbalance.

### Feature Engineering
- **New Features:** A new feature, `Income_to_Loan_Ratio`, was created to capture the relationship between applicant income and loan amount.
- **Feature Dropping:** Irrelevant features like `Gender`, `Self_Employed`, and `Loan_Amount_Term` were removed to simplify the model and improve performance.

### Model Selection and Training
- **Algorithm Used:** Decision Tree Classifier was selected for its interpretability and ability to handle both numerical and categorical data.
- **Hyperparameter Tuning:** Grid Search with Cross-Validation was used to optimize hyperparameters such as `max_depth`, `min_samples_split`, and `min_samples_leaf`.
- **Training:** The model was trained on 80% of the dataset, with 20% reserved for testing.

### Model Evaluation
- **Metrics:** Accuracy, precision, recall, and F1-score were used to evaluate model performance.
- **Confusion Matrix:** A confusion matrix was generated to visualize the performance of the model.
- **Feature Importance:** For Decision Trees, feature importance was analyzed to understand which features contributed most to the predictions.

---

## Results

### Model Performance
- **Initial Decision Tree:**
  - Accuracy: 74%
  - Precision: 79%
  - Recall: 82%
  - F1-Score: 80%
- **Optimized Decision Tree:**
  - Best F1-Score: 81.6%

### Key Insights
- The dataset was initially imbalanced, with more eligible applicants than ineligible ones. SMOTE was applied to balance the classes.
- After hyperparameter tuning, the optimized Decision Tree model achieved an F1-score of **81.6%**, demonstrating the effectiveness of optimization techniques.
- Feature importance analysis revealed that `Credit_History`, `Income_to_Loan_Ratio`, and `LoanAmount` were the most influential features in predicting loan eligibility.

---

## Future Work
- **Experiment with Ensemble Methods:** Explore algorithms like Random Forest or Gradient Boosting to potentially outperform the single Decision Tree model.
- **Threshold Tuning:** Adjust the decision threshold to prioritize recall for ineligible applicants, ensuring fewer risky loans are approved.
- **Incorporate Additional Data:** Gather more features, such as employment history, detailed credit scores, or external datasets, to enhance the model's predictive power.
- **Deploy the Model:** Develop a web application or API for real-time loan eligibility prediction.

---
