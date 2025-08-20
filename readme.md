# Insurance Cost Prediction Project

## Problem Statement

Insurance companies face the critical challenge of accurately predicting health insurance costs for individuals to set appropriate premiums. Traditional methods often lack the account for individual differences, leading to potential financial losses for insurers or unfairly high premiums for policyholders. This project aims to leverage machine learning techniques to predict insurance costs tailored to individual profiles, thereby enhancing pricing precisio and improving customer satisfaction.

## Target Metric

The primary target metric for this regression problem is the **PremiumPrice**, which represents the health insurance cost in currency. Our goal is to minimize the prediction error for this continuous variable. The key evaluation metric used for model performance is **RMSE**, which measures the proportion of the variance in the dependent variable that is predictable from the independent variables. Additionally, **R-squared** and **Mean Absolute Error (MAE)** were considered to quantify prediction errors.

## Steps Taken to Solve the Problem

### 1. Exploratory Data Analysis (EDA)

*   **Dataset Overview:** The dataset comprises 986 records with 11 attributes, including demographic information (Age, Height, Weight), health conditions (Diabetes, BloodPressureProblems, AnyTransplants, AnyChronicDiseases, KnownAllergies, HistoryOfCancerInFamily), NumberOfMajorSurgeries, and the target variable PremiumPrice.
*   **Data Quality:** The dataset was found to be clean with no missing values.
*   **Distribution Analysis:** Premium prices exhibited a right-skewed distribution. Distributions of binary health conditions were analyzed, revealing their importance in the dataset.
*   **Feature Engineering:**
    *   **BMI (Body Mass Index):** Calculated from Height and Weight to provide a comprehensive measure of body composition.
    *   **Age Groups:** Categorized Age into 'Young' (18-30), 'Middle' (31-50), and 'Senior' (51-66) to capture non-linear age effects.
    *   **Health Score:** A composite score created by summing up all binary health condition indicators, providing an aggregated measure of an individual's health condition.
    *   **Age Health Surgery:** Multiplicative value of Age, Health and Number of Surgeries
    *   **Surgery Per Age:** Provides an information where people had more surgeries in early stage or not
    *   **Age BMI Interaction:**
    *   **Risk Density:** Its the health scored divided by Age

### 2. Hypothesis Testing

Statistical tests were conducted to validate relationships observed during EDA and to confirm the statistical significance of factors influencing insurance costs.

*   **Chi-square Tests:** Revealed significant associations (p < 0.05) between:
    *   Diabetes, BloodPressureProblems, KnownAllergies, HistoryOfCancerInFamily, and NumberOfMajorSurgeries.
    *   NumberOfMajorSurgeries and Age Group.
    *   NumberOfMajorSurgeries and Health Score.
*   **T-tests and ANOVA:** Confirmed significant differences in PremiumPrice based on:
    *   Presence of various health conditions (Diabetes, BloodPressureProblems, ChronicDiseases, etc.).
    *   Different Age Groups.
    *   Varying Health Score categories.

These tests provided strong statistical evidence that age, number of major surgeries, and various health conditions are significant predictors of insurance premiums.

### 3. Machine Learning Modeling

*   **Data Preprocessing for Modeling:** Numerical features were scaled using `StandardScaler` to ensure uniform contribution to the models.
*   **Models Tested:** A range of regression models were evaluated:
    *   Linear Regression
    *   Decision Tree Regressor
    *   Random Forest Regressor
    *   Gradient Boosting Regressor
    *   XGBoost Regressor
*   **Model Evaluation:** K-fold cross-validation was used to assess model performance and ensure generalization to unseen data. R² was the primary metric, complemented by MSE and MAE.

## Insights and Recommendations

### Key Insights

1.  **Age is the Dominant Factor:** Age consistently emerged as the strongest predictor of insurance premiums, aligning with the general understanding that health risks increase with age.
2.  **Surgical History Matters:** The number of major surgeries an individual has undergone significantly impacts their premium costs, indicating a higher likelihood of future medical expenses.
3.  **Cumulative Health Burden:** The engineered `Health Score` proved highly effective, demonstrating that the cumulative effect of multiple health conditions leads to substantially higher premiums.
4.  **BMI and Weight are Crucial:** Body Mass Index and weight are important risk factors, highlighting the link between physical health metrics and insurance costs.
5.  **Family History's Role:** A family history of cancer also contributes to increased premium pricing.
6.  **High Predictive Accuracy:** The models, particularly ensemble methods, achieved high accuracy in predicting premiums, indicating their practical utility.

## Final Scores Achieved

During cross-validation, the following performance metrics were observed for the tested models:

As shown in the table above, the **RMSE (Root Mean Squared Error)** was a key metric for evaluating our models, as it provides a measure of the average magnitude of the errors in the same units as the target variable. Here's a summary of the results:

*   **Linear Regression:** RMSE = 3542.13, MAE = 2419.16, R² = 0.678
*   **Decision Tree Regressor:** RMSE = 3889.32, MAE = 1147.06, R² = 0.612
*   **Random Forest Regressor:** RMSE = **2858.16**, MAE = 1249.14, R² = 0.791 (Best performing model in terms of RMSE)
*   **Gradient Boosting Regressor:** RMSE = 3109.07, MAE = 1724.89, R² = 0.752
*   **XGBoost Regressor:** RMSE = 3039.57, MAE = 1509.54, R² = 0.763

The **Random Forest Regressor** demonstrated superior performance with the lowest RMSE of **2858.16**, indicating its strong capability in accurately predicting insurance costs and minimizing prediction errors. This model is the most reliable tool for practical application based on these metrics.