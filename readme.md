# Insurance Cost Prediction Project

## Problem Statement

Insurance companies face the critical challenge of accurately predicting health insurance costs for individuals to set appropriate premiums. Traditional methods often lack the account for individual differences, leading to potential financial losses for insurers or unfairly high premiums for policyholders. This project aims to leverage machine learning techniques to predict insurance costs tailored to individual profiles, thereby enhancing pricing precisio and improving customer satisfaction.

## Target Metric

The primary target metric for this regression problem is the **PremiumPrice**, representing the health insurance cost in currency. Our goal is to minimize the prediction error for this continuous variable. The key evaluation metric used for model performance is **RMSE**, which measures the average magnitude of prediction errors in the same units as the target variable. Additionally, **R-squared** and **Mean Absolute Error (MAE)** were considered to assess how well the model explains variance and the average prediction error, respectively.

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

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| **Linear Regression** | 3542.13 | 2419.16 | 0.678 |
| **Decision Tree Regressor** | 3889.32 | 1147.06 | 0.612 |
| **Random Forest Regressor** | **2858.16** | 1249.14 | **0.791** |
| **Gradient Boosting Regressor** | 3109.07 | 1724.89 | 0.752 |
| **XGBoost Regressor** | 3039.57 | 1509.54 | 0.763 |

**Best performing model:** Random Forest Regressor with RMSE = **2858.16** and R² = **0.791**
The **Random Forest Regressor** demonstrated superior performance with the lowest RMSE of **2858.16**, indicating its strong capability in accurately predicting insurance costs and minimizing prediction errors. This model is the most reliable tool for practical application based on these metrics.

## Deployment

To make our insurance cost prediction model accessible and user-friendly, we developed a web-based application using **Streamlit**. This application allows users to input their health and demographic data and receive an estimated insurance premium in real-time.

### Project Structure for Deployment

The project repository is structured to facilitate easy deployment and understanding:

*   **`app.py`:** This is the main Streamlit application file. It handles the user interface, collects inputs, and proceeds with the prediction process.
*   **`src/`:** This directory contains all the necessary backend components, including the trained machine learning model, data preprocessing scripts, and any other utility functions required by `app.py`.

### Project Structure for Deployment

The project repository is structured to facilitate easy deployment and understanding:

```
Insurance-Cost-Prediction/
│
├── app.py                              # Main Streamlit application file
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
├── Dockerfile
├── .gitignore
├── tableau/
    ├── insurance_cost_prediction_tableau_workbook.twb
├── src/
    ├── __init__.py
    ├── config.py
    ├── features.py
    ├── model_utils.py
    ├── preprocessing.py
├── notebooks/
    ├── Insurance_Cost_Prediction.ipynb
├── models/
    
```

### Key Components

#### **`app.py`**
This is the main Streamlit application file that handles:
- User interface design and layout
- Input collection from users (age, height, weight, health conditions, etc.)
- Data validation and preprocessing
- Model prediction calls
- Results visualization and display

#### **`src/` Directory**
Contains all backend components:
- **`model/`**: Stores the trained Random Forest model (`trained_model.pkl`)
- **`preprocessing/`**: Contains the fitted scaler and feature engineering functions
- **`utils/`**: Utility functions for prediction and data validation
- **`config/`**: Configuration files and constants

### Deployment Steps

#### 1. **Local Development Setup**

```bash
# Clone the repository
git clone https://github.com/mhdSharuk/Insurance-Cost-Prediction.git
cd Insurance-Cost-Prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application locally
streamlit run app.py
```

#### 2. **Docker Deployment**

```bash
# Build the Docker image
docker build -t insurance-cost-prediction .

# Run the container
docker run -p 8501:8501 insurance-cost-prediction

# Run container in background
docker run -d -p 8501:8501 --name insurance-app insurance-cost-prediction

# View running containers
docker ps

# Stop the container
docker stop insurance-app

# Remove the container
docker rm insurance-app
```

### Application Features

#### **User Interface**
- **Input Form**: Clean, intuitive form for entering personal and health information
- **Real-time Validation**: Input validation with helpful error messages
- **Interactive Visualization**: Charts showing risk factors and premium breakdown
- **Responsive Design**: Mobile-friendly interface

#### **Prediction Pipeline**
1. **Data Collection**: User inputs collected through Streamlit widgets
2. **Feature Engineering**: Automatic calculation of BMI, Health Score, and other engineered features
3. **Preprocessing**: Data scaling using the trained StandardScaler
4. **Prediction**: Random Forest model generates premium estimate
5. **Results Display**: Premium amount with confidence intervals and risk factor analysis

### Using the Application
1. Enter your personal information (age, height, weight)
2. Select your health conditions
3. Specify the number of major surgeries
4. Click "Predict Premium" to get your estimated insurance cost
5. View detailed risk factor analysis and recommendations


## Technical Blog
[dev.to Insurance Cost Prediction](https://dev.to/mhdsharuk/insurance-cost-prediction-ai3)

## Public Streamlit Application

[Streamlit Application](https://mhd-insurance-cost-prediction.streamlit.app/)

## Contact

**Mohammed Sharuk**
- GitHub: [@mhdSharuk](https://github.com/mhdSharuk)
- Email: [msharuk589@gmail.com](mailto:msharuk589@gmail.com)

## Acknowledgments

- [Dataset source : From Scaler](https://drive.google.com/file/d/1NBk1TFkK4NeKdodR2DxIdBp2Mk1mh4AS/view?usp=drive_link) 
- Inspired by real-world insurance industry challenges
- Built with love for machine learning