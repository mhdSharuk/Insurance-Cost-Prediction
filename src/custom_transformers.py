import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor


class FeatureCreator(BaseEstimator, TransformerMixin):
    def __init__(self, binary_cols):
        self.binary_cols = binary_cols

    def overall_risk_category(self, row):
        risk_score = 0
        risk_score += row['age'] > 50
        risk_score += row['bmi'] > 30
        risk_score += row['health_score'] >= 3
        risk_score += row['any_transplants']
        risk_score += row['number_of_major_surgeries'] >= 2
        if risk_score >= 4:
            return 'critical'
        elif risk_score >= 3:
            return 'high'
        elif risk_score >= 2:
            return 'medium'
        else:
            return 'low'

    def bmi_category(self, bmi):
        if bmi < 18.5:
            return 'underweight'
        elif bmi < 25:
            return 'normal'
        elif bmi < 30:
            return 'overweight'
        else:
            return 'obese'

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['bmi'] = np.round(df['weight'] / ((df['height'] / 100) ** 2), 2)
        bins = [17, 25, 35, 45, 55, 66, float('inf')]
        labels = [1, 2, 3, 4, 5, 6]
        df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels).astype(int)
        df['health_score'] = df[self.binary_cols].sum(axis=1)
        df['age_health_surgery'] = np.round(df['age'] * df['health_score'] * df['number_of_major_surgeries'], 3)
        df['surgery_per_age'] = np.round(df['number_of_major_surgeries'] / df['age'], 2)
        df['age_bmi_interaction'] = df['age'] * df['bmi']
        df['bmi_health_interaction'] = df['bmi'] * df['health_score']
        df['age_squared'] = df['age'] ** 2
        df['overall_risk_category'] = df.apply(self.overall_risk_category, axis=1)
        df['bmi_category'] = df['bmi'].apply(self.bmi_category)
        df['high_cost_condition'] = ((df['any_transplants'] == 1) | (df['history_of_cancer_in_family'] == 1)).astype(int)
        df['metabolic_syndrome_proxy'] = ((df['diabetes'] == 1) &
                                          (df['blood_pressure_problems'] == 1) &
                                          (df['bmi'] > 30)).astype(int)
        df['risk_density'] = df['health_score'] / df['age']
        return df


class ImputerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, binary_cols):
        self.binary_cols = binary_cols
        self.imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=50, random_state=42),
            max_iter=10,
            random_state=69
        )

    def fit(self, X, y=None):
        X_copy = X.copy()
        for col in self.binary_cols:
            X_copy[col] = X_copy[col].astype(float)
        self.imputer.fit(X_copy)
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.binary_cols:
            X_copy[col] = X_copy[col].astype(float)
        df_imputed = pd.DataFrame(self.imputer.transform(X_copy), columns=X_copy.columns)
        for col in self.binary_cols:
            df_imputed[col] = (df_imputed[col] >= 0.7).astype(int)
        return df_imputed
