import numpy as np
import pandas as pd

def overall_risk_category(row):
    risk_score = (
        (row['age'] > 50) +
        (row['bmi'] > 30) +
        (row['health_score'] >= 3) +
        row['any_transplants'] +
        (row['number_of_major_surgeries'] >= 2)
    )

    if risk_score >= 4:
        return 'critical'
    elif risk_score >= 3:
        return 'high'
    elif risk_score >= 2:
        return 'medium'
    else:
        return 'low'

def bmi_category(bmi):
    if bmi < 18.5:
        return 'underweight'
    elif bmi < 25:
        return 'normal'
    elif bmi < 30:
        return 'overweight'
    else:
        return 'obese'

def create_features(df):
    df['bmi'] = np.round(df['weight'] / ((df['height'] / 100) ** 2), 2)
    bins = [17, 25, 35, 45, 55, 66]
    labels = [1, 2, 3, 4, 5]
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

    health_issues = [
        'diabetes', 'blood_pressure_problems', 'any_transplants',
        'any_chronic_diseases', 'known_allergies',
        'history_of_cancer_in_family'
    ]
    df['health_score'] = df[health_issues].sum(axis=1)

    df['age_health_surgery'] = np.round(
        df['age'] * df['health_score'] * df['number_of_major_surgeries'], 3
    )
    df['surgery_per_age'] = np.round(
        df['number_of_major_surgeries'] / df['age'], 2
    )
    df['age_bmi_interaction'] = df['age'] * df['bmi']
    df['bmi_health_interaction'] = df['bmi'] * df['health_score']
    df['age_squared'] = df['age'] ** 2
    df['overall_risk_category'] = df.apply(overall_risk_category, axis=1)
    df['bmi_category'] = df['bmi'].apply(bmi_category)

    df['high_cost_condition'] = (
        (df['any_transplants'] == 1) |
        (df['history_of_cancer_in_family'] == 1)
    ).astype(int)

    df['metabolic_syndrome_proxy'] = (
        (df['diabetes'] == 1) &
        (df['blood_pressure_problems'] == 1) &
        (df['bmi'] > 30)
    ).astype(int)

    df['risk_density'] = df['health_score'] / df['age']
    return df
