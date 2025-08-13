import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from features import create_features

def handle_feature_scaling(df, numeric_cols):
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def handle_ordinal_encoding(df, columns_list, categories_list):
    encoder = OrdinalEncoder(categories=categories_list)
    df[columns_list] = encoder.fit_transform(df[columns_list])
    return df


def process_input(
    age, height, weight, diabetes_val, bp_val, transplants_val,
    chronic_val, allergies_val, cancer_val, surgeries
):
    df = pd.DataFrame([{
        "age": age,
        "height": height,
        "weight": weight,
        "diabetes": diabetes_val,
        "blood_pressure_problems": bp_val,
        "any_transplants": transplants_val,
        "any_chronic_diseases": chronic_val,
        "known_allergies": allergies_val,
        "history_of_cancer_in_family": cancer_val,
        "number_of_major_surgeries": surgeries
    }])

    df = create_features(df)
    df = df[['age', 'diabetes', 'blood_pressure_problems', 'any_transplants',
       'any_chronic_diseases', 'height', 'weight', 'known_allergies',
       'history_of_cancer_in_family', 'number_of_major_surgeries', 'bmi',
       'age_group', 'health_score', 'age_health_surgery', 'surgery_per_age',
       'age_bmi_interaction', 'bmi_health_interaction', 'age_squared',
       'overall_risk_category', 'bmi_category', 'high_cost_condition',
       'metabolic_syndrome_proxy', 'risk_density']]
    
    df = handle_ordinal_encoding(df, ['overall_risk_category', 'bmi_category'],
                             [['low', 'medium', 'high', 'critical'],
                              ['underweight', 'normal', 'overweight', 'obese']])
    
    numeric_cols = ['age', 'height', 'weight', 'number_of_major_surgeries', 
                    'bmi', 'age_health_surgery',
                    'surgery_per_age', 'age_bmi_interaction',
                    'bmi_health_interaction', 'age_squared',
                    'risk_density']

    df = handle_feature_scaling(df, numeric_cols)

    return df