import pandas as pd
# from src.features import create_features
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

def process_input(
    age, height, weight, diabetes_val, bp_val, transplants_val,
    chronic_val, allergies_val, cancer_val, surgeries
):
    df = pd.DataFrame([{
        "age": age,
        "diabetes": diabetes_val,
        "blood_pressure_problems": bp_val,
        "any_transplants": transplants_val,
        "any_chronic_diseases": chronic_val,
        "height": height,
        "weight": weight,
        "known_allergies": allergies_val,
        "history_of_cancer_in_family": cancer_val,
        "number_of_major_surgeries": surgeries
    }])

    return df