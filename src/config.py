import os 
import glob
import warnings
import pandas as pd

pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')

model_files = glob.glob("./models/insurance_cost_prediction_best_model_*.pkl")
preprocessing_file = glob.glob('./models/insurance_preprocessing_pipeline_*.pkl')

if model_files:
    latest_model_path = max(model_files, key=os.path.getmtime)
    MODEL_FILE_NAME = os.path.basename(latest_model_path)
else:
    MODEL_FILE_NAME = None

if preprocessing_file:
    latest_model_path = max(preprocessing_file, key=os.path.getmtime)
    PREPROCESSING_FILE_NAME = os.path.basename(latest_model_path)
else:
    PREPROCESSING_FILE_NAME = None

print("Latest preprocessing file:", PREPROCESSING_FILE_NAME)
print("Latest model file:", MODEL_FILE_NAME)
