import os 
import glob
import warnings
import pandas as pd

pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')

model_files = glob.glob("insurance_cost_predictions_best_model_*.pkl")
if model_files:
    MODEL_FILE_NAME = max(model_files, key=lambda x: os.path.getmtime(x))
else:
    MODEL_FILE_NAME = None

# MODEL_FILE_NAME = 'insurance_cost_prediction_best_model_rf.pkl'
