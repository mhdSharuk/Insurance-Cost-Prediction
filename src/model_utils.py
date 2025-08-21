import pickle
import streamlit as st
from src.config import MODEL_FILE_NAME, PREPROCESSING_FILE_NAME

@st.cache_resource
def load_model():
    with open(f'models/{MODEL_FILE_NAME}', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

@st.cache_resource
def load_preprocessing_model():
    with open(f'models/{PREPROCESSING_FILE_NAME}', 'rb') as preprocessing_file:
        model = pickle.load(preprocessing_file)
    return model