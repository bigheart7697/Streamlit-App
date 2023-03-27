import timeit
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st
from streamlit_shap import st_shap
from IPython.core.interactiveshell import InteractiveShell
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import streamlit.components.v1 as components

from XGboost import XGBmodel

warnings.filterwarnings('ignore')
InteractiveShell.ast_node_interactivity = 'all'
shap.initjs()

category_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                 'PaymentMethod']

numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

session_state_parameters = ['parameters', 'reports', 'tcc', 'model', 'roc_plot', 'confusion_matrix_plot', 'X_train_OE',
                            'X_test_OE', 'y_train', 'y_test']


# Initializes the streamlit session_state
def initialize_session_state():
    for param in session_state_parameters:
        if param not in st.session_state:
            if param == 'parameters' or param == 'reports':
                st.session_state[param] = []
            else:
                st.session_state[param] = None

if __name__ == '__main__':
    st.title('Welcome to the Churn Analysis app')
    st.text('This is an app that helps analyzing data related to churn rate. ')
    initialize_session_state()
