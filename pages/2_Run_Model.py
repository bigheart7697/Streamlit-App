import timeit
import warnings

import numpy as np
import pandas as pd
import shap
import streamlit as st
from IPython.core.interactiveshell import InteractiveShell
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

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


# creates a integer list fro the string
def create_int_list_from_string(s):
    return [int(num) for num in s[1:-1].split(',')]


# creates a float list from the string
def create_float_list_from_string(s):
    return [float(num) for num in s[1:-1].split(',')]


# Data preprocessing to format feature columns and get prediction target column 'labels'
def preprocess_data(tcc):
    target = 'Churn'
    ID_col = 'customerID'
    assert len(category_cols) + len(numeric_cols) + 2 == tcc.shape[1]

    tcc['TotalCharges'] = tcc['TotalCharges'].apply(lambda x: x if x != ' ' else np.nan).astype(float)
    tcc['MonthlyCharges'] = tcc['MonthlyCharges'].astype(float)

    tcc['TotalCharges'] = tcc['TotalCharges'].fillna(0)

    tcc['Churn'].replace(to_replace='Yes', value=1, inplace=True)
    tcc['Churn'].replace(to_replace='No', value=0, inplace=True)

    features = tcc.drop(columns=[ID_col, target]).copy()
    labels = tcc['Churn'].copy()
    return features, labels


# Split input data into training dataset, and holdout testing dataset
@st.cache_data
def create_train_and_test_data(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=22)

    # Apply ordinal encoder to convert categorical values into numerical values

    ord_enc = OrdinalEncoder()
    ord_enc.fit(X_train[category_cols])

    X_train_OE = pd.DataFrame(ord_enc.transform(X_train[category_cols]), columns=category_cols)
    X_train_OE.index = X_train.index
    X_train_OE = pd.concat([X_train_OE, X_train[numeric_cols]], axis=1)

    X_test_OE = pd.DataFrame(ord_enc.transform(X_test[category_cols]), columns=category_cols)
    X_test_OE.index = X_test.index
    X_test_OE = pd.concat([X_test_OE, X_test[numeric_cols]], axis=1)
    return X_train_OE, X_test_OE, y_train, y_test


# In order to find the best set of hyperparameter,
# xgboost model would try different combination of parameter values from user input lists,
# and then return the model with best accuracy
@st.cache_data
def run_model(X_train_OE, X_test_OE, y_train, y_test,
              xgb_param=None):
    if xgb_param is None:
        xgb_param = dict(learning_rate=[0.01, 0.016, 0.027, 0.046, 0.077, 0.129, 0.215, 0.359],
                         max_depth=[5],
                         n_estimators=[50],
                         colsample_bytree=[0.4]
                         )
    start = timeit.default_timer()
    xgb_model = XGBmodel(xgb_param)
    xgb_model.fit(X_train_OE, y_train)
    stop = timeit.default_timer()
    print('using Time:{:.2f} s'.format(stop - start))
    precision_recal_f1_report = xgb_model.precision_recall_f1_visual(X_test_OE, y_test)
    roc_plot, confusion_matrix_plot = xgb_model.show_test_result(X_test_OE, y_test)
    return roc_plot, confusion_matrix_plot, precision_recal_f1_report, xgb_model


# Show model ROC CURV and
@st.cache_data
def show_plots(xgb_model, X_test_OE, y_test):
    return xgb_model.show_test_result(X_test_OE, y_test)


# Initializes the streamlit session_state
def initialize_session_state():
    for param in session_state_parameters:
        if param not in st.session_state:
            if param == 'parameters' or param == 'reports':
                st.session_state[param] = []
            else:
                st.session_state[param] = None


# Enables the fitting when the enable fitting button is clicked
def enable_fitting():
    if st.session_state.uploaded_file is not None:
        # create a DataFrame from the csv file
        data_load_state = st.text('Loading data...')
        st.session_state.tcc = pd.read_csv(st.session_state.uploaded_file)
        data_load_state.text("Data loaded!")

        # creates train and test data
        features, labels = preprocess_data(st.session_state.tcc)
        X_train_OE, X_test_OE, y_train, y_test = create_train_and_test_data(features, labels)
        st.session_state.X_train_OE = X_train_OE
        st.session_state.X_test_OE = X_test_OE
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test

        # tuning the parameters
        learning_rate = create_float_list_from_string(st.session_state.learning_rate)
        max_depth = create_int_list_from_string(st.session_state.max_depth)
        n_estimators = create_int_list_from_string(st.session_state.n_estimators)
        colsample_bytree = create_float_list_from_string(st.session_state.colsample_bytree)
        # Run the model
        with st.spinner('Running the model'):
            roc_plot, confusion_matrix_plot, report, model = run_model(X_train_OE, X_test_OE, y_train, y_test,
                                                                       dict(learning_rate=learning_rate,
                                                                            max_depth=max_depth,
                                                                            n_estimators=n_estimators,
                                                                            colsample_bytree=colsample_bytree))
        st.success('Running the model is done!')

        # Update the session state to store the model data
        st.session_state.parameters.append(dict(learning_rate=learning_rate,
                                                max_depth=max_depth,
                                                n_estimators=n_estimators,
                                                colsample_bytree=colsample_bytree))
        st.session_state.model = model
        st.session_state.roc_plot = roc_plot
        st.session_state.confusion_matrix_plot = confusion_matrix_plot
        st.session_state.reports.append(report)


# creates the form to get user input for tuning parameters
def create_form():
    with st.form('param_tuning'):
        uploaded_file = st.file_uploader("Choose a file", key='uploaded_file')
        learning_rate = st.text_input(label='Learning Rate',
                                      value='[0.01, 0.016, 0.027, 0.046, 0.077, 0.129, 0.215, 0.359]',
                                      key='learning_rate')
        max_depth = st.text_input(label='Max Depth',
                                  value='[5, 10, 15]',
                                  key='max_depth')
        n_estimators = st.text_input(label='n-Estimators',
                                     value='[50, 70, 90, 110, 130, 150]',
                                     key='n_estimators')
        colsample_bytree = st.text_input(label='Column Sample Bytree:',
                                         value='[0.4, 0.6, 0.8]',
                                         key='colsample_bytree')
        st.form_submit_button('Enable Fitting', on_click=enable_fitting)


if __name__ == '__main__':
    st.title('Churn Rate Predictor')
    initialize_session_state()
    create_form()