import warnings

import shap
import streamlit as st
from IPython.core.interactiveshell import InteractiveShell

warnings.filterwarnings('ignore')
InteractiveShell.ast_node_interactivity = 'all'
shap.initjs()

if __name__ == '__main__':
    st.title('Churn Rate Application User Guide')
    st.header('How to begin?')
    st.markdown('You should start with Run Model page. In this page you can upload a data file and tune the learning'
                ' parameters. Note that application will not show any charts or data if you do not upload data and run'
                ' a Machine Learning model. First, you need to choose a file that contains the data related to the churn'
                ' rate. In the beginning of the form, click on the \'Browse files\' button and choose a file.  Note that '
                'the data should contain following columns otherwise it raises error:')
    st.markdown('gender, SeniorCitizen, Partner,'
                ' Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup,'
                'DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling,'
                'PaymentMethod, tenure, MonthlyCharges, TotalCharges')
    st.markdown(' Then you should tune the learning parameters.'
                'Please follow the format given each text field. That is start the input with \'[\', fill out the list'
                ' of parameters seperated by , (comma) and end it with \']\'. Even if you are inputting one parameter'
                ' in each field, you should begin with \'[\' and end with \']\'. Otherwise, or you will face an error.'
                ' When all the inputs are ready click on \'Enable Fitting\' button, so that the model get trained. When'
                ' the success message gets printed on the page, the model is ready.')
    st.header('Data Analysis page')
    st.markdown('First note that this page will not show anything if you have not run the model. This page contains'
                ' charts related to the churn rate based on different attributes of customers. Attributes like gender,'
                ' seniority level, services they purchased and their billing information.')
    st.header('Model Info page')
    st.markdown('This page contains curves and charts related to the performance of the model. You can see how the'
                ' model performs based on these metrics to predict the customers who unsubscribe. It also compares '
                'different models that has been run until now in this session. It shows the parameters for each model'
                ' in a table and plots a line chart based on the precision, recall and f1-score measures. Again,'
                ' nothing will be shown if you do not run a model.')
    st.header('Model Feature Analysis')
    st.markdown('This page gives advanced information on how each feature is involved in the Machine Learning model.'
                ' Based on Shap values of the features, we can see how important that feature is in identifying the '
                'customers who leave. It may help to identify how to improve and prevent customers from leaving.')
