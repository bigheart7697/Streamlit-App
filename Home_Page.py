import warnings

import shap
import streamlit as st
from IPython.core.interactiveshell import InteractiveShell

warnings.filterwarnings('ignore')
InteractiveShell.ast_node_interactivity = 'all'
shap.initjs()

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
    st.title('Churn Rate Analysis App')
    st.text('This is an app that helps analyzing data related to churn rate.')
    initialize_session_state()
    st.header('What is the churn rate?')
    st.markdown('Churn rate is the percentage of subscribers of a service that are leaving. Businesses use this rate'
                ' to measure their performance. Thus, it is important to identify the parameters that effect this rate'
                '. Each business tries to decrease its churn rate. In order to do that, the business should identify'
                ' category of its subscribers who are leaving. It should be able to identify which part of the service'
                ' is not satisfying and be able to effectively predict which customers are going to leave.')
    st.header('How does this app contribute to the problem?')
    st.markdown('In this app, client uploads the data for the past customers. Then app provides some charts related to'
                ' churn rate to help the business identify the reason for losing customers. '
                'The app trains an XGBoost model to predict the customers who churn. There are plots for'
                ' measuring the model\'s success, and some Shap value charts to see which parameters are more important '
                'in the Machine Learning model. This would again help the business to identify the main parameter of '
                'unsubscribing. So, it can improve it and reduce the churn rate.')
    st.header('Different Components')
    st.markdown('On the run the model page, the user inputs the data file and can tune the parameters for the'
                ' Machine Learning model, then enable the fitting for the model. After the model was run, '
                'on the Data Analysis page user can find charts related to churn rate based on different parameters.'
                'On page Model Info, user can see how the model performed. Also, user can compare different models'
                ' that had been trained. Finally, on page Model Feature Analysis, Shaply values for the model are '
                'presented. Based on these values, the reason for leaving the business can be identified. '
                'For more information, please refer to the User Guide page.')
