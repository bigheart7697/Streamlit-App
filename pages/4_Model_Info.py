import warnings

import pandas as pd
import shap
import streamlit as st
from IPython.core.interactiveshell import InteractiveShell

warnings.filterwarnings('ignore')
InteractiveShell.ast_node_interactivity = 'all'
shap.initjs()

category_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                 'PaymentMethod']

numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']


# shows a plot with a button to download it as an image
def show_image_with_download(name, figure):
    st.pyplot(figure)
    fn = 'static\\' + name
    figure.savefig(fn)
    with open(fn, "rb") as img:
        st.download_button(
            label="Download image",
            data=img,
            file_name=fn,
            mime="image/png"
        )


# Compares different parameters for each model that has been run until now in this session
def compare_models():
    data = {'learning_rate': [], 'max_depth': [], 'n_estimators': [], 'colsample_bytree': []}
    for el in st.session_state.parameters:
        data['learning_rate'].append(el['learning_rate'])
        data['max_depth'].append(el['max_depth'])
        data['n_estimators'].append(el['n_estimators'])
        data['colsample_bytree'].append(el['colsample_bytree'])
    parameters = pd.DataFrame(data=data)
    st.subheader('Table of Parameters for Each Model')
    st.markdown('This table contains parameters for all the models that have been run in this session:')
    st.write(parameters)
    churn_data = {'precision': [], 'recall': [], 'f1-score': []}
    accuracy = []
    support = []
    macro_average = {'precision': [], 'recall': [], 'f1-score': []}
    weighted_average = {'precision': [], 'recall': [], 'f1-score': []}
    # print(st.session_state.reports)
    for el in st.session_state.reports:
        churn_data['precision'].append(el['1']['precision'])
        churn_data['recall'].append(el['1']['recall'])
        churn_data['f1-score'].append(el['1']['f1-score'])

        support.append([el['0']['support'], el['0']['support']])

        accuracy.append(el['accuracy'])

        macro_average['precision'].append(el['macro avg']['precision'])
        macro_average['recall'].append(el['macro avg']['recall'])
        macro_average['f1-score'].append(el['macro avg']['f1-score'])

        weighted_average['precision'].append(el['weighted avg']['precision'])
        weighted_average['recall'].append(el['weighted avg']['recall'])
        weighted_average['f1-score'].append(el['weighted avg']['f1-score'])

    line_chart_data = {'positive churn precision': churn_data['precision'],
                       'positive churn recall': churn_data['recall'],
                       'positive churn f1-score': churn_data['f1-score'], 'accuracy': accuracy,
                       'macro average precision': macro_average['precision'],
                       'macro average recall': macro_average['recall'],
                       'macro average f1-score': macro_average['f1-score'],
                       'weighted average precision': weighted_average['precision'],
                       'weighted average recall': weighted_average['recall'],
                       'weighted average f1-score': weighted_average['f1-score']}
    data = pd.DataFrame(data=line_chart_data)
    st.subheader('Comparison Line Chart')
    st.markdown('This is chart the compares models based on the precision, recall, f1-score for positive (churn) output'
                ', macro average and weighted average. The parameters for each model can be found in the corresponding'
                ' index in the above table.')
    st.line_chart(data=data)
    st.caption('Model comparison line chart')


if __name__ == '__main__':
    st.header('Trained Machine Learning Models Information')
    tab1, tab2 = st.tabs(['Current Model Info', 'Models Comparison'])
    with tab1:
        if st.session_state.tcc is not None:
            st.header('Current Machine Learning Model Validation Report')
            st.subheader('ROC Curve')
            st.markdown('An ROC (receiver operating characteristic) curve is a plot that represents'
                        ' the performance of the model. The x-axis represents the false positive rate (the data that '
                        'falsely has been categorized as positive; the y-axis '
                        'represents the true positive rate (the data that has been correctly predicted as positive.'
                        ' The area under ROC curve (AUC) is another measurement of the model performance.')
            show_image_with_download('roc.png', st.session_state.roc_plot.figure)
            st.caption('ROC curve')
            st.subheader('Confusion Matrix')
            st.markdown('Confusion Matrix is a table that is used for measuring a classification model performance. '
                        'The rows show the true label and the columns show the predicted label. each cell has a color '
                        'which represents the quantity of the related rate.')
            show_image_with_download('confusion-matrix.png', st.session_state.confusion_matrix_plot.figure)
            st.caption('Confusion matrix')
    with tab2:
        st.header('Different Models Comparison')
        compare_models()
