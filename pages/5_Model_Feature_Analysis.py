import matplotlib.pyplot as plt
import shap
import streamlit as st
from streamlit_shap import st_shap

# Plots the summary and data visualization plots for Shap values
def all_shap_values():
    st.header("Shap Values for All Data")
    st.markdown('Shap Values are used to see impact of each feature in the Machine Learning model. The first two shap'
                'plots are the summary plot and the value plot for all the data. In summary plot we have each feature'
                'on the left. Each data and their Shap values for that feature are shown in the plot. The second chart'
                ' shows all the data and their Shap values (sum of shap values or for the feature). On the top of the '
                'chart you can choose how to sample data. On the left, you can choose whether see the sum Shap values'
                ' or Shap values for a specific feature.')
    plt.clf()
    explainer = shap.TreeExplainer(st.session_state.model.clf)
    shap_values = explainer.shap_values(st.session_state.X_train_OE)
    st.subheader('Shap Summary Plot')
    st_shap(shap.summary_plot(shap_values, st.session_state.X_train_OE))
    st.caption('The summary plot for Shap values of each data for each feature')
    st.subheader('Shap All Data Plot')
    st_shap(shap.plots.force(explainer.expected_value, shap_values[:1000, :],
                             st.session_state.X_train_OE.iloc[:1000, :]), height=400)
    st.caption('A visualization plot that shows all the Shap values for 1000 data and for all the features')


# plots  the mean Shap value plot
def mean_shap_values():
    st.header("Mean Shap Values")
    st.markdown(
        'This plot shows the mean Shap values for each feature. This shows how important each feature is '
        'in the Machine Learning model.')
    plt.clf()
    explainer = shap.TreeExplainer(st.session_state.model.clf, st.session_state.X_train_OE)
    shap_values = explainer(st.session_state.X_train_OE)
    st_shap(shap.plots.bar(shap_values))
    st.caption('Mean Shap values')


if __name__ == '__main__':
    st.header('Shap Values for the Machine Learning Model')
    tab1, tab2 = st.tabs(['All Shap Values', 'Mean Shap Values'])
    with tab1:
        all_shap_values()
    with tab2:
        mean_shap_values()
