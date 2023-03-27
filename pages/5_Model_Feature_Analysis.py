import matplotlib.pyplot as plt
import shap
import streamlit as st
from streamlit_shap import st_shap


def all_shap_values():
        st.header("Shap Values for All Data")
        plt.clf()
        explainer = shap.TreeExplainer(st.session_state.model.clf)
        shap_values = explainer.shap_values(st.session_state.X_train_OE)
        st_shap(shap.summary_plot(shap_values, st.session_state.X_train_OE))
        st_shap(shap.plots.force(explainer.expected_value, shap_values[:1000, :],
                                 st.session_state.X_train_OE.iloc[:1000, :]),
                height=400)

if __name__ == '__main__':
        st.header('Shap Values for the Machine Learning Model')
        tab1, tab2 = st.tabs(['All Shap Values', 'Mean Shap Values'])


        plt.clf()
        explainer = shap.TreeExplainer(st.session_state.model.clf, st.session_state.X_train_OE)
        shap_values = explainer(st.session_state.X_train_OE)
        st_shap(shap.plots.bar(shap_values))
