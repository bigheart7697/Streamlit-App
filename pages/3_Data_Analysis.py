import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

services = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']


# creates a sunburst plot based on gender and seniority for customers who unsubscribed
def gender_seniority_plot():
    tcc = st.session_state.tcc
    sunburst_values = [tcc[tcc.Churn == 1].count().gender,
                       tcc[(tcc.Churn == 1) & (tcc.gender == 'Male')].count().gender,
                       tcc[(tcc.Churn == 1) & (tcc.gender == 'Female')].count().gender,
                       tcc[(tcc.Churn == 1) & (tcc.gender == 'Male') & (tcc.SeniorCitizen == 1)].count().gender,
                       tcc[(tcc.Churn == 1) & (tcc.gender == 'Male') & (tcc.SeniorCitizen == 0)].count().gender,
                       tcc[(tcc.Churn == 1) & (tcc.gender == 'Female') & (tcc.SeniorCitizen == 1)].count().gender,
                       tcc[(tcc.Churn == 1) & (tcc.gender == 'Female') & (tcc.SeniorCitizen == 0)].count().gender]
    sunburst_data = pd.DataFrame({'labels': ['Churn Customers', 'Male', 'Female', 'M-Senior', 'M-Non-Senior',
                                             'F-Senior', 'F-Non-Senior'],
                                  'parents': ['', 'Churn Customers', 'Churn Customers', 'Male', 'Male', 'Female',
                                              'Female'],
                                  'values': sunburst_values})

    fig = px.sunburst(sunburst_data,
                      names='labels',
                      parents='parents',
                      values='values',
                      branchvalues='total')
    st.header('Gender-Seniority Churn Sunburst Chart')
    st.markdown('This chart represents what percent of each gender and seniority level decided to unsubscribe.\nThe'
                ' first level shows the stat among genders. Then, the second level shows the percentage of each gender'
                ' based on the seniority level. Note that F and M in M-Senior, M-Non-Senior, F-Senior, F-Non-Senior '
                'refers to Female and Male.')
    st.plotly_chart(fig)
    st.caption('Sunburst chart based on gender and seniority level')


# Creates two bar chart based on the rate of people who unsubscribe
# first plot is based on (number of people who unsubscribed and use the service) / (number of people who unsubscribe)
# second one is based on (number of people who unsubscribed and use the service) / (number of people who use service)
def service_bar_plots():
    tcc = st.session_state.tcc
    bar_values = [{}, {}]
    for service in services:
        nominator = tcc[(tcc.Churn == 1) & (tcc[service] == "Yes")].count().gender
        denominator = tcc[(tcc.Churn == 1)].count().gender
        bar_values[0][service] = nominator / denominator
        denominator = tcc[(tcc[service] == "Yes")].count().gender
        bar_values[1][service] = nominator / denominator
    st.header('Service-Based Churn Bar Charts')
    st.subheader('Service Usage among Churn Customers Bar Chart')
    st.markdown('This bar chart is based on the usage rate of each service among people who decided to unsubscribe.'
                ' In other words each bar\'s value is calculated based on the following formula:\n'
                '(number of people who unsubscribe and use the service) / (number of people who unsubscribe)')
    st.bar_chart(bar_values[0])
    st.caption('Service usage rate among the customers who unsubscribe')
    st.subheader('Churn Rate among Users of Service Bar Chart')
    st.markdown('This bar chart is based on the churn rate of customers who use each service.'
                ' In other words each bar\'s value is calculated based on the following formula:\n'
                '(number of people who unsubscribe and use the service) / (number of people who use the service)')
    st.bar_chart(bar_values[1])
    st.caption('Churn rate among the customers who use each service')


# Creates a scatter plot based on people who unsubscribe with x_axis as tenure and y-axix as monthly charges
def monthly_charge_tenure_scatter_plot():
    tcc = st.session_state.tcc
    fig = px.scatter(tcc,
                     x="tenure",
                     y="MonthlyCharges",
                     color="Churn")
    st.subheader('Tenure-MonthlyCharges Scatter Chart')
    st.markdown('This chart scatters the data based on the tenure amount and monthly charges. Tenure is shown on the '
                'x-axis. Monthly charges have been shown on the y-axis. The color of the point shows whether that'
                ' person has unsubscribed the services')
    st.plotly_chart(fig)
    st.caption('Scatter chart based on tenure and monthly charges')


# Creates a histogram based on the number of people who churn with x_axis as monthly charges
def monthly_charge_churn_histogram():
    tcc = st.session_state.tcc
    st.subheader('Monthly Charges Histogram Chart')
    st.markdown('This is a histogram based on monthly charges. Each bar show the number of data in that range. '
                'The x-axis show the monthly charges; the y-axis show the count of data. The color represents '
                'whether that person unsubscribed or not. Each data also has been show on the top of the chart.')
    fig = px.histogram(tcc, x="MonthlyCharges", color='Churn', histfunc='count', marginal="rug")
    st.plotly_chart(fig)
    st.caption('Monthly charge data count histogram chart')


# This function turns non float amounts to 0
def refine_total_charges(tcc):
    charges = []
    for index, row in tcc.iterrows():
        try:
            f_value = float(row['TotalCharges'])
            charges.append(f_value)
        except ValueError:
            charges.append(0.0)
    return charges


# Creates a distplot for the number of people who unsubscribe based on the total charges
def total_charges_distplot():
    tcc = st.session_state.tcc.copy()
    tcc['TotalCharges'] = refine_total_charges(tcc)
    churn_charges = pd.to_numeric(tcc[tcc['Churn'] == 1]['TotalCharges'], errors='coerce')
    non_churn_charges = pd.to_numeric(tcc[tcc['Churn'] == 0]['TotalCharges'], errors='coerce')
    group_labels = ['Yes', 'No']
    hist_data = [churn_charges, non_churn_charges]
    fig = ff.create_distplot(hist_data, group_labels, bin_size=100.)
    st.subheader('Total Charges Distribution Chart')
    st.markdown('This is a distribution chart based on total charges. It shows the distribution of data over total'
                ' charges. The x-axis shows the total charges; the y-axis shows the distribution. The colors represent'
                ' whether the customer unsubscribed or not. Each data also has been shown on the bottom.')
    st.plotly_chart(fig)
    st.caption('Total charges distribution chart')


if __name__ == '__main__':
    st.title('Data Analytic Plots')
    tab1, tab2, tab3 = st.tabs(["Personal Data", "Service Data", "Payment Data"])
    try:
        with tab1:
            if st.session_state.tcc is not None:
                gender_seniority_plot()
        with tab2:
            if st.session_state.tcc is not None:
                service_bar_plots()
        with tab3:
            if st.session_state.tcc is not None:
                st.header('Payment-Based Charts')
                monthly_charge_tenure_scatter_plot()
                monthly_charge_churn_histogram()
                total_charges_distplot()
    except:
        st.error('Your data file is corrupted!')
