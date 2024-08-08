import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.express as px
import plotly.graph_objects as go
from helper import *

st.title("Custom Training Analytics")

st.write("""
This app is a reworked version with fewer features than the original currently employed in the B737MAX Team in CAE. 
This version only has features for trend analysis and forecasting training costs to be incurred by customers for the flight simulators. 
The app is used to automate data from CSV files containing the same features, surpassing the use of Excel for automated analysis and reports generation.
""")

# Step 1: Upload CSV File
st.markdown("### **Step 1: Upload your CSV file**")
uploaded_file = st.file_uploader("", type="csv")

if uploaded_file is not None:
    # Load data
    data = load_dataset(uploaded_file)
    
    if st.button("Submit"):
        # Store data in session state
        st.session_state['data'] = data

# Ensure data is loaded and stored in session state
if 'data' in st.session_state:
    data = st.session_state['data']
    
    # Display first 10 rows
    st.header("First 10 Rows of the Dataset")
    st.write(data.head(10))

# Step 2: Data Preprocessing
if 'data' in st.session_state:
    st.markdown("### **Step 2: Data Preprocessing**")

    if 'processed_data' not in st.session_state:
        if st.button("Start Preprocessing"):
            initial_rows = data.shape[0]
            
            # Preprocess data
            processed_data = preprocess_data(data)
            
            final_rows = processed_data.shape[0]
            rows_dropped_missing = initial_rows - data.dropna().shape[0]
            rows_dropped_duplicates = data.shape[0] - data.drop_duplicates().shape[0]
            
            # Store processed data and preprocessing info in session state
            st.session_state['processed_data'] = processed_data
            st.session_state['rows_dropped_missing'] = rows_dropped_missing
            st.session_state['rows_dropped_duplicates'] = rows_dropped_duplicates

    # Display preprocessing information
    if 'processed_data' in st.session_state:
        st.write(f"Number of rows dropped due to missing values: {st.session_state['rows_dropped_missing']}")
        st.write(f"Number of rows removed due to duplicates: {st.session_state['rows_dropped_duplicates']}")

# Step 3: Exploratory Data Analysis (EDA)
if 'processed_data' in st.session_state:
    processed_data = st.session_state['processed_data']
    
    st.markdown("### **Step 3: Exploratory Data Analysis (EDA)**")
    
    if st.button("Start EDA"):
        st.session_state['eda'] = True

    if 'eda' in st.session_state:
        st.header("Exploratory Data Analysis")

        # Display Descriptive Statistics
        st.subheader("Descriptive Statistics")
        st.write(processed_data.describe())

        # Visualization 1: Distribution of a specific column (e.g., 'Training Status')
        st.subheader("Distribution of Training Status")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.countplot(x='Training Status', data=processed_data, palette='viridis', ax=ax1)
        ax1.set_title('Distribution of Training Status')
        ax1.set_xlabel('Training Status')
        ax1.set_ylabel('Count')
        st.pyplot(fig1)

        # Visualization 2: Histogram of a numerical column (e.g., 'Training Cost')
        st.subheader("Histogram of Training Cost")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.histplot(processed_data['Training Cost'], kde=True, color='blue', ax=ax2)
        ax2.set_title('Histogram of Training Cost')
        ax2.set_xlabel('Training Cost')
        ax2.set_ylabel('Frequency')
        st.pyplot(fig2)

        # Visualization 3: Pairplot for numerical columns
        st.subheader("Pairplot of Numerical Columns")
        numerical_columns = processed_data.select_dtypes(include=['float64', 'int64']).columns
        fig3 = sns.pairplot(processed_data[numerical_columns])
        st.pyplot(fig3)

# Step 4: Trend Analysis
if 'processed_data' in st.session_state:
    st.markdown("### **Step 4: Trend Analysis**")

    # Sub-section: Select Independent Variable
    st.subheader("Select Independent Variable")
    independent_variable = st.selectbox("Independent Variable (only one variable can be selected for now (i.e., Training Enrollment Date))", options=processed_data.columns.tolist(), key='independent')

    # Sub-section: Select Dependent Variable
    st.subheader("Select Dependent Variable")
    dependent_variable = st.selectbox("Dependent Variable (only one variable can be selected for now (i.e., Training Cost))", options=processed_data.columns.tolist(), key='dependent')

    if independent_variable and dependent_variable:
        if st.button("Perform Trend Analysis"):
            st.session_state['trend_analysis'] = True
            st.session_state['independent_variable'] = independent_variable
            st.session_state['dependent_variable'] = dependent_variable

    if 'trend_analysis' in st.session_state:
        def trend_analysis(df, column, date_column):
            # Ensure the date column is in datetime format
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

            # Drop rows with NaN in the date column
            df = df.dropna(subset=[date_column])

            # Sort the dataframe by date
            df = df.sort_values(by=date_column)

            # Set the date column as index
            df = df.set_index(date_column)

            # Ensure the column is numeric
            df[column] = pd.to_numeric(df[column], errors='coerce')

            # Resample to monthly frequency and calculate the mean
            df_resampled = df[column].resample('ME').mean()

            # Drop rows with NaN values after resampling
            df_resampled = df_resampled.dropna()

            # Perform seasonal decomposition
            result = seasonal_decompose(df_resampled, model='additive')

            return result

        # Perform trend analysis
        result = trend_analysis(processed_data, st.session_state['dependent_variable'], st.session_state['independent_variable'])

        # Plotting the results
        st.header("Trend Analysis Results")
        fig, ax = plt.subplots(4, 1, figsize=(10, 12))

        result.observed.plot(ax=ax[0], color='blue', linewidth=2)
        ax[0].set_title('Observed')
        ax[0].set_ylabel(st.session_state['dependent_variable'])

        result.trend.plot(ax=ax[1], color='green', linewidth=2)
        ax[1].set_title('Trend')
        ax[1].set_ylabel(st.session_state['dependent_variable'])

        result.seasonal.plot(ax=ax[2], color='red', linewidth=2)
        ax[2].set_title('Seasonal')
        ax[2].set_ylabel(st.session_state['dependent_variable'])

        result.resid.plot(ax=ax[3], color='purple', linewidth=2)
        ax[3].set_title('Residual')
        ax[3].set_ylabel(st.session_state['dependent_variable'])

        plt.tight_layout()
        st.pyplot(fig)

# Step 5: Time Series Forecasting
if 'processed_data' in st.session_state:
    st.markdown("### **Step 5: Time Series Forecasting**")

    # Sub-section: Select Independent Variable
    st.subheader("Select Independent Variable")
    forecast_independent_variable = st.selectbox("Independent Variable (only one variable can be selected for now (i.e., Training Enrollment Date))", options=processed_data.columns.tolist(), key='forecast_independent')

    # Sub-section: Select Dependent Variable
    st.subheader("Select Dependent Variable")
    forecast_dependent_variable = st.selectbox("Dependent Variable (only one variable can be selected for now (i.e., Training Cost))", options=processed_data.columns.tolist(), key='forecast_dependent')

    # Sub-section: Select Number of Years for Forecast
    st.subheader("Select Number of Years for Forecast")
    forecast_years = st.number_input("Number of Years", min_value=1, max_value=10, value=1)

    if forecast_independent_variable and forecast_dependent_variable and forecast_years:
        if st.button("Perform Forecasting"):
            st.session_state['forecasting'] = True
            st.session_state['forecast_independent_variable'] = forecast_independent_variable
            st.session_state['forecast_dependent_variable'] = forecast_dependent_variable
            st.session_state['forecast_years'] = forecast_years

    if 'forecasting' in st.session_state:
        def time_series_forecasting(df, column, date_column, years):
            # Ensure the date column is in datetime format
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

            # Drop rows with NaN in the date column
            df = df.dropna(subset=[date_column])

            # Sort the dataframe by date
            df = df.sort_values(by=date_column)

            # Set the date column as index
            df = df.set_index(date_column)

            # Ensure the column is numeric
            df[column] = pd.to_numeric(df[column], errors='coerce')

            # Resample to monthly frequency and calculate the mean
            df_resampled = df[column].resample('ME').mean()

            # Drop rows with NaN values after resampling
            df_resampled = df_resampled.dropna()

            # Split data into training and testing sets
            train_size = int(len(df_resampled) * 0.8)
            train, test = df_resampled[:train_size], df_resampled[train_size:]

            # Fit the model
            model = ExponentialSmoothing(train, seasonal='additive', seasonal_periods=12)
            fit = model.fit()

            # Forecast
            forecast_periods = years * 12  # Monthly forecast
            forecast = fit.forecast(forecast_periods)

            return train, test, forecast

        # Perform time series forecasting
        train, test, forecast = time_series_forecasting(processed_data, st.session_state['forecast_dependent_variable'], st.session_state['forecast_independent_variable'], st.session_state['forecast_years'])

        # Plotting the results
        st.header("Time Series Forecasting Results")
        fig, ax = plt.subplots(figsize=(10, 6))

        train.plot(ax=ax, color='blue', linewidth=5, label='Observed')
        #test.plot(ax=ax, color='green', linewidth=5, label='Test')
        forecast.plot(ax=ax, color='red', linewidth=5, label='Forecast')
        ax.fill_between(forecast.index, forecast - 1.96 * forecast.std(), forecast + 1.96 * forecast.std(), color='lightgrey')
        ax.set_title('Observed vs Forecast')
        ax.set_xlabel(st.session_state['forecast_independent_variable'])
        ax.set_ylabel(st.session_state['forecast_dependent_variable'])
        ax.legend()

        plt.tight_layout()
        st.pyplot(fig)

# Step 6: Plotly Dashboard/Report
if 'processed_data' in st.session_state:
    st.markdown("### **Step 6: Plotly Dashboard/Report**")

    if st.button("Generate Plotly Dashboard"):
        st.session_state['generate_dashboard'] = True

    if 'generate_dashboard' in st.session_state:
        # Plotly Distribution of Training Status
        fig1 = px.histogram(processed_data, x='Training Status', title='Distribution of Training Status')

        # Plotly Histogram of Training Cost
        fig2 = px.histogram(processed_data, x='Training Cost', title='Histogram of Training Cost', nbins=50)

        # Plotly Pairplot for numerical columns
        numerical_columns = processed_data.select_dtypes(include=['float64', 'int64']).columns
        fig3 = px.scatter_matrix(processed_data, dimensions=numerical_columns, title='Pairplot of Numerical Columns')

        # Plotly Trend Analysis
        trend_result = trend_analysis(processed_data, st.session_state['dependent_variable'], st.session_state['independent_variable'])
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=trend_result.observed.index, y=trend_result.observed, mode='lines', name='Observed'))
        fig4.add_trace(go.Scatter(x=trend_result.trend.index, y=trend_result.trend, mode='lines', name='Trend'))
        fig4.add_trace(go.Scatter(x=trend_result.seasonal.index, y=trend_result.seasonal, mode='lines', name='Seasonal'))
        fig4.add_trace(go.Scatter(x=trend_result.resid.index, y=trend_result.resid, mode='lines', name='Residual'))
        fig4.update_layout(title='Trend Analysis', xaxis_title=st.session_state['independent_variable'], yaxis_title=st.session_state['dependent_variable'])

        # Plotly Forecasting Results
        train, test, forecast = time_series_forecasting(processed_data, st.session_state['forecast_dependent_variable'], st.session_state['forecast_independent_variable'], st.session_state['forecast_years'])
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=train.index, y=train, mode='lines', name='Observed'))
        fig5.add_trace(go.Scatter(x=forecast.index, y=forecast, mode='lines', name='Forecast'))
        fig5.update_layout(title='Time Series Forecasting', xaxis_title=st.session_state['forecast_independent_variable'], yaxis_title=st.session_state['forecast_dependent_variable'])

        # Display Plotly Dashboard
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)
        st.plotly_chart(fig3)
        st.plotly_chart(fig4)
        st.plotly_chart(fig5)
else:
    st.write("Please upload and preprocess a CSV file to proceed.")
