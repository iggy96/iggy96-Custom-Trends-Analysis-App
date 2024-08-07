import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from datetime import datetime





"Import Required Libraries"
def load_dataset(file_path):
    return pd.read_csv(file_path)




"Data Preprocessing"
def preprocess_data(df):
    # Handle missing values
    df = df.dropna()

    # Convert data types (example: date columns)
    df['Training Enrollment Date'] = pd.to_datetime(df['Training Enrollment Date'], errors='coerce')
    df['Verification Date'] = pd.to_datetime(df['Verification Date'], errors='coerce')
    df['Session Start Date'] = pd.to_datetime(df['Session Start Date'], errors='coerce')
    df['Session End Date'] = pd.to_datetime(df['Session End Date'], errors='coerce')
    df['Last Update Date'] = pd.to_datetime(df['Last Update Date'], errors='coerce')
    df['Report Generation Date'] = pd.to_datetime(df['Report Generation Date'], errors='coerce')
    df['Data Extraction Date'] = pd.to_datetime(df['Data Extraction Date'], errors='coerce')
    
    # Remove duplicates
    df = df.drop_duplicates()

    return df





"Exploratory Data Analysis (EDA)"
def exploratory_data_analysis(df):
    # Descriptive statistics
    print(df.describe())

    # Data visualization for Training Status Distribution
    plt.figure(figsize=(14, 8))
    sns.countplot(x='Training Status', data=df, palette='viridis')
    plt.title('Distribution of Training Status Among Participants', fontsize=18)
    plt.xlabel('Training Status (e.g., Enrolled, Completed, In Progress)', fontsize=14)
    plt.ylabel('Number of Participants', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()

    # Data visualization for Simulator Type Distribution
    plt.figure(figsize=(14, 8))
    sns.countplot(x='Simulator Type', data=df, palette='viridis')
    plt.title('Distribution of Simulator Types Used in Training', fontsize=18)
    plt.xlabel('Simulator Type (e.g., CAE 7000XR, Simfinity XR)', fontsize=14)
    plt.ylabel('Number of Training Sessions', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()

    # Data visualization for Training Cost Over Time
    plt.figure(figsize=(14, 8))
    sns.lineplot(x='Training Enrollment Date', y='Training Cost', data=df, marker='o', color='blue')
    plt.title('Trend of Training Costs Over Time', fontsize=18)
    plt.xlabel('Date of Training Enrollment', fontsize=14)
    plt.ylabel('Training Cost (in USD)', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()






"Trend Analysis"
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
    
    # Plot the decomposition with enhanced visualization
    plt.figure(figsize=(14, 12))
    
    plt.subplot(411)
    plt.plot(result.observed, color='blue', linewidth=2)
    plt.title('Observed', fontsize=18)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Training Cost', fontsize=14)
    plt.grid(True)
    
    plt.subplot(412)
    plt.plot(result.trend, color='green', linewidth=2)
    plt.title('Trend', fontsize=18)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Training Cost (Trend)', fontsize=14)
    plt.grid(True)
    
    plt.subplot(413)
    plt.plot(result.seasonal, color='red', linewidth=2)
    plt.title('Seasonal', fontsize=18)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Training Cost (Seasonal)', fontsize=14)
    plt.grid(True)
    
    plt.subplot(414)
    plt.plot(result.resid, color='purple', linewidth=2)
    plt.title('Residual', fontsize=18)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Training Cost (Residual)', fontsize=14)
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()








"Time Series Forecasting"
def prepare_time_series(df, column, date_column):
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df = df.dropna(subset=[date_column])
    df = df.set_index(date_column)
    ts = df[column].resample('ME').mean()
    ts = ts.dropna()  # Drop NaN values
    return ts

def split_train_test(ts, split_ratio=0.8):
    split_point = int(len(ts) * split_ratio)
    train, test = ts[:split_point], ts[split_point:]
    return train, test

def apply_forecasting_model(train, periods=6):
    model = ExponentialSmoothing(train, seasonal='additive', seasonal_periods=12)
    fit = model.fit()
    forecast = fit.forecast(periods)
    return fit, forecast

def evaluate_forecast(train, test, forecast):
    combined = pd.concat([train, test])
    forecast_index = pd.date_range(start=combined.index[-1], periods=len(forecast) + 1, freq='ME')[1:]

    plt.figure(figsize=(14, 8))
    plt.plot(train.index, train, label='Observed', color='blue', linewidth=2)
    plt.plot(forecast_index, forecast, label='Forecast', linestyle='--', color='red', linewidth=2)
    plt.fill_between(forecast_index, forecast - 1.96 * np.std(forecast), forecast + 1.96 * np.std(forecast), color='gray', alpha=0.3)
    plt.title('Observed vs Forecast', fontsize=18)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Training Cost', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.show()

