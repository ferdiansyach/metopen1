import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os
import json
import joblib
import warnings
import holidays
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime, timedelta
import re
import zipfile
from scipy.optimize import fsolve
from scipy import stats
import uuid

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Energy Consumption Analysis Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with improved contrast and table styling
st.markdown("""
<style>
    /* --- Base Styles --- */
    body {
        color: #212529; /* Default text color for the app */
    }
    
    /* --- Headers & Titles --- */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    
    /* --- Custom Cards --- */
    .metric-card, .economic-card, .anomaly-card, .clt-card {
        padding: 1.5rem;
        border-radius: 10px;
        color: white; /* Ensure text inside cards is white */
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .economic-card {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
    }
    .anomaly-card {
        background: linear-gradient(135deg, #dc3545 0%, #fd7e14 100%);
    }
    .clt-card {
        background: linear-gradient(135deg, #6f42c1 0%, #e83e8c 100%);
    }
    
    /* Ensure headers inside cards are also white */
    .clt-card h4, .economic-card h4 {
        color: white;
    }

    /* --- Informational Boxes --- */
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.375rem;
        margin: 1rem 0;
    }
    .warning-message {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.375rem;
        margin: 1rem 0;
    }
    .explanation-box {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 15px;
        margin: 10px 0;
        border-radius: 4px;
        color: #212529;
    }
    
    /* --- Table (st.dataframe) Styling --- */
    .stDataFrame {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        overflow: hidden; /* Keeps the border radius */
    }
    .stDataFrame thead th {
        background-color: #f2f2f2;
        color: #333;
        font-weight: bold;
        border-bottom: 2px solid #d0d0d0;
    }
    .stDataFrame tbody tr:nth-of-type(even) {
        background-color: #f9f9f9;
    }
    .stDataFrame tbody tr:hover {
        background-color: #f1f1f1;
    }

    /* --- Responsive Design --- */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.8rem;
        }
        
        [data-testid="stHorizontalBlock"] {
            flex-direction: column;
        }

        [data-testid="stHorizontalBlock"] > div {
            width: 100%;
            margin-bottom: 1rem;
        }

        .economic-card, .metric-card, .anomaly-card, .clt-card {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)


# Main header
st.markdown('<div class="main-header">⚡ Energy Consumption Analysis Dashboard</div>', unsafe_allow_html=True)

# Configuration
TARGET_VARIABLE = 'Konsumsi Energi'
RELEVANT_COLUMNS = [
    'Konsumsi Energi', 'Temperature', 'Showers', 'Cloud Cover', 'Weather Code',
    'Relative Humidity', 'Dew Point', 'Precipitation',
    'Pressure MSL', 'Surface Pressure', 'Evapotranspiration',
    'Vapour Pressure Deficit', 'Wind Speed', 'Wind Direction', 'Wind Gusts',
    'Soil Temperature', 'Sunshine Duration', 'UV Index', 'Direct Radiation'
]

DEVICE_OPERATING_HOURS = {
    'ahu': (8, 16), 'sdp': (0, 23), 'lift': (7, 20), 'chiller': (8, 17)
}
CORE_BUSINESS_HOURS = (9, 17)
DEFAULT_OPERATING_HOURS = (8, 17)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = {}
if 'economic_parameters' not in st.session_state:
    st.session_state.economic_parameters = {}
if 'model_based_savings' not in st.session_state:
    st.session_state.model_based_savings = {}
if 'last_bulk_run_devices' not in st.session_state:
    st.session_state.last_bulk_run_devices = []
if 'anomaly_results' not in st.session_state:
    st.session_state.anomaly_results = {}
if 'training_summary' not in st.session_state:
    st.session_state.training_summary = {}

# Helper Functions
def generate_unique_key():
    """Generate a unique key for Streamlit elements"""
    return str(uuid.uuid4())

### UPDATE ###
# Helper function to convert dataframe to CSV for download
@st.cache_data
def to_csv(df):
    """Converts a dataframe to a UTF-8 encoded CSV file."""
    return df.to_csv(index=False).encode('utf-8')

# Decoupled feature engineering logic to be reusable for aggregated data
def engineer_features(df):
    """Applies feature engineering to a given dataframe."""
    df_engineered = df.copy()
    
    # Feature engineering
    for i in range(1, 4):
        df_engineered[f'Konsumsi_Energi_Lag_{i}'] = df_engineered[TARGET_VARIABLE].shift(i)
    
    df_engineered['rolling_mean_3h'] = df_engineered[TARGET_VARIABLE].shift(1).rolling(window=3).mean()
    df_engineered['rolling_mean_24h'] = df_engineered[TARGET_VARIABLE].shift(1).rolling(window=24).mean()
    
    df_engineered['is_weekend'] = (df_engineered.index.dayofweek >= 5).astype(int)
    years = df_engineered.index.year.unique()
    id_holidays = holidays.Indonesia(years=years)
    df_engineered['isHoliday'] = df_engineered.index.isin(id_holidays).astype(int)
    df_engineered['hour'] = df_engineered.index.hour
    df_engineered['day_of_week'] = df_engineered.index.dayofweek
    df_engineered['week_of_year'] = df_engineered.index.isocalendar().week.astype(int)
    df_engineered['month_of_year'] = df_engineered.index.month
    
    df_engineered.dropna(inplace=True)
    df_engineered = df_engineered[df_engineered[TARGET_VARIABLE] > 0].copy()
    
    return df_engineered

@st.cache_data
def load_and_process_data(csv_file_object, building, device_type, floor, original_filename, min_rows):
    """Load and process a CSV file object from the ZIP archive."""
    try:
        df = pd.read_csv(csv_file_object, index_col='id_time', parse_dates=True)
        df.sort_index(inplace=True)
        
        # Create additional features
        current_cols = ['id_i1', 'id_i2', 'id_i3']
        if all(col in df.columns for col in current_cols):
            for col in current_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=current_cols, inplace=True)
            df['Current'] = df[current_cols].sum(axis=1)

        if 'Power Factor' in df.columns:
            df['Power Factor'] = pd.to_numeric(df['Power Factor'], errors='coerce')

        existing_cols = [col for col in RELEVANT_COLUMNS if col in df.columns]
        df = df[existing_cols].copy()
        
        for col in existing_cols:
            if col != TARGET_VARIABLE:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(subset=[TARGET_VARIABLE], inplace=True)
        
        # Call the new decoupled feature engineering function
        df_processed = engineer_features(df)
        
        if len(df_processed) < min_rows:
            return None, None, None, None, f"Insufficient data for **{original_filename}**: {len(df_processed)} rows found (minimum {min_rows} required)."
        
        return df_processed, building, device_type, floor, None
    except Exception as e:
        return None, None, None, None, f"Error processing {original_filename}: {e}"

def train_models(X_train, y_train, X_val, y_val, X_test, y_test, status_text=None, progress_callback=None):
    """Train multiple models and return results, with progress updates."""
    results = {}
    
    # Random Forest
    if status_text: status_text.text("Training Model 1/3: Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    results['RandomForest'] = {
        'model': rf_model, 
        'predictions': y_pred_rf,
        'metrics': {
            'mae': mean_absolute_error(y_test, y_pred_rf),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
            'r2': r2_score(y_test, y_pred_rf)
        }
    }
    if progress_callback: progress_callback(1/3)

    # Gradient Boosting
    if status_text: status_text.text("Training Model 2/3: Gradient Boosting...")
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)
    results['GradientBoosting'] = {
        'model': gb_model, 
        'predictions': y_pred_gb,
        'metrics': {
            'mae': mean_absolute_error(y_test, y_pred_gb),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_gb)),
            'r2': r2_score(y_test, y_pred_gb)
        }
    }
    if progress_callback: progress_callback(2/3)
    
    # LSTM
    if status_text: status_text.text("Training Model 3/3: LSTM...")
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    y_train_s = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    X_val_s = scaler_X.transform(X_val)
    y_val_s = scaler_y.transform(y_val.values.reshape(-1, 1))
    X_test_s = scaler_X.transform(X_test)
    
    X_train_r = X_train_s.reshape((X_train_s.shape[0], 1, X_train_s.shape[1]))
    X_val_r = X_val_s.reshape((X_val_s.shape[0], 1, X_val_s.shape[1]))
    X_test_r = X_test_s.reshape((X_test_s.shape[0], 1, X_test_s.shape[1]))
    
    lstm = Sequential([
        LSTM(50, activation='relu', input_shape=(1, X_train_r.shape[2])),
        Dense(1)
    ])
    lstm.compile(optimizer='adam', loss='mse')
    lstm.fit(X_train_r, y_train_s, epochs=50, batch_size=32, 
             validation_data=(X_val_r, y_val_s), verbose=0, shuffle=False)
    
    y_pred_lstm = scaler_y.inverse_transform(lstm.predict(X_test_r, verbose=0))
    results['LSTM'] = {
        'model': lstm, 
        'predictions': y_pred_lstm.flatten(),
        'metrics': {
            'mae': mean_absolute_error(y_test, y_pred_lstm),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_lstm)),
            'r2': r2_score(y_test, y_pred_lstm)
        },
        'scaler_X': scaler_X, 
        'scaler_y': scaler_y
    }
    if progress_callback: progress_callback(3/3)
    
    return results

def analyze_consumption_patterns(df, device_identifier):
    """Analyze energy consumption patterns"""
    patterns = {}
    
    # Hourly patterns
    hourly_avg = df.groupby(df.index.hour)[TARGET_VARIABLE].mean()
    patterns['hourly'] = hourly_avg
    
    # Daily patterns
    daily_avg = df.groupby(df.index.dayofweek)[TARGET_VARIABLE].mean()
    patterns['daily'] = daily_avg
    
    # Monthly patterns
    monthly_avg = df.groupby(df.index.month)[TARGET_VARIABLE].mean()
    patterns['monthly'] = monthly_avg
    
    # Peak consumption
    patterns['peak_hour'] = hourly_avg.idxmax()
    patterns['peak_consumption'] = hourly_avg.max()
    patterns['off_peak_hour'] = hourly_avg.idxmin()
    patterns['off_peak_consumption'] = hourly_avg.min()
    
    # Business hours vs non-business hours analysis
    business_hours_data = df[(df.index.hour >= 9) & (df.index.hour <= 17) & (df.index.dayofweek < 5)]
    non_business_data = df[~((df.index.hour >= 9) & (df.index.hour <= 17) & (df.index.dayofweek < 5))]
    
    patterns['business_hours_avg'] = business_hours_data[TARGET_VARIABLE].mean() if not business_hours_data.empty else 0
    patterns['non_business_avg'] = non_business_data[TARGET_VARIABLE].mean() if not non_business_data.empty else 0
    
    # Weekend vs weekday analysis
    weekday_data = df[df.index.dayofweek < 5]
    weekend_data = df[df.index.dayofweek >= 5]
    
    patterns['weekday_avg'] = weekday_data[TARGET_VARIABLE].mean() if not weekday_data.empty else 0
    patterns['weekend_avg'] = weekend_data[TARGET_VARIABLE].mean() if not weekend_data.empty else 0
    
    return patterns

def detect_anomalies_detailed(df, model_data, features_for_model, device_id):
    """Enhanced anomaly detection with detailed analysis"""
    best_model_name = model_data['best_model_name']
    best_model_data = model_data['model_results'][best_model_name]
    prediction_error_std = model_data['prediction_error_std']
    
    # Generate predictions for all data
    full_predictions = None
    if best_model_name == 'LSTM':
        scaler_X, scaler_y = best_model_data['scaler_X'], best_model_data['scaler_y']
        X_full_s = scaler_X.transform(df[features_for_model])
        X_full_r = X_full_s.reshape((X_full_s.shape[0], 1, X_full_s.shape[1]))
        pred_full_s = best_model_data['model'].predict(X_full_r, verbose=0)
        full_predictions = scaler_y.inverse_transform(pred_full_s).flatten()
    else:
        full_predictions = best_model_data['model'].predict(df[features_for_model])

    # Create comprehensive anomaly analysis
    anomaly_df = pd.DataFrame({
        'timestamp': df.index,
        'actual_consumption': df[TARGET_VARIABLE].values,
        'predicted_consumption': full_predictions,
        'hour': df.index.hour,
        'day_of_week': df.index.dayofweek,
        'month': df.index.month
    })
    
    # Calculate statistical thresholds
    threshold_2std = anomaly_df['predicted_consumption'] + (2 * prediction_error_std)
    threshold_3std = anomaly_df['predicted_consumption'] + (3 * prediction_error_std)
    
    anomaly_df['threshold_2std'] = threshold_2std
    anomaly_df['threshold_3std'] = threshold_3std
    anomaly_df['anomaly_2std'] = np.maximum(0, anomaly_df['actual_consumption'] - threshold_2std)
    anomaly_df['anomaly_3std'] = np.maximum(0, anomaly_df['actual_consumption'] - threshold_3std)
    anomaly_df['is_anomaly_2std'] = anomaly_df['anomaly_2std'] > 0
    anomaly_df['is_anomaly_3std'] = anomaly_df['anomaly_3std'] > 0
    
    # Calculate prediction error
    anomaly_df['prediction_error'] = anomaly_df['actual_consumption'] - anomaly_df['predicted_consumption']
    
    # Summary statistics
    total_anomalies_2std = anomaly_df['is_anomaly_2std'].sum()
    total_anomalies_3std = anomaly_df['is_anomaly_3std'].sum()
    total_savings_2std = anomaly_df['anomaly_2std'].sum()
    total_savings_3std = anomaly_df['anomaly_3std'].sum()
    total_consumption = anomaly_df['actual_consumption'].sum()
    
    anomaly_summary = {
        'device_id': device_id,
        'best_model': best_model_name,
        'total_data_points': len(anomaly_df),
        'total_anomalies_2std': total_anomalies_2std,
        'total_anomalies_3std': total_anomalies_3std,
        'anomaly_rate_2std': (total_anomalies_2std / len(anomaly_df)) * 100,
        'anomaly_rate_3std': (total_anomalies_3std / len(anomaly_df)) * 100,
        'total_savings_2std_wh': total_savings_2std,
        'total_savings_3std_wh': total_savings_3std,
        'savings_percentage_2std': (total_savings_2std / total_consumption) * 100,
        'savings_percentage_3std': (total_savings_3std / total_consumption) * 100,
        'prediction_error_std': prediction_error_std
    }
    
    return anomaly_df, anomaly_summary

def create_aggregated_analysis(all_data):
    """Create comprehensive aggregated analysis"""
    building_data = {}
    device_type_data = {}
    floor_data = {}
    
    for device_id, data in all_data.items():
        building = data['building']
        device_type = data['device_type']
        floor = data['floor']
        df = data['df']
        
        if building not in building_data:
            building_data[building] = []
        building_data[building].append(df)
        
        if device_type not in device_type_data:
            device_type_data[device_type] = []
        device_type_data[device_type].append(df)
        
        floor_key = f"{building}-{floor}"
        if floor_key not in floor_data:
            floor_data[floor_key] = []
        floor_data[floor_key].append(df)
    
    # Create aggregated DataFrames
    building_agg = {}
    for building, dfs in building_data.items():
        combined_df = pd.concat(dfs, ignore_index=False)
        building_agg[building] = {
            'df': combined_df,
            'total_consumption': combined_df[TARGET_VARIABLE].sum(),
            'avg_consumption': combined_df[TARGET_VARIABLE].mean(),
            'device_count': len(dfs),
            'patterns': analyze_consumption_patterns(combined_df, building)
        }
    
    device_agg = {}
    for device_type, dfs in device_type_data.items():
        combined_df = pd.concat(dfs, ignore_index=False)
        device_agg[device_type] = {
            'df': combined_df,
            'total_consumption': combined_df[TARGET_VARIABLE].sum(),
            'avg_consumption': combined_df[TARGET_VARIABLE].mean(),
            'device_count': len(dfs),
            'patterns': analyze_consumption_patterns(combined_df, device_type)
        }
    
    floor_agg = {}
    for floor_key, dfs in floor_data.items():
        combined_df = pd.concat(dfs, ignore_index=False)
        floor_agg[floor_key] = {
            'df': combined_df,
            'total_consumption': combined_df[TARGET_VARIABLE].sum(),
            'avg_consumption': combined_df[TARGET_VARIABLE].mean(),
            'device_count': len(dfs),
            'patterns': analyze_consumption_patterns(combined_df, floor_key)
        }
    
    return building_agg, device_agg, floor_agg

def analyze_and_visualize_consumption_by_device_type(device_agg, chart_title="Device Type Analysis"):
    """Analyze and visualize consumption using average per unit methodology"""
    consumption_data = {}
    unit_counts = {}
    
    for device_type, data in device_agg.items():
        consumption_data[device_type.upper()] = data['total_consumption']
        unit_counts[device_type.upper()] = data['device_count']
    
    if not consumption_data:
        return None, None
    
    # Create DataFrame
    df = pd.DataFrame(list(consumption_data.items()), columns=['Tipe Perangkat', 'Total Konsumsi (Wh)'])
    df['Jumlah Unit'] = df['Tipe Perangkat'].map(unit_counts)
    df['Total Konsumsi (kWh)'] = df['Total Konsumsi (Wh)'] / 1000
    
    # Calculate average per unit
    df['Rata-rata per Unit (kWh)'] = df['Total Konsumsi (kWh)'] / df['Jumlah Unit']
    
    # Calculate percentage based on average
    df['Persentase (%)'] = (df['Rata-rata per Unit (kWh)'] / df['Rata-rata per Unit (kWh)'].sum()) * 100
    df = df.sort_values(by='Rata-rata per Unit (kWh)', ascending=False)
    
    # Create pie chart
    fig = px.pie(df, 
                 values='Rata-rata per Unit (kWh)', 
                 names='Tipe Perangkat',
                 title=f'{chart_title} - Distribusi Konsumsi Rata-rata per Unit',
                 hole=0.4)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig, df

def calculate_clt_statistics(data, sample_size=30, num_samples=1000):
    """Calculate Central Limit Theorem statistics"""
    if len(data) < sample_size:
        return None, None, None, None
    
    # Population statistics
    population_mean = np.mean(data)
    population_std = np.std(data)
    
    # Generate sample means
    sample_means = []
    for _ in range(num_samples):
        sample = np.random.choice(data, size=sample_size, replace=True)
        sample_means.append(np.mean(sample))
    
    sample_means = np.array(sample_means)
    
    # Theoretical CLT parameters
    theoretical_mean = population_mean
    theoretical_std = population_std / np.sqrt(sample_size)
    
    return sample_means, population_mean, theoretical_mean, theoretical_std

def visualize_clt(data, title_prefix="", unique_key=""):
    """Create CLT visualization"""
    sample_means, pop_mean, theo_mean, theo_std = calculate_clt_statistics(data)
    
    if sample_means is None:
        st.warning("Insufficient data for CLT analysis (minimum 30 points required)")
        return
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Population Distribution', 'Sample Means Distribution (CLT)'),
        horizontal_spacing=0.1
    )
    
    # Population distribution
    fig.add_trace(
        go.Histogram(x=data, nbinsx=50, name='Population', opacity=0.7, 
                     marker_color='skyblue'), row=1, col=1
    )
    fig.add_vline(x=pop_mean, line_width=2, line_dash="dash", 
                  line_color="red", row=1, col=1)
    
    # Sample means distribution
    fig.add_trace(
        go.Histogram(x=sample_means, nbinsx=50, name='Sample Means', opacity=0.7,
                     marker_color='lightgreen'), row=1, col=2
    )
    fig.add_vline(x=theo_mean, line_width=2, line_dash="dash", 
                  line_color="red", row=1, col=2)
    
    # Add theoretical normal curve
    x_norm = np.linspace(sample_means.min(), sample_means.max(), 100)
    y_norm = stats.norm.pdf(x_norm, theo_mean, theo_std)
    # Scale to match histogram
    y_norm = y_norm * len(sample_means) * (sample_means.max() - sample_means.min()) / 50
    
    fig.add_trace(
        go.Scatter(x=x_norm, y=y_norm, mode='lines', name='Theoretical Normal',
                   line=dict(color='darkred', width=3)), row=1, col=2
    )
    
    fig.update_layout(
        title=f'{title_prefix} Central Limit Theorem Visualization',
        height=500,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Energy Consumption (Wh)", row=1, col=1)
    fig.update_xaxes(title_text="Sample Mean (Wh)", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    
    st.plotly_chart(fig, use_container_width=True, key=f"clt_{unique_key}")
    
    # Statistics card
    st.markdown(f"""
    <div class="clt-card">
        <h4>Central Limit Theorem Statistics</h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
            <div>
                <strong>Population Mean:</strong> {pop_mean:.2f} Wh<br>
                <strong>Population Std:</strong> {np.std(data):.2f} Wh<br>
                <strong>Sample Size:</strong> 30
            </div>
            <div>
                <strong>Theoretical Mean:</strong> {theo_mean:.2f} Wh<br>
                <strong>Theoretical Std:</strong> {theo_std:.2f} Wh<br>
                <strong>Actual Sample Mean Std:</strong> {np.std(sample_means):.2f} Wh
            </div>
        </div>
        <p><strong>CLT Verification:</strong> The sample means should approximate a normal distribution with mean ≈ {theo_mean:.2f} and std ≈ {theo_std:.2f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add explanation box
    st.markdown("""
    <div class="explanation-box">
        <h4>Penjelasan Grafik</h4>
        <p>Grafik ini menunjukkan Teorema Batas Sentral (CLT) dalam aksi:</p>
        <ul>
            <li><strong>Population Distribution (kiri):</strong> Menampilkan distribusi asli dari seluruh data konsumsi energi Anda. Seringkali, distribusi ini tidak berbentuk lonceng (tidak normal).</li>
            <li><strong>Sample Means Distribution (kanan):</strong> Menampilkan distribusi dari rata-rata banyak sampel acak yang diambil dari populasi. Sesuai dengan CLT, distribusi ini akan mendekati kurva normal (berbentuk lonceng), yang memvalidasi penggunaan metode statistik pada data Anda.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def calculate_irr(cash_flows):
    """Helper function to calculate IRR using scipy fsolve."""
    def npv(rate):
        return sum([cf / (1 + rate) ** i for i, cf in enumerate(cash_flows)])
    try:
        irr_result = fsolve(npv, 0.1)[0]
        return irr_result * 100
    except Exception:
        return None

def calculate_economic_metrics(energy_savings_kwh_annual, capex, annual_maintenance, electricity_rate, project_years=10, discount_rate=0.08):
    """Calculate comprehensive economic metrics for energy efficiency projects"""
    
    # Annual savings
    annual_energy_savings = energy_savings_kwh_annual * electricity_rate
    
    # OPEX (annual operational expenditure - mainly maintenance)
    annual_opex = annual_maintenance
    
    # Net annual savings (Revenue - OPEX)
    net_annual_savings = annual_energy_savings - annual_opex
    
    # Cumulative cash flow calculation
    cumulative_cash_flow = [-capex]
    for year in range(1, project_years + 1):
        annual_cash_flow = net_annual_savings
        cumulative_cash_flow.append(cumulative_cash_flow[-1] + annual_cash_flow)
    
    # Payback period calculation
    payback_period = None
    if net_annual_savings > 0:
        for i, cash_flow in enumerate(cumulative_cash_flow):
            if cash_flow >= 0:
                if i > 0:
                    prev_cash_flow = cumulative_cash_flow[i-1]
                    payback_period = (i - 1) + abs(prev_cash_flow) / net_annual_savings
                else:
                    payback_period = 0
                break
        
        if payback_period is None:
            payback_period = float('inf')
    else:
        payback_period = float('inf')
    
    # Total project revenue and profit
    total_project_revenue = annual_energy_savings * project_years
    total_project_profit = net_annual_savings * project_years
    
    # ROI calculation
    roi = (total_project_profit / capex) * 100 if capex > 0 else float('inf')
    
    # NPV calculation
    npv = -capex
    for year in range(1, project_years + 1):
        npv += net_annual_savings / ((1 + discount_rate) ** year)
    
    # IRR calculation
    cash_flows = [-capex] + [net_annual_savings] * project_years
    irr = calculate_irr(cash_flows)
    
    return {
        'annual_energy_savings_kwh': energy_savings_kwh_annual,
        'annual_energy_savings_currency': annual_energy_savings,
        'annual_opex': annual_opex,
        'net_annual_savings': net_annual_savings,
        'payback_period': payback_period if payback_period != float('inf') else None,
        'roi': roi if roi != float('inf') else None,
        'npv': npv,
        'irr': irr,
        'cumulative_cash_flow': cumulative_cash_flow,
        'total_project_revenue': total_project_revenue,
        'total_project_profit': total_project_profit
    }

# Sidebar
st.sidebar.title("Dashboard Controls")

# File upload section
st.sidebar.header("Data Upload")
uploaded_zip_file = st.sidebar.file_uploader(
    "Upload a ZIP file with your data",
    type=['zip'],
    accept_multiple_files=False,
    help="Upload a single ZIP file containing your data. The folder structure inside the ZIP should be: building/floor/device/data.csv or building/device/data.csv"
)

# Analysis parameters
st.sidebar.header("Analysis Parameters")
minimum_rows = st.sidebar.slider("Minimum Data Points", min_value=100, max_value=2000, value=500, step=50, help="Minimum number of data points required for analysis")
kwh_price = st.sidebar.number_input("Harga per kWh (Rp)", min_value=0, value=1445, step=1, help="Masukkan harga listrik per kWh dalam Rupiah untuk menghitung potensi penghematan biaya.")
st.sidebar.header("Model Training Parameters")
train_split_ratio = st.sidebar.slider("Training Data Split (%)", min_value=50, max_value=80, value=70, help="Persentase data yang digunakan untuk melatih model.")

def remove_multicollinear_features(df, target_col, threshold=0.7):
    """Remove highly correlated features"""
    potential_features = [col for col in df.columns if col not in [target_col] and 'Lag' not in col]
    
    if len(potential_features) <= 1:
        return potential_features, []
    
    feature_corr_matrix = df[potential_features].corr().abs()
    upper_tri = feature_corr_matrix.where(np.triu(np.ones(feature_corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] >= threshold)]
    
    independent_features = [f for f in potential_features if f not in to_drop]
    return independent_features, to_drop

def run_training_process(device_id, device_data, status_text=None, progress_callback=None):
    """A self-contained function to train models and store results in session_state."""
    df = device_data['df']
    
    # Time-series split
    train_size = int(len(df) * (train_split_ratio / 100))
    val_size = int(len(df) * 0.15)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]

    independent_features, dropped_features = remove_multicollinear_features(df, TARGET_VARIABLE)
    lag_cols = [col for col in df.columns if 'Lag' in col or 'rolling' in col]
    features_for_model = lag_cols + independent_features
    features_for_model = [f for f in features_for_model if f in df.columns]

    X_train, y_train = train_df[features_for_model], train_df[TARGET_VARIABLE]
    X_val, y_val = val_df[features_for_model], val_df[TARGET_VARIABLE]
    X_test, y_test = test_df[features_for_model], test_df[TARGET_VARIABLE]
    
    # Model Training
    model_results = train_models(X_train, y_train, X_val, y_val, X_test, y_test, status_text, progress_callback)
    best_model_name = min(model_results, key=lambda k: model_results[k]['metrics']['mae'])
    
    # Anomaly detection
    best_model_data = model_results[best_model_name]
    val_predictions = None
    if best_model_name == 'LSTM':
        scaler_X, scaler_y = best_model_data['scaler_X'], best_model_data['scaler_y']
        X_val_s = scaler_X.transform(X_val)
        X_val_r = X_val_s.reshape((X_val_s.shape[0], 1, X_val_s.shape[1]))
        pred_val_s = best_model_data['model'].predict(X_val_r, verbose=0)
        val_predictions = scaler_y.inverse_transform(pred_val_s).flatten()
    else:
        val_predictions = best_model_data['model'].predict(X_val)
    
    prediction_error_std = np.std(y_val - val_predictions)
    
    # Store comprehensive results
    st.session_state.analysis_results[device_id] = {
        'model_results': model_results, 
        'best_model_name': best_model_name,
        'features_for_model': features_for_model, 
        'dropped_features': dropped_features,
        'test_data': {'X_test': X_test, 'y_test': y_test},
        'full_data': {'X': df[features_for_model], 'y': df[TARGET_VARIABLE]},
        'prediction_error_std': prediction_error_std,
        'all_features': df.columns.tolist()
    }
    
    # Detailed anomaly detection
    anomaly_df, anomaly_summary = detect_anomalies_detailed(
        df, st.session_state.analysis_results[device_id], features_for_model, device_id
    )
    
    st.session_state.anomaly_results[device_id] = {
        'anomaly_df': anomaly_df,
        'anomaly_summary': anomaly_summary
    }
    
    st.session_state.model_based_savings[device_id] = {
        'savings_wh': anomaly_summary['total_savings_2std_wh'],
        'total_consumption_wh': anomaly_df['actual_consumption'].sum(),
        'savings_percentage': anomaly_summary['savings_percentage_2std']
    }

def display_detailed_results(device_id):
    """Renders all detailed visualizations for a trained model from session_state."""
    if device_id not in st.session_state.analysis_results:
        st.warning(f"No training results found for {device_id}. Please train the model first.")
        return

    results = st.session_state.analysis_results[device_id]
    
     # Ambil DataFrame dari session_state yang sudah tersimpan dengan aman
    if device_id in st.session_state.uploaded_files:
        device_data = st.session_state.uploaded_files[device_id]
        df = device_data['df']
    # Fallback jika data berasal dari model agregat yang tidak ada di 'uploaded_files'
    elif 'df' in results.get('full_data', {}):
        df = results['full_data']['df']
    else: 
        st.error(f"Could not find the dataframe for {device_id} to display results.")
        return
    
    # Handle both individual and aggregate data sources
    if device_id in valid_files:
        device_data = valid_files[device_id]
        df = device_data['df']
    elif 'df' in st.session_state.analysis_results[device_id].get('full_data', {}):
        df = st.session_state.analysis_results[device_id]['full_data']['df']
    else: # Fallback if df not found
        st.error(f"Could not find dataframe for {device_id}")
        return
    
    # Tahap 1: Analisis Fitur & Korelasi - SHOW ALL FEATURES
    st.markdown("---")
    st.subheader("Tahap 1: Analisis Fitur & Korelasi")
    
    # Reorder columns to place target variable first for visualization
    all_features_for_corr = df.columns.tolist()
    if TARGET_VARIABLE in all_features_for_corr:
        all_features_for_corr.insert(0, all_features_for_corr.pop(all_features_for_corr.index(TARGET_VARIABLE)))
    
    corr_matrix_all = df[all_features_for_corr].corr()

    # Dynamically adjust height to ensure all features are visible
    num_features = len(corr_matrix_all.columns)
    height = max(600, num_features * 25)  # 25px per feature, min 600px
    
    fig_corr_all = go.Figure(go.Heatmap(
        z=corr_matrix_all.values, 
        x=corr_matrix_all.columns, 
        y=corr_matrix_all.columns, 
        colorscale='RdBu', 
        zmin=-1, zmax=1, 
        text=np.around(corr_matrix_all.values, 2), 
        texttemplate="%{text}",
        textfont={"size": 9}
    ))
    fig_corr_all.update_layout(
        title='Matriks Korelasi Semua Fitur', 
        height=height,
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig_corr_all, use_container_width=True, key=f"corr_all_{device_id}")
    
    # Show used features and their correlation with target
    used_features = results['features_for_model']
    target_correlations = []
    for feature in used_features:
        if feature in df.columns and feature != TARGET_VARIABLE:
            corr_val = df[feature].corr(df[TARGET_VARIABLE])
            target_correlations.append({'Feature': feature, 'Correlation': corr_val})
    
    if target_correlations:
        corr_df = pd.DataFrame(target_correlations).sort_values('Correlation', key=abs, ascending=False)
        st.subheader("Fitur yang Digunakan")
        st.dataframe(corr_df, use_container_width=True)
        ### UPDATE ###
        csv = to_csv(corr_df)
        st.download_button("Download Used Features as CSV", csv, f"{device_id}_used_features.csv", "text/csv", key=f'download-corr-{device_id}')
    
    # Show dropped features with reasons
    dropped_features = results['dropped_features']
    if dropped_features:
        st.subheader("Fitur yang Dihapus (Korelasi > 0.7)")
        dropped_reasons = []
        for feature in dropped_features:
            dropped_reasons.append({
                'Feature': feature, 
                'Reason': 'Highly correlated with other features (r>0.7)'
            })
        dropped_df = pd.DataFrame(dropped_reasons)
        st.dataframe(dropped_df, use_container_width=True)
        ### UPDATE ###
        csv = to_csv(dropped_df)
        st.download_button("Download Dropped Features as CSV", csv, f"{device_id}_dropped_features.csv", "text/csv", key=f'download-dropped-{device_id}')
    else:
        st.info("Tidak ada fitur yang dihapus karena multikolinearitas.")
    
    # Tahap 2: Evaluasi Model
    st.markdown("---")
    st.subheader("Tahap 2: Evaluasi Model")
    model_results = results['model_results']
    comparison_data = [{'Model': name, **res['metrics']} for name, res in model_results.items()]
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df.rename(columns={'mae': 'MAE (Wh)', 'rmse': 'RMSE (Wh)', 'r2': 'R² Score'}), use_container_width=True)
    ### UPDATE ###
    csv = to_csv(comparison_df)
    st.download_button("Download Model Metrics as CSV", csv, f"{device_id}_model_metrics.csv", "text/csv", key=f'download-metrics-{device_id}')
    
    best_model_name = results['best_model_name']
    st.success(f"🏆 Model Terbaik: **{best_model_name}**")

    # Add explanation for model selection
    best_model_mae = results['model_results'][best_model_name]['metrics']['mae']
    st.markdown(f"""
    <div class="explanation-box">
        <h4>Mengapa {best_model_name} Dipilih?</h4>
        <p>Model <strong>{best_model_name}</strong> dipilih sebagai model terbaik karena memiliki nilai <strong>Mean Absolute Error (MAE)</strong> terendah, yaitu <strong>{best_model_mae:.2f} Wh</strong>.</p>
        <ul>
            <li><strong>MAE</strong> mengukur rata-rata selisih absolut antara nilai prediksi dan nilai aktual. Dalam konteks ini, ini berarti prediksi model rata-rata menyimpang sebesar {best_model_mae:.2f} Wh dari konsumsi energi sebenarnya.</li>
            <li>Semakin rendah nilai MAE, semakin akurat prediksi model tersebut.</li>
        </ul>
        <p>Meskipun metrik lain seperti RMSE dan R² juga penting, MAE dipilih sebagai kriteria utama karena mudah diinterpretasikan dan tidak terlalu sensitif terhadap nilai-nilai ekstrem (pencilan) dibandingkan RMSE.</p>
    </div>
    """, unsafe_allow_html=True)

    # Tahap 3: Visualisasi Hasil Model Terbaik
    st.markdown("---")
    st.subheader(f"Tahap 3: Detail Kinerja Model - {best_model_name}")
    best_model_data = model_results[best_model_name]
    best_model = best_model_data['model']
    best_predictions = best_model_data['predictions']
    y_test = results['test_data']['y_test']

    if hasattr(best_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Fitur': used_features, 
            'Pentingnya': best_model.feature_importances_
        }).sort_values('Pentingnya', ascending=False).head(15)
        fig_imp = px.bar(importance_df, x='Pentingnya', y='Fitur', orientation='h', 
                         title=f'15 Fitur Terpenting - {best_model_name}')
        st.plotly_chart(fig_imp, use_container_width=True, key=f"imp_{device_id}")

    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(x=y_test.values, y=best_predictions, mode='markers', 
                                     name='Prediksi', opacity=0.6))
    min_val, max_val = min(y_test.min(), best_predictions.min()), max(y_test.max(), best_predictions.max())
    fig_scatter.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', 
                                     name='Prediksi Sempurna', line=dict(color='red', dash='dash')))
    fig_scatter.update_layout(title=f'Aktual vs. Prediksi - {best_model_name}', 
                              xaxis_title='Konsumsi Aktual (Wh)', yaxis_title='Konsumsi Prediksi (Wh)', height=500)
    st.plotly_chart(fig_scatter, use_container_width=True, key=f"scatter_{device_id}")

    plot_df = pd.DataFrame({'Aktual': y_test, 'Prediksi': best_predictions}).sort_index()
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Aktual'], mode='lines', 
                               name='Data Aktual', line=dict(color='blue', width=2)))
    fig_ts.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Prediksi'], mode='lines', 
                               name='Hasil Prediksi', line=dict(color='red', dash='dash', width=2)))
    fig_ts.update_layout(title=f'Prediksi vs. Aktual pada Data Uji - {best_model_name}', 
                         xaxis_title='Waktu', yaxis_title='Konsumsi Energi (Wh)', height=500, 
                         legend=dict(x=0.01, y=0.99))
    st.plotly_chart(fig_ts, use_container_width=True, key=f"ts_{device_id}")
    
    # Tahap 4: Deteksi Anomali dan Potensi Penghematan
    st.markdown("---")
    st.subheader("Tahap 4: Analisis Deteksi Anomali")
    
    if device_id in st.session_state.anomaly_results:
        anomaly_data = st.session_state.anomaly_results[device_id]
        anomaly_df = anomaly_data['anomaly_df']
        anomaly_summary = anomaly_data['anomaly_summary']
        
        # Anomaly summary cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Anomali (2σ)", f"{anomaly_summary['total_anomalies_2std']:,}")
        with col2:
            st.metric("Tingkat Anomali (2σ)", f"{anomaly_summary['anomaly_rate_2std']:.2f}%")
        with col3:
            st.metric("Potensi Penghematan", f"{anomaly_summary['total_savings_2std_wh']/1000:.2f} kWh")
        with col4:
            st.metric("Persentase Penghematan", f"{anomaly_summary['savings_percentage_2std']:.2f}%")
        
        # Anomaly visualization
        fig_anomaly = go.Figure()
        
        # Plot actual consumption
        fig_anomaly.add_trace(go.Scatter(
            x=anomaly_df['timestamp'], 
            y=anomaly_df['actual_consumption'],
            mode='lines', name='Konsumsi Aktual', line=dict(color='blue', width=1)
        ))
        
        # Plot predicted consumption
        fig_anomaly.add_trace(go.Scatter(
            x=anomaly_df['timestamp'], 
            y=anomaly_df['predicted_consumption'],
            mode='lines', name='Konsumsi Prediksi', line=dict(color='green', width=1)
        ))
        
        # Plot 2-sigma threshold
        fig_anomaly.add_trace(go.Scatter(
            x=anomaly_df['timestamp'], 
            y=anomaly_df['threshold_2std'],
            mode='lines', name='Ambang Batas (2σ)', line=dict(color='red', dash='dash', width=1)
        ))
        
        # Highlight anomalies
        anomaly_points = anomaly_df[anomaly_df['is_anomaly_2std']]
        if not anomaly_points.empty:
            fig_anomaly.add_trace(go.Scatter(
                x=anomaly_points['timestamp'], 
                y=anomaly_points['actual_consumption'],
                mode='markers', name='Anomali Terdeteksi', 
                marker=dict(color='red', size=6, symbol='circle')
            ))
        
        fig_anomaly.update_layout(
            title='Deteksi Anomali Konsumsi Energi',
            xaxis_title='Waktu', yaxis_title='Konsumsi Energi (Wh)',
            height=500, hovermode='x unified'
        )
        st.plotly_chart(fig_anomaly, use_container_width=True, key=f"anomaly_{device_id}")
        
        # Anomaly patterns analysis
        st.subheader("Pola Anomali")
        
        col1, col2 = st.columns(2)
        with col1:
            # Hourly anomaly pattern
            hourly_anomalies = anomaly_df.groupby('hour').agg({
                'is_anomaly_2std': 'sum',
                'anomaly_2std': 'sum'
            }).reset_index()
            
            fig_hourly = px.bar(hourly_anomalies, x='hour', y='is_anomaly_2std',
                                  title='Distribusi Anomali per Jam')
            fig_hourly.update_xaxes(title='Jam')
            fig_hourly.update_yaxes(title='Jumlah Anomali')
            st.plotly_chart(fig_hourly, use_container_width=True, key=f"hourly_anomaly_{device_id}")
            
        with col2:
            # Daily anomaly pattern
            daily_anomalies = anomaly_df.groupby('day_of_week').agg({
                'is_anomaly_2std': 'sum',
                'anomaly_2std': 'sum'
            }).reset_index()
            daily_anomalies['day_name'] = daily_anomalies['day_of_week'].map({
                0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'
            })
            
            fig_daily = px.bar(daily_anomalies, x='day_name', y='is_anomaly_2std',
                                  title='Distribusi Anomali per Hari')
            fig_daily.update_xaxes(title='Hari')
            fig_daily.update_yaxes(title='Jumlah Anomali')
            st.plotly_chart(fig_daily, use_container_width=True, key=f"daily_anomaly_{device_id}")
        
        # Detailed explanation
        st.markdown(f"""
        <div class="explanation-box">
            <h4>Penjelasan Deteksi Anomali</h4>
            <p><strong>Metodologi:</strong> Sistem menggunakan model {best_model_name} untuk memprediksi konsumsi normal, 
            kemudian menerapkan ambang batas statistik 2 standar deviasi untuk mendeteksi anomali.</p>
            <p><strong>Ambang Batas:</strong> Konsumsi dianggap anomali jika melebihi prediksi + (2 × σ = {anomaly_summary['prediction_error_std']:.2f})</p>
            <p><strong>Interpretasi:</strong> Anomali menunjukkan konsumsi berlebihan yang berpotensi dapat dikurangi melalui 
            optimisasi operasional, pemeliharaan, atau penyesuaian pengaturan perangkat.</p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.warning("Data hasil deteksi anomali tidak ditemukan.")


# Main content area
# Main content area - FIXED VERSION
if uploaded_zip_file:
    valid_files = {}
    filename_info = []
    
    try:
        # Read the entire ZIP file content into memory first
        zip_content = uploaded_zip_file.getvalue()
        
        with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_ref:
            file_list = zip_ref.infolist()
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, file_info in enumerate(file_list):
                file_path = file_info.filename
                
                progress_bar.progress((i + 1) / len(file_list))
                status_text.text(f"Processing: {file_path}")

                if file_info.is_dir() or not file_path.lower().endswith('.csv') or '__MACOSX' in file_path:
                    continue

                # --- LOGIKA PARSING PATH YANG LENGKAP ---
                path_parts = [part for part in re.split(r'[/\\]', file_path) if part]
                
                first_building_index = -1
                valid_buildings = ['opmc', 'witel']
                for idx, part in enumerate(path_parts):
                    if part.lower() in valid_buildings:
                        first_building_index = idx
                        break

                if first_building_index == -1:
                    st.warning(f"Skipping '{file_path}': No valid building ('opmc' or 'witel') found in path.")
                    continue

                remaining_parts = path_parts[first_building_index:]
                
                # Inisialisasi variabel sebelum digunakan
                building = "Unknown"
                floor = "Unknown"
                device_type = "Unknown"
                original_filename = "Unknown"

                if len(remaining_parts) == 4:
                    building = remaining_parts[0].strip()
                    floor = remaining_parts[1].strip()
                    device_type = remaining_parts[2].strip()
                    original_filename = remaining_parts[3].strip()
                elif len(remaining_parts) == 3:
                    building = remaining_parts[0].strip()
                    device_type = remaining_parts[1].strip()
                    floor = device_type.capitalize()
                    original_filename = remaining_parts[2].strip()
                else:
                    st.warning(f"Skipping '{file_path}': Invalid folder structure after building.")
                    continue

                # --- LOGIKA MEMBACA FILE (FIXED) ---
                try:
                    # Read file bytes from the ZIP archive
                    file_bytes = zip_ref.read(file_info.filename)
                    csv_file_object = io.StringIO(file_bytes.decode('utf-8'))

                    df, building, device_type, floor, error = load_and_process_data(
                        csv_file_object, building, device_type, floor, original_filename, minimum_rows
                    )
                except Exception as read_error:
                    st.error(f"Failed to read or process file {file_path}: {read_error}")
                    continue
                
                # --- LANJUTAN LOGIKA ---
                if error:
                    st.error(error, icon="⚠️")
                    continue

                if df is None: # Pengecekan tambahan jika load_and_process_data gagal
                    continue
                
                device_id = f"{building}-{device_type.upper()}-{floor}-{i+1}"
                
                valid_files[device_id] = {
                    'df': df.copy(),
                    'file_name': original_filename,
                    'device_id': device_id,
                    'building': building,
                    'device_type': device_type,
                    'floor': floor
                }
                
                filename_info.append({
                    'File Name': original_filename,
                    'Building': building,
                    'Device Type': device_type.upper(),
                    'Floor': floor,
                    'Records': len(df),
                    'Device ID': device_id
                })
                
                st.session_state.uploaded_files[device_id] = valid_files[device_id]

        status_text.success(f"✅ Processing complete. {len(valid_files)} valid data file(s) loaded.")
        progress_bar.empty()

    except zipfile.BadZipFile:
        st.error("❌ The uploaded file is not a valid ZIP file.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.stop()

    if not valid_files:
        st.error(f"❌ No valid CSV files found or processed from the ZIP. Files must have at least {minimum_rows} data points.")
        st.stop()
    
    # ... rest of your code continues here ...
    
    st.subheader("📁 File Information (Auto-Detected from ZIP)")
    filename_df = pd.DataFrame(filename_info)
    st.dataframe(filename_df, use_container_width=True)
    ### UPDATE ###
    csv = to_csv(filename_df)
    st.download_button(
        label="Download File Info as CSV",
        data=csv,
        file_name='file_information.csv',
        mime='text/csv',
        key='download-file-info'
    )
    
    # Create comprehensive aggregated analysis
    building_agg, device_agg, floor_agg = create_aggregated_analysis(valid_files)

    # Create tabs
    tab_list = [
        "📈 Data Overview", "🏢 Building Analysis", "🔧 Device Analysis", "🏢 Floor Analysis", 
        "🤖 Model Training", "🔍 Individual Analysis", "💰 Economic Analysis", "📊 Methodology"
    ]
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(tab_list)
    
    with tab1:
        st.subheader("📈 Data Overview - All Files")
        total_records = sum(len(data['df']) for data in valid_files.values())
        total_consumption = sum(data['df'][TARGET_VARIABLE].sum() for data in valid_files.values())
        avg_consumption = total_consumption / total_records if total_records > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Total Devices", len(valid_files))
        with col2: st.metric("Total Records", f"{total_records:,}")
        with col3: st.metric("Total Consumption", f"{total_consumption/1000:.2f} kWh")
        with col4: st.metric("Avg Consumption", f"{avg_consumption:.2f} Wh")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Buildings", len(building_agg))
            for building in building_agg.keys(): 
                st.write(f"• {building}")
        with col2:
            st.metric("Device Types", len(device_agg))
            for device_type in device_agg.keys(): 
                st.write(f"• {device_type.upper()}")
        with col3:
            st.metric("Floors", len(floor_agg))
            for floor in floor_agg.keys(): 
                st.write(f"• {floor}")
        
        # Time series visualization
        fig = go.Figure()
        for device_id, data in valid_files.items():
            df = data['df']
            fig.add_trace(go.Scatter(
                x=df.index, y=df[TARGET_VARIABLE], mode='lines', 
                name=f"{data['building']}-{data['device_type'].upper()}-{data['floor']}", 
                line=dict(width=1), opacity=0.7
            ))
        fig.update_layout(
            title="Energy Consumption Over Time - All Devices", 
            xaxis_title="Time", yaxis_title="Consumption (Wh)", height=500
        )
        st.plotly_chart(fig, use_container_width=True, key="overview_timeseries")
        
        # Central Limit Theorem Analysis
        st.markdown("---")
        st.subheader("Central Limit Theorem Analysis - All Data")
        all_consumption_data = np.concatenate([data['df'][TARGET_VARIABLE].values for data in valid_files.values()])
        visualize_clt(all_consumption_data, "All Devices", "overview_clt")
    
    with tab2:
        st.subheader("🏢 Building-Level Analysis")
        
        # Building metrics table
        building_metrics = []
        for building, data in building_agg.items():
            building_metrics.append({
                'Building': building,
                'Total Consumption (kWh)': data['total_consumption']/1000,
                'Average Consumption (Wh)': data['avg_consumption'],
                'Device Count': data['device_count'],
                'Peak Hour': data['patterns']['peak_hour'],
                'Peak Consumption (Wh)': data['patterns']['peak_consumption']
            })
        
        building_df = pd.DataFrame(building_metrics)
        st.dataframe(building_df, use_container_width=True)
        ### UPDATE ###
        csv = to_csv(building_df)
        st.download_button("Download Building Metrics as CSV", csv, "building_metrics.csv", "text/csv", key='download-building-metrics')
        
        # Device type analysis within buildings
        for building, data in building_agg.items():
            st.markdown(f"### {building.upper()} - Device Type Distribution")
            
            # Get device types in this building
            building_device_types = {}
            for device_id, device_data in valid_files.items():
                if device_data['building'] == building:
                    dev_type = device_data['device_type']
                    if dev_type not in building_device_types:
                        building_device_types[dev_type] = []
                    building_device_types[dev_type].append(device_data['df'])
            
            # Create aggregated device type data for this building
            building_device_agg = {}
            for dev_type, dfs in building_device_types.items():
                combined_df = pd.concat(dfs, ignore_index=False)
                building_device_agg[dev_type] = {
                    'df': combined_df,
                    'total_consumption': combined_df[TARGET_VARIABLE].sum(),
                    'device_count': len(dfs)
                }
            
            # Create pie chart for this building
            fig_pie, df_pie = analyze_and_visualize_consumption_by_device_type(
                building_device_agg, f"{building.upper()} Building"
            )
            if fig_pie:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.plotly_chart(fig_pie, use_container_width=True, key=f"building_pie_{building}")
                with col2:
                    st.dataframe(df_pie[['Tipe Perangkat', 'Rata-rata per Unit (kWh)', 'Persentase (%)']], 
                                 use_container_width=True)
                    ### UPDATE ###
                    csv = to_csv(df_pie)
                    st.download_button(f"Download {building} Device Data as CSV", csv, f"{building}_device_distribution.csv", "text/csv", key=f'download-pie-{building}')
        
        # Building comparison charts
        if len(building_agg) > 1:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(building_df, x='Building', y='Total Consumption (kWh)', 
                             title='Total Consumption by Building', 
                             color='Total Consumption (kWh)', color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True, key="building_total_bar")
            with col2:
                fig = px.bar(building_df, x='Building', y='Device Count', 
                             title='Device Count by Building', 
                             color='Device Count', color_continuous_scale='Greens')
                st.plotly_chart(fig, use_container_width=True, key="building_count_bar")
        
        # Detailed building analysis
        selected_building = st.selectbox("Select Building for Detailed Analysis", 
                                         list(building_agg.keys()), key="building_select")
        if selected_building:
            building_data = building_agg[selected_building]
            patterns = building_data['patterns']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Total Consumption", f"{building_data['total_consumption']/1000:.2f} kWh")
            with col2: st.metric("Average Consumption", f"{building_data['avg_consumption']:.2f} Wh")
            with col3: st.metric("Peak Hour", f"{patterns['peak_hour']}:00")
            with col4: st.metric("Peak Consumption", f"{patterns['peak_consumption']:.2f} Wh")
            
            # Consumption patterns
            col1, col2, col3 = st.columns(3)
            with col1:
                fig = go.Figure(go.Scatter(
                    x=list(patterns['hourly'].index), y=patterns['hourly'].values, 
                    mode='lines+markers', name=f'{selected_building} Hourly Pattern'
                ))
                fig.update_layout(title=f"Hourly Pattern - {selected_building}", 
                                  xaxis_title="Hour of Day", yaxis_title="Consumption (Wh)", height=400)
                st.plotly_chart(fig, use_container_width=True, key=f"building_hourly_{selected_building}")
            
            with col2:
                days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                fig = go.Figure(go.Bar(x=days, y=patterns['daily'].values, 
                                       name=f'{selected_building} Daily Pattern', marker_color='lightblue'))
                fig.update_layout(title=f"Daily Pattern - {selected_building}", 
                                  xaxis_title="Day of Week", yaxis_title="Consumption (Wh)", height=400)
                st.plotly_chart(fig, use_container_width=True, key=f"building_daily_{selected_building}")
            
            with col3:
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                month_labels = [months[i-1] for i in patterns['monthly'].index]
                fig = go.Figure(go.Bar(x=month_labels, y=patterns['monthly'].values, 
                                       name=f'{selected_building} Monthly Pattern', marker_color='lightcoral'))
                fig.update_layout(title=f"Monthly Pattern - {selected_building}", 
                                  xaxis_title="Month", yaxis_title="Consumption (Wh)", height=400)
                st.plotly_chart(fig, use_container_width=True, key=f"building_monthly_{selected_building}")
            
            # Consumption distribution
            fig = px.histogram(building_data['df'], x=TARGET_VARIABLE, nbins=50, 
                               title=f'Consumption Distribution - {selected_building}')
            fig.update_layout(xaxis_title='Consumption (Wh)', yaxis_title='Frequency')
            st.plotly_chart(fig, use_container_width=True, key=f"building_dist_{selected_building}")
            
            # Central Limit Theorem for selected building
            st.markdown("---")
            st.subheader(f"Central Limit Theorem Analysis - {selected_building}")
            visualize_clt(building_data['df'][TARGET_VARIABLE].values, selected_building, f"building_clt_{selected_building}")

    with tab3:
        st.subheader("🔧 Device Type Analysis")
        
        # Device type metrics
        device_metrics = []
        for device_type, data in device_agg.items():
            device_metrics.append({
                'Device Type': device_type.upper(),
                'Total Consumption (kWh)': data['total_consumption']/1000,
                'Average Consumption (Wh)': data['avg_consumption'],
                'Device Count': data['device_count'],
                'Peak Hour': data['patterns']['peak_hour'],
                'Peak Consumption (Wh)': data['patterns']['peak_consumption']
            })
        
        device_df = pd.DataFrame(device_metrics)
        st.dataframe(device_df, use_container_width=True)
        ### UPDATE ###
        csv = to_csv(device_df)
        st.download_button("Download Device Metrics as CSV", csv, "device_type_metrics.csv", "text/csv", key='download-device-metrics')
        
        # Average consumption per unit pie chart
        st.markdown("### Average Consumption Analysis per Unit")
        fig_pie, df_pie = analyze_and_visualize_consumption_by_device_type(device_agg, "Device Type")
        if fig_pie:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.plotly_chart(fig_pie, use_container_width=True, key="device_type_pie")
            with col2:
                st.dataframe(df_pie[['Tipe Perangkat', 'Jumlah Unit', 'Rata-rata per Unit (kWh)', 'Persentase (%)']], 
                             use_container_width=True)
                ### UPDATE ###
                csv = to_csv(df_pie)
                st.download_button("Download Device Distribution as CSV", csv, "device_type_distribution.csv", "text/csv", key='download-device-pie')

        # Additional charts
        if len(device_agg) > 1:
            fig_bar = px.bar(device_df, x='Device Type', y='Average Consumption (Wh)', 
                             title='Average Consumption by Device Type', 
                             color='Average Consumption (Wh)', color_continuous_scale='Oranges')
            st.plotly_chart(fig_bar, use_container_width=True, key="device_avg_bar")

        # Detailed device type analysis
        selected_device_type = st.selectbox("Select Device Type for Detailed Analysis", 
                                            list(device_agg.keys()), key="device_select")
        if selected_device_type:
            device_data = device_agg[selected_device_type]
            patterns = device_data['patterns']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Total Consumption", f"{device_data['total_consumption']/1000:.2f} kWh")
            with col2: st.metric("Average Consumption", f"{device_data['avg_consumption']:.2f} Wh")
            with col3: st.metric("Peak Hour", f"{patterns['peak_hour']}:00")
            with col4: st.metric("Device Count", device_data['device_count'])
            
            # Consumption patterns
            col1, col2, col3 = st.columns(3)
            with col1:
                fig = go.Figure(go.Scatter(
                    x=list(patterns['hourly'].index), y=patterns['hourly'].values, 
                    mode='lines+markers', name=f'{selected_device_type.upper()} Hourly Pattern', 
                    line=dict(color='orange')
                ))
                fig.update_layout(title=f"Hourly Pattern - {selected_device_type.upper()}", 
                                  xaxis_title="Hour of Day", yaxis_title="Consumption (Wh)", height=400)
                st.plotly_chart(fig, use_container_width=True, key=f"device_hourly_{selected_device_type}")
            
            with col2:
                days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                fig = go.Figure(go.Bar(x=days, y=patterns['daily'].values, 
                                       name=f'{selected_device_type.upper()} Daily Pattern', 
                                       marker_color='lightblue'))
                fig.update_layout(title=f"Daily Pattern - {selected_device_type.upper()}", 
                                  xaxis_title="Day of Week", yaxis_title="Consumption (Wh)", height=400)
                st.plotly_chart(fig, use_container_width=True, key=f"device_daily_{selected_device_type}")
            
            with col3:
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                month_labels = [months[i-1] for i in patterns['monthly'].index]
                fig = go.Figure(go.Bar(x=month_labels, y=patterns['monthly'].values, 
                                       name=f'{selected_device_type.upper()} Monthly Pattern', 
                                       marker_color='lightgreen'))
                fig.update_layout(title=f"Monthly Pattern - {selected_device_type.upper()}", 
                                  xaxis_title="Month", yaxis_title="Consumption (Wh)", height=400)
                st.plotly_chart(fig, use_container_width=True, key=f"device_monthly_{selected_device_type}")
            
            # Consumption distribution
            fig = px.histogram(device_data['df'], x=TARGET_VARIABLE, nbins=50, 
                               title=f'Consumption Distribution - {selected_device_type.upper()}')
            fig.update_layout(xaxis_title='Consumption (Wh)', yaxis_title='Frequency')
            st.plotly_chart(fig, use_container_width=True, key=f"device_dist_{selected_device_type}")
            
            # Central Limit Theorem for selected device type
            st.markdown("---")
            st.subheader(f"Central Limit Theorem Analysis - {selected_device_type.upper()}")
            visualize_clt(device_data['df'][TARGET_VARIABLE].values, selected_device_type.upper(), 
                          f"device_clt_{selected_device_type}")

    with tab4:
        st.subheader("🏢 Floor Analysis")
        
        # Floor metrics
        floor_metrics = []
        for floor_key, data in floor_agg.items():
            floor_metrics.append({
                'Building-Floor': floor_key,
                'Total Consumption (kWh)': data['total_consumption']/1000,
                'Average Consumption (Wh)': data['avg_consumption'],
                'Device Count': data['device_count'],
                'Peak Hour': data['patterns']['peak_hour'],
                'Peak Consumption (Wh)': data['patterns']['peak_consumption']
            })
        
        floor_df = pd.DataFrame(floor_metrics)
        st.dataframe(floor_df, use_container_width=True)
        ### UPDATE ###
        csv = to_csv(floor_df)
        st.download_button("Download Floor Metrics as CSV", csv, "floor_metrics.csv", "text/csv", key='download-floor-metrics')
        
        if len(floor_agg) > 1:
            fig = px.bar(floor_df, x='Building-Floor', y='Total Consumption (kWh)', 
                         title='Total Consumption by Floor', 
                         color='Total Consumption (kWh)', color_continuous_scale='Reds')
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True, key="floor_bar")
        
        # Detailed floor analysis
        selected_floor = st.selectbox("Select Floor for Detailed Analysis", 
                                      list(floor_agg.keys()), key="floor_select")
        if selected_floor:
            floor_data = floor_agg[selected_floor]
            patterns = floor_data['patterns']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Total Consumption", f"{floor_data['total_consumption']/1000:.2f} kWh")
            with col2: st.metric("Average Consumption", f"{floor_data['avg_consumption']:.2f} Wh")
            with col3: st.metric("Peak Hour", f"{patterns['peak_hour']}:00")
            with col4: st.metric("Device Count", floor_data['device_count'])
            
            # Consumption patterns
            col1, col2, col3 = st.columns(3)
            with col1:
                fig = go.Figure(go.Scatter(
                    x=list(patterns['hourly'].index), y=patterns['hourly'].values, 
                    mode='lines+markers', name=f'{selected_floor} Hourly Pattern', 
                    line=dict(color='red')
                ))
                fig.update_layout(title=f"Hourly Pattern - {selected_floor}", 
                                  xaxis_title="Hour of Day", yaxis_title="Consumption (Wh)", height=400)
                st.plotly_chart(fig, use_container_width=True, key=f"floor_hourly_{selected_floor}")
            
            with col2:
                days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                fig = go.Figure(go.Bar(x=days, y=patterns['daily'].values, 
                                       name=f'{selected_floor} Daily Pattern', marker_color='lightpink'))
                fig.update_layout(title=f"Daily Pattern - {selected_floor}", 
                                  xaxis_title="Day of Week", yaxis_title="Consumption (Wh)", height=400)
                st.plotly_chart(fig, use_container_width=True, key=f"floor_daily_{selected_floor}")
            
            with col3:
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                month_labels = [months[i-1] for i in patterns['monthly'].index]
                fig = go.Figure(go.Bar(x=month_labels, y=patterns['monthly'].values, 
                                       name=f'{selected_floor} Monthly Pattern', marker_color='lightcoral'))
                fig.update_layout(title=f"Monthly Pattern - {selected_floor}", 
                                  xaxis_title="Month", yaxis_title="Consumption (Wh)", height=400)
                st.plotly_chart(fig, use_container_width=True, key=f"floor_monthly_{selected_floor}")
            
            # Consumption distribution
            fig = px.histogram(floor_data['df'], x=TARGET_VARIABLE, nbins=50, 
                               title=f'Consumption Distribution - {selected_floor}')
            fig.update_layout(xaxis_title='Consumption (Wh)', yaxis_title='Frequency')
            st.plotly_chart(fig, use_container_width=True, key=f"floor_dist_{selected_floor}")
            
            # Central Limit Theorem for selected floor
            st.markdown("---")
            st.subheader(f"Central Limit Theorem Analysis - {selected_floor}")
            visualize_clt(floor_data['df'][TARGET_VARIABLE].values, selected_floor, f"floor_clt_{selected_floor}")

    with tab5:
        st.subheader("🤖 Model Training and Evaluation")

        training_mode = st.radio(
            "Select Training Mode",
            ("Train on a single device", "Bulk Training (Individual & Aggregated)"),
            key="training_mode",
            horizontal=True
        )
        st.markdown("---")

        if training_mode == "Train on a single device":
            st.caption("Mode ini cocok untuk melakukan analisis mendalam pada satu perangkat spesifik.")
            device_options = list(valid_files.keys())
            selected_device_for_training = st.selectbox("Select Device for Model Training", 
                                                        device_options, key="single_device_select")
            
            if st.button("Train Model & Calculate Savings for Selected Device", key="single_train_btn"):
                run_training_process(selected_device_for_training, valid_files[selected_device_for_training])

            if selected_device_for_training in st.session_state.analysis_results:
                display_detailed_results(selected_device_for_training)

        else: # Bulk Training (Individual & Aggregated)
            st.subheader("Bulk Training - Individual Devices")
            st.caption("Pilih beberapa perangkat individu untuk dilatih secara massal.")
            with st.form(key="bulk_individual_form"):
                devices_by_building = {}
                for device_id, data in valid_files.items():
                    building = data['building']
                    if building not in devices_by_building:
                        devices_by_building[building] = []
                    devices_by_building[building].append(device_id)

                select_all_individual = st.checkbox("Select/Deselect All Devices", value=False)
                selected_devices_map = {}

                for building in sorted(devices_by_building.keys()):
                    with st.expander(f"🏢 {building.upper()}", expanded=True):
                        select_all_building = st.checkbox(f"Select All in {building.upper()}", value=select_all_individual, key=f"select_all_{building}")
                        for device_id in sorted(devices_by_building[building]):
                            selected_devices_map[device_id] = st.checkbox(device_id, value=select_all_building, key=device_id)

                submitted_individual = st.form_submit_button("Start Bulk Training for Selected Individual Devices")

                if submitted_individual:
                    devices_to_train = [device_id for device_id, is_selected in selected_devices_map.items() if is_selected]
                    
                    if not devices_to_train:
                        st.warning("Please select at least one device to train.")
                    else:
                        st.session_state.last_bulk_run_devices = devices_to_train
                        st.session_state.training_summary = {}
                        
                        total_tasks = len(devices_to_train)
                        progress_bar = st.progress(0, text="Starting bulk training...")
                        status_text = st.empty()
                        
                        summary_data = []
                        
                        for i, device_id in enumerate(devices_to_train):
                            device_data = valid_files[device_id]
                            status_text.text(f"Training device {i+1}/{total_tasks}: {device_id}")

                            # Define a callback to update the main progress bar from the inner function
                            def update_progress(model_progress):
                                # model_progress is 1/3, 2/3, or 1.0
                                overall_progress = (i + model_progress) / total_tasks
                                progress_bar.progress(overall_progress, text=f"Device {i+1}/{total_tasks}: {device_id}")

                            run_training_process(device_id, device_data, status_text=status_text, progress_callback=update_progress)
                            
                            if device_id in st.session_state.analysis_results:
                                results = st.session_state.analysis_results[device_id]
                                model_results = results['model_results']
                                best_model = results['best_model_name']
                                
                                summary_data.append({
                                    'Device_ID': device_id,
                                    'Building': device_data['building'],
                                    'Device_Type': device_data['device_type'].upper(),
                                    'Floor': device_data['floor'],
                                    'Best_Model': best_model,
                                    'RF_MAE': model_results['RandomForest']['metrics']['mae'],
                                    'RF_RMSE': model_results['RandomForest']['metrics']['rmse'],
                                    'RF_R2': model_results['RandomForest']['metrics']['r2'],
                                    'GB_MAE': model_results['GradientBoosting']['metrics']['mae'],
                                    'GB_RMSE': model_results['GradientBoosting']['metrics']['rmse'],
                                    'GB_R2': model_results['GradientBoosting']['metrics']['r2'],
                                    'LSTM_MAE': model_results['LSTM']['metrics']['mae'],
                                    'LSTM_RMSE': model_results['LSTM']['metrics']['rmse'],
                                    'LSTM_R2': model_results['LSTM']['metrics']['r2']
                                })

                            # Ensure the progress bar marks the completion of one full device
                            progress_bar.progress((i + 1) / total_tasks, text=f"Completed: {device_id}")

                        st.session_state.training_summary['summary_data'] = summary_data
                        status_text.success("Pelatihan massal untuk perangkat individu selesai!")
                        progress_bar.empty()

            # Display the summary if it exists in the session state (from the run above, or a previous run)
            if 'summary_data' in st.session_state.training_summary and st.session_state.training_summary['summary_data']:
                st.markdown("---")
                st.subheader("📊 Individual Training Summary")
                
                summary_df = pd.DataFrame(st.session_state.training_summary['summary_data'])
                
                summary_tab1, summary_tab2, summary_tab3 = st.tabs(["📋 Results Table", "🔥 Performance Heatmaps", "📈 Model Comparison"])
                
                with summary_tab1:
                    st.dataframe(summary_df, use_container_width=True)
                    ### UPDATE ###
                    csv = to_csv(summary_df)
                    st.download_button("Download Training Summary as CSV", csv, "training_summary.csv", "text/csv", key='download-training-summary')
                
                with summary_tab2:
                    st.markdown("<h5>Model Performance Heatmap (MAE)</h5>", unsafe_allow_html=True)
                    st.caption("Lower is better. Blue indicates better performance.")
                    heatmap_mae = summary_df[['Device_ID', 'RF_MAE', 'GB_MAE', 'LSTM_MAE']].set_index('Device_ID')
                    heatmap_mae.columns = ['Random Forest', 'Gradient Boosting', 'LSTM']
                    fig_mae = go.Figure(data=go.Heatmap(
                        z=heatmap_mae.values.T, x=heatmap_mae.index, y=heatmap_mae.columns,
                        colorscale='RdYlBu_r', text=np.around(heatmap_mae.values.T, 2),
                        texttemplate="%{text}", textfont={"size": 10}
                    ))
                    fig_mae.update_layout(title='Model Performance (MAE)', xaxis_title='Device ID', yaxis_title='Model Type', height=400)
                    st.plotly_chart(fig_mae, use_container_width=True, key="mae_heatmap")

                    st.markdown("<h5>Model Performance Heatmap (RMSE)</h5>", unsafe_allow_html=True)
                    st.caption("Lower is better. Blue indicates better performance.")
                    heatmap_rmse = summary_df[['Device_ID', 'RF_RMSE', 'GB_RMSE', 'LSTM_RMSE']].set_index('Device_ID')
                    heatmap_rmse.columns = ['Random Forest', 'Gradient Boosting', 'LSTM']
                    fig_rmse = go.Figure(data=go.Heatmap(
                        z=heatmap_rmse.values.T, x=heatmap_rmse.index, y=heatmap_rmse.columns,
                        colorscale='RdYlBu_r', text=np.around(heatmap_rmse.values.T, 2),
                        texttemplate="%{text}", textfont={"size": 10}
                    ))
                    fig_rmse.update_layout(title='Model Performance (RMSE)', xaxis_title='Device ID', yaxis_title='Model Type', height=400)
                    st.plotly_chart(fig_rmse, use_container_width=True, key="rmse_heatmap")
                    
                    st.markdown("<h5>Model Performance Heatmap (R² Score)</h5>", unsafe_allow_html=True)
                    st.caption("Higher is better. Green indicates better performance.")
                    heatmap_r2 = summary_df[['Device_ID', 'RF_R2', 'GB_R2', 'LSTM_R2']].set_index('Device_ID')
                    heatmap_r2.columns = ['Random Forest', 'Gradient Boosting', 'LSTM']
                    fig_r2 = go.Figure(data=go.Heatmap(
                        z=heatmap_r2.values.T, x=heatmap_r2.index, y=heatmap_r2.columns,
                        colorscale='RdYlGn', zmin=0, zmax=1,
                        text=np.around(heatmap_r2.values.T, 2),
                        texttemplate="%{text}", textfont={"size": 10}
                    ))
                    fig_r2.update_layout(title='Model Performance (R² Score)', xaxis_title='Device ID', yaxis_title='Model Type', height=400)
                    st.plotly_chart(fig_r2, use_container_width=True, key="r2_heatmap")

                with summary_tab3:
                    avg_performance = {
                        'Random Forest': {
                            'MAE': summary_df['RF_MAE'].mean(), 'RMSE': summary_df['RF_RMSE'].mean(), 'R²': summary_df['RF_R2'].mean()
                        },
                        'Gradient Boosting': {
                            'MAE': summary_df['GB_MAE'].mean(), 'RMSE': summary_df['GB_RMSE'].mean(), 'R²': summary_df['GB_R2'].mean()
                        },
                        'LSTM': {
                            'MAE': summary_df['LSTM_MAE'].mean(), 'RMSE': summary_df['LSTM_RMSE'].mean(), 'R²': summary_df['LSTM_R2'].mean()
                        }
                    }
                    models = list(avg_performance.keys())
                    mae_values = [avg_performance[model]['MAE'] for model in models]
                    r2_values = [avg_performance[model]['R²'] for model in models]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_mae_avg = px.bar(x=models, y=mae_values, title='Average MAE by Model')
                        fig_mae_avg.update_yaxes(title='Mean Absolute Error (Wh)')
                        st.plotly_chart(fig_mae_avg, use_container_width=True, key="avg_mae_comparison")
                    
                    with col2:
                        fig_r2_avg = px.bar(x=models, y=r2_values, title='Average R² Score by Model')
                        fig_r2_avg.update_yaxes(title='R² Score')
                        st.plotly_chart(fig_r2_avg, use_container_width=True, key="avg_r2_comparison")
                        
                    best_model_counts = summary_df['Best_Model'].value_counts()
                    fig_best = px.pie(values=best_model_counts.values, names=best_model_counts.index,
                                      title='Distribution of Best Performing Models')
                    st.plotly_chart(fig_best, use_container_width=True, key="best_model_distribution")

            st.markdown("---")
            st.subheader("Aggregate Model Training")
            st.caption("Latih model pada dataset yang dibuat dengan menggabungkan beberapa perangkat.")
            with st.form(key="bulk_aggregate_form"):
                agg_options = {
                    "all_data": st.checkbox("1. All Data Combined: Train one model on all available data."),
                    "per_building": st.checkbox("2. Per Building: Train one model for each building (e.g., 'OPMC', 'WITEL')."),
                    "per_floor": st.checkbox("3. Per Floor (in each Building): Train one model for each floor (e.g., 'OPMC-LT.1')."),
                    "per_device_type": st.checkbox("4. Per Device Type (Combined): Train one model for each device type across all buildings (e.g., 'AHU')."),
                    "per_device_type_building": st.checkbox("5. Per Device Type per Building: Train one model for each device type within each building (e.g., 'OPMC-AHU').")
                }
                submitted_aggregate = st.form_submit_button("Start Aggregate Training")

                if submitted_aggregate:
                    training_tasks = {}
                    
                    # 1. All data
                    if agg_options["all_data"]:
                        all_dfs = [data['df'].copy() for data in valid_files.values()]
                        if all_dfs:
                            training_tasks["AGG-ALL_DATA"] = pd.concat(all_dfs).sort_index()

                    # 2. Per Building
                    if agg_options["per_building"]:
                        buildings = {data['building'] for data in valid_files.values()}
                        for building in buildings:
                            dfs = [data['df'].copy() for data in valid_files.values() if data['building'] == building]
                            if dfs:
                                training_tasks[f"AGG-BUILDING-{building.upper()}"] = pd.concat(dfs).sort_index()

                    # 3. Per Floor per Building
                    if agg_options["per_floor"]:
                        floors = {f"{data['building']}-{data['floor']}" for data in valid_files.values()}
                        for floor_key in floors:
                            building, floor = floor_key.split('-', 1)
                            dfs = [data['df'].copy() for data in valid_files.values() if data['building'] == building and data['floor'] == floor]
                            if dfs:
                                training_tasks[f"AGG-FLOOR-{floor_key.upper()}"] = pd.concat(dfs).sort_index()

                    # 4. Per Device Type
                    if agg_options["per_device_type"]:
                        device_types = {data['device_type'] for data in valid_files.values()}
                        for dt in device_types:
                            dfs = [data['df'].copy() for data in valid_files.values() if data['device_type'] == dt]
                            if dfs:
                                training_tasks[f"AGG-DEVICE-{dt.upper()}"] = pd.concat(dfs).sort_index()

                    # 5. Per Device Type per Building
                    if agg_options["per_device_type_building"]:
                        combos = {(data['building'], data['device_type']) for data in valid_files.values()}
                        for building, dt in combos:
                            dfs = [data['df'].copy() for data in valid_files.values() if data['building'] == building and data['device_type'] == dt]
                            if dfs:
                                training_tasks[f"AGG-{building.upper()}-{dt.upper()}"] = pd.concat(dfs).sort_index()
                    
                    if not training_tasks:
                        st.warning("No aggregation options selected.")
                    else:
                        st.info(f"Starting training for {len(training_tasks)} aggregated model(s).")
                        total_agg_tasks = len(training_tasks)
                        progress_bar_agg = st.progress(0, text="Starting aggregate training...")
                        status_text_agg = st.empty()
                        
                        for i, (task_name, agg_df) in enumerate(training_tasks.items()):
                            status_text_agg.text(f"Training aggregate {i+1}/{total_agg_tasks}: {task_name}")
                            
                            # Re-engineer features for the aggregated dataframe
                            engineered_df = engineer_features(agg_df)
                            
                            if len(engineered_df) < minimum_rows:
                                st.warning(f"Skipping '{task_name}': Insufficient data ({len(engineered_df)} rows).")
                                continue

                            def update_agg_progress(model_progress):
                                overall_progress = (i + model_progress) / total_agg_tasks
                                progress_bar_agg.progress(overall_progress, text=f"Aggregate {i+1}/{total_agg_tasks}: {task_name}")

                            synthetic_data = {'df': engineered_df, 'building': 'Agg', 'device_type': task_name, 'floor': 'N/A'}
                            run_training_process(task_name, synthetic_data, status_text=status_text_agg, progress_callback=update_agg_progress)
                            
                            # Ensure progress bar marks full completion for the aggregate task
                            progress_bar_agg.progress((i + 1) / total_agg_tasks, text=f"Completed: {task_name}")

                        st.success("Aggregate model training complete!")

            # Display training summary and results after any training run
            if st.session_state.analysis_results:
                st.markdown("---")
                st.subheader("📊 Training Results")
                
                trained_devices = list(st.session_state.analysis_results.keys())
                selected_device_for_results = st.selectbox(
                    "Select a device/aggregate to view detailed results",
                    options=trained_devices,
                    index=len(trained_devices)-1 # Default to the last trained model
                )
                if selected_device_for_results:
                    display_detailed_results(selected_device_for_results)

    with tab6:
        st.subheader("🔍 Individual Device Analysis")
        device_options = list(valid_files.keys())
        selected_device = st.selectbox("Select Device for Individual Analysis", 
                                       device_options, key="individual_analysis")
        if selected_device:
            device_data = valid_files[selected_device]
            df = device_data['df']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Total Consumption", f"{df[TARGET_VARIABLE].sum()/1000:.2f} kWh")
            with col2: st.metric("Average Consumption", f"{df[TARGET_VARIABLE].mean():.2f} Wh")
            with col3: st.metric("Max Consumption", f"{df[TARGET_VARIABLE].max():.2f} Wh")
            with col4: st.metric("Data Points", len(df))
            
            patterns = analyze_consumption_patterns(df, selected_device)
            
            fig = go.Figure(go.Scatter(
                x=df.index, y=df[TARGET_VARIABLE], mode='lines', 
                name='Energy Consumption', line=dict(width=1)
            ))
            fig.update_layout(
                title=f'Energy Consumption Time Series - {selected_device}', 
                xaxis_title='Time', yaxis_title='Consumption (Wh)', height=400
            )
            st.plotly_chart(fig, use_container_width=True, key=f"individual_timeseries_{selected_device}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                fig = px.line(x=patterns['hourly'].index, y=patterns['hourly'].values, 
                              title='Hourly Consumption Pattern')
                fig.update_layout(xaxis_title='Hour of Day', yaxis_title='Average Consumption (Wh)')
                st.plotly_chart(fig, use_container_width=True, key=f"individual_hourly_{selected_device}")
            
            with col2:
                days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                fig = px.bar(x=days, y=patterns['daily'].values, title='Daily Consumption Pattern')
                fig.update_layout(xaxis_title='Day of Week', yaxis_title='Average Consumption (Wh)')
                st.plotly_chart(fig, use_container_width=True, key=f"individual_daily_{selected_device}")
            
            with col3:
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                month_labels = [months[i-1] for i in patterns['monthly'].index]
                fig = px.bar(x=month_labels, y=patterns['monthly'].values, title='Monthly Consumption Pattern')
                fig.update_layout(xaxis_title='Month', yaxis_title='Average Consumption (Wh)')
                st.plotly_chart(fig, use_container_width=True, key=f"individual_monthly_{selected_device}")
            
            fig = px.histogram(df, x=TARGET_VARIABLE, nbins=50, title='Consumption Distribution')
            fig.update_layout(xaxis_title='Consumption (Wh)', yaxis_title='Frequency')
            st.plotly_chart(fig, use_container_width=True, key=f"individual_distribution_{selected_device}")
            
            # Central Limit Theorem for selected device
            st.markdown("---")
            st.subheader(f"Central Limit Theorem Analysis - {selected_device}")
            visualize_clt(df[TARGET_VARIABLE].values, selected_device, f"individual_clt_{selected_device}")

    with tab7:
        st.subheader("💰 Economic Feasibility Analysis")
        
        # Create sub-tabs for different analysis types
        econ_tab1, econ_tab2 = st.tabs(["📊 Model-Based Analysis", "🧮 Manual Project Calculator"])
        
        with econ_tab1:
            st.subheader("Model-Based Anomaly Detection Economic Analysis")
            
            total_savings_wh = 0
            duration_days = 0
            savings_available = False
            
            if 'model_based_savings' in st.session_state and st.session_state.model_based_savings:
                total_savings_wh = sum(item['savings_wh'] for item in st.session_state.model_based_savings.values())
                
                # Calculate duration from any available data
                for device_id, device_data in valid_files.items():
                    df = device_data['df']
                    if not df.empty:
                        duration_days = (df.index.max() - df.index.min()).days + 1
                        break
                
                if total_savings_wh > 0: 
                    savings_available = True
                st.info(f"Aggregated savings from **{len(st.session_state.model_based_savings)}** trained model(s).")
            else: 
                st.warning("Please visit the '🤖 Model Training' tab and train at least one model to generate anomaly-based savings.")
            
            if savings_available and duration_days > 0:
                annual_savings_kwh = (total_savings_wh / duration_days) * 365 / 1000
                st.markdown(f"""
                <div class="economic-card">
                    <h4>Annual Energy Savings Potential (Model-Based Anomaly Detection)</h4>
                    <p style="font-size: 1.5rem; font-weight: bold;">{annual_savings_kwh:,.2f} kWh / year</p>
                    <small>This is an annualized projection based on the anomalies identified in the uploaded data period of ~{duration_days} days.</small>
                </div>
                """, unsafe_allow_html=True)
                
                st.warning("Note: Annualization is a simple linear extrapolation and may not capture seasonal variations.")
                
                st.markdown("Enter the project's financial details below to calculate its viability.")
                
                with st.form(key="economic_form_model"):
                    st.markdown("<h5>Project Investment</h5>", unsafe_allow_html=True)
                    capex = st.number_input("Initial Investment (CAPEX) (Rp)", min_value=0, value=0, step=100000)
                    
                    st.markdown("<h5>Operating Parameters</h5>", unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    with col1:
                        annual_maintenance = st.number_input("Annual Maintenance Cost (OPEX) (Rp)", min_value=0, value=0, step=50000)
                        project_years = st.number_input("Project Lifetime (Years)", min_value=1, max_value=50, value=10)
                    with col2:
                        discount_rate = st.slider("Discount Rate (%)", min_value=0.0, max_value=25.0, value=8.0, step=0.5)
                    
                    submit_button = st.form_submit_button(label="Calculate Financial Metrics")
                
                if submit_button:
                    with st.spinner("Calculating economic metrics..."):
                        metrics = calculate_economic_metrics(annual_savings_kwh, capex, annual_maintenance, kwh_price, project_years, discount_rate/100)
                        st.session_state.economic_parameters = metrics
                        
                        # Display results
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            payback_val = f"{metrics['payback_period']:.2f} Years" if metrics['payback_period'] is not None and metrics['payback_period'] < project_years else "Never"
                            st.metric("Payback Period", payback_val)
                        with col2:
                            roi_val = f"{metrics['roi']:.2f}%" if metrics['roi'] is not None else "N/A"
                            st.metric("Return on Investment (ROI)", roi_val)
                        with col3: 
                            st.metric("Net Present Value (NPV)", f"Rp {metrics['npv']:,.0f}")
                        with col4:
                            irr_val = f"{metrics['irr']:.2f}%" if metrics['irr'] is not None else "N/A"
                            st.metric("Internal Rate of Return (IRR)", irr_val)
                        
                        # Cash flow chart
                        st.markdown("---")
                        cash_flow_df = pd.DataFrame({
                            'Year': list(range(project_years + 1)), 
                            'Cumulative Cash Flow (Rp)': metrics['cumulative_cash_flow']
                        })
                        
                        # Bar chart
                        fig_bar = go.Figure(go.Bar(
                            x=cash_flow_df['Year'], 
                            y=cash_flow_df['Cumulative Cash Flow (Rp)'], 
                            marker_color=['crimson' if c < 0 else 'mediumseagreen' for c in cash_flow_df['Cumulative Cash Flow (Rp)']],
                            name='Cumulative Cash Flow'
                        ))
                        fig_bar.add_hline(y=0, line_width=2, line_dash="dash", line_color="black")
                        fig_bar.update_layout(
                            title='Cumulative Cash Flow Over Project Lifetime', 
                            xaxis_title='Year', yaxis_title='Cumulative Cash Flow (Rp)', 
                            height=450, showlegend=False
                        )
                        st.plotly_chart(fig_bar, use_container_width=True, key="economic_cashflow_bar")
                        
                        # Line chart
                        fig_line = px.line(cash_flow_df, x='Year', y='Cumulative Cash Flow (Rp)',
                                           title='Cumulative Cash Flow Trend',
                                           markers=True)
                        fig_line.add_hline(y=0, line_width=2, line_dash="dash", line_color="red")
                        st.plotly_chart(fig_line, use_container_width=True, key="economic_cashflow_line")
        
        with econ_tab2:
            st.subheader("Manual Project Calculator")
            st.info("Use this calculator to evaluate any energy efficiency project with your own parameters.")
            
            with st.form(key="manual_economic_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("<h5>Energy Savings</h5>", unsafe_allow_html=True)
                    manual_annual_savings_kwh = st.number_input("Annual Energy Savings (kWh/year)", min_value=0.0, value=1000.0, step=100.0)
                    manual_electricity_rate = st.number_input("Electricity Rate (Rp/kWh)", min_value=0, value=kwh_price, step=1)
                    
                    st.markdown("<h5>Investment</h5>", unsafe_allow_html=True)
                    manual_capex = st.number_input("Initial Investment (CAPEX) (Rp)", min_value=0, value=50000000, step=1000000, key="manual_capex")
                
                with col2:
                    st.markdown("<h5>Operating Parameters</h5>", unsafe_allow_html=True)
                    manual_annual_maintenance = st.number_input("Annual Maintenance Cost (OPEX) (Rp)", min_value=0, value=1000000, step=100000, key="manual_opex")
                    manual_project_years = st.number_input("Project Lifetime (Years)", min_value=1, max_value=50, value=10, key="manual_years")
                    manual_discount_rate = st.slider("Discount Rate (%)", min_value=0.0, max_value=25.0, value=8.0, step=0.5, key="manual_discount")
                
                manual_submit = st.form_submit_button(label="Calculate Manual Project")
            
            if manual_submit:
                with st.spinner("Calculating manual project metrics..."):
                    manual_metrics = calculate_economic_metrics(
                        manual_annual_savings_kwh, manual_capex, manual_annual_maintenance, 
                        manual_electricity_rate, manual_project_years, manual_discount_rate/100
                    )
                    
                    st.markdown("### Manual Calculation Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        payback_val = f"{manual_metrics['payback_period']:.2f} Yrs" if manual_metrics['payback_period'] is not None and manual_metrics['payback_period'] < manual_project_years else "Never"
                        st.metric("Payback Period", payback_val)
                    with col2:
                        roi_val = f"{manual_metrics['roi']:.2f}%" if manual_metrics['roi'] is not None else "N/A"
                        st.metric("ROI", roi_val)
                    with col3: 
                        st.metric("NPV", f"Rp {manual_metrics['npv']:,.0f}")
                    with col4:
                        irr_val = f"{manual_metrics['irr']:.2f}%" if manual_metrics['irr'] is not None else "N/A"
                        st.metric("IRR", irr_val)
                    
                    # Manual cash flow visualization
                    manual_cash_flow_df = pd.DataFrame({
                        'Year': list(range(manual_project_years + 1)), 
                        'Cumulative Cash Flow (Rp)': manual_metrics['cumulative_cash_flow']
                    })
                    
                    fig_manual = go.Figure(go.Bar(
                        x=manual_cash_flow_df['Year'], 
                        y=manual_cash_flow_df['Cumulative Cash Flow (Rp)'], 
                        marker_color=['crimson' if c < 0 else 'mediumseagreen' for c in manual_cash_flow_df['Cumulative Cash Flow (Rp)']],
                        name='Cumulative Cash Flow'
                    ))
                    fig_manual.add_hline(y=0, line_width=2, line_dash="dash", line_color="black")
                    fig_manual.update_layout(
                        title='Manual Project - Cumulative Cash Flow', 
                        xaxis_title='Year', yaxis_title='Cumulative Cash Flow (Rp)', 
                        height=450, showlegend=False
                    )
                    st.plotly_chart(fig_manual, use_container_width=True, key="manual_cashflow")

    with tab8:
        st.subheader("📊 Methodology & Technical Details")
        st.markdown(
            r"""
            ### 🔬 Analysis Methodology
            
            This dashboard implements a comprehensive energy consumption analysis system using machine learning and statistical methods.
            
            #### 📁 Data Processing & Feature Engineering
            1.  **Time-Series Aware Splitting**: Data is split chronologically based on a user-defined ratio to prevent data leakage and respect the temporal nature of the data.
            2.  **Advanced Feature Creation**: In addition to time-based features (hour, day, etc.), the model uses:
                -   **Lag Features**: Consumption from the previous 3 hours (`Lag_1`, `Lag_2`, `Lag_3`).
                -   **Rolling Window Features**: 3-hour and 24-hour moving averages to capture short-term and daily trends.
            3.  **Multicollinearity Removal**: Features with correlation > 0.7 are automatically removed to prevent model overfitting.
            
            #### 🧠 Machine Learning Models
            -   **Random Forest & Gradient Boosting**: Ensemble tree-based models that are robust and effective for tabular data.
            -   **LSTM Neural Network**: A recurrent neural network specialized for capturing complex temporal dependencies in time series data.
            
            #### 🚨 Enhanced Anomaly Detection Method
            This dashboard focuses on **statistical anomaly detection** using trained machine learning models:

            **Statistical Anomaly Detection Process:**
            1.  **Model Training**: Train multiple ML models on historical consumption data
            2.  **Prediction Generation**: Use the best-performing model to predict "normal" consumption
            3.  **Statistical Threshold**: Calculate prediction error standard deviation from validation set
            4.  **Anomaly Identification**: Flag consumption exceeding prediction + (2σ) as anomalies
            5.  **Savings Calculation**: Sum of all anomalous consumption represents potential savings
            
            **Mathematical Formula:**
            $
            \text{Threshold} = \text{Predicted Value} + (2 \times \sigma_{\text{validation error}})
            $
            $
            \text{Anomaly} = \max(0, \text{Actual Consumption} - \text{Threshold})
            $
            
            #### 📈 Central Limit Theorem (CLT) Analysis
            The dashboard includes CLT visualization to demonstrate statistical properties of energy consumption:
            - **Population Distribution**: Shows the actual distribution of energy consumption data
            - **Sample Means Distribution**: Shows how sample means (n=30) approximate normal distribution
            - **Statistical Validation**: Verifies that sample means follow CLT predictions
            
            #### 💰 Economic Analysis
            Two calculation methods are provided:
            1. **Model-Based**: Uses anomaly detection results to estimate savings
            2. **Manual Calculator**: Allows custom input for any energy efficiency project
            
            **Key Financial Metrics:**
            - **Payback Period**: Time to recover initial investment
            - **ROI**: Return on Investment over project lifetime  
            - **NPV**: Net Present Value considering discount rate
            - **IRR**: Internal Rate of Return
            
            #### 🎯 Device Operating Hours
            """)
        
        hours_df = pd.DataFrame([
            {'Device Type': k.upper(), 'Operating Hours': f"{v[0]:02d}:00 - {v[1]:02d}:00"} 
            for k, v in DEVICE_OPERATING_HOURS.items()
        ])
        st.dataframe(hours_df, use_container_width=True)
        ### UPDATE ###
        csv = to_csv(hours_df)
        st.download_button("Download Operating Hours as CSV", csv, "device_operating_hours.csv", "text/csv", key='download-operating-hours')
        
        st.markdown("""
            #### 🔍 Key Features
            - **Multi-level Analysis**: Individual devices, device types, floors, and buildings
            - **Real-time Anomaly Detection**: Identifies consumption patterns that deviate from normal
            - **Statistical Validation**: Uses CLT to verify data quality and distribution properties  
            - **Economic Feasibility**: Complete financial analysis with multiple metrics
            - **Interactive Visualizations**: Dynamic charts for pattern exploration
            - **Scalable Processing**: Handles multiple devices with bulk training capabilities
            """)

else:
    st.info("""
    ## 🚀 Welcome to the Enhanced Energy Consumption Analysis Dashboard
    
    This dashboard provides a comprehensive analysis of energy consumption data using machine learning and statistical methods, with a focus on anomaly detection.
    
    ### 📁 Getting Started
    1.  **Prepare Your Data**: Organize your CSV files inside a ZIP folder. The structure must be `building_name/floor_name/device_type/your_data.csv` or `building_name/device_type/your_data.csv`.
    2.  **Upload ZIP File**: Use the sidebar to upload your single, structured ZIP file.
    3.  **Required Columns**: Ensure your CSV has `id_time` (as the index) and `Konsumsi Energi` columns.
    4.  **Data Quality**: Each file should have at least the minimum number of records (adjustable in sidebar).
    
    ### 🎯 What You'll Get
    - **Multi-level Analysis**: Insights from individual devices, device types, floors, and entire buildings.
    - **ML-Powered Anomaly Detection**: Advanced statistical methods to identify consumption anomalies.
    - **Central Limit Theorem Analysis**: Statistical validation of your energy consumption data.
    - **Enhanced Visualizations**: Correlation matrices, performance heatmaps, and pattern analysis.
    - **Economic Feasibility**: Both model-based and manual project calculators.
    - **Interactive Dashboards**: Explore patterns and trends with dynamic, responsive charts.
    
    ### 📊 Supported Device Types
    - **AHU** (Air Handling Units) - Operating hours: 08:00-16:00
    - **SDP** (Sub Distribution Panels) - Operating hours: 00:00-23:00
    - **LIFT** (Elevators) - Operating hours: 07:00-20:00
    - **CHILLER** (Cooling Systems) - Operating hours: 08:00-17:00
    
    ### 🆕 New Features
    - **Adjustable minimum data points** - Customize quality requirements
    - **Complete correlation analysis** - View all feature relationships
    - **Training summary with heatmaps** - Comprehensive model performance overview
    - **Enhanced anomaly detection** - Detailed statistical analysis with visualizations
    - **Central Limit Theorem validation** - Statistical quality assurance
    - **Manual project calculator** - Evaluate any energy efficiency project
    - **Improved contrast and visibility** - Better user experience
    
    Upload your ZIP file using the sidebar to begin the analysis!
    """)