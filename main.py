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

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Energy Consumption Analysis Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
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
    'Soil Temperature', 'Sunshine Duration', 'UV Index', 'Direct Radiation',
    'Current', 'Power Factor'
]

DEVICE_OPERATING_HOURS = {
    'ahu': (8, 16), 'sdp': (0, 23), 'lift': (7, 20), 'chiller': (8, 17)
}
CORE_BUSINESS_HOURS = (9, 17)
DEFAULT_OPERATING_HOURS = (8, 17)
MINIMUM_ROWS = 500

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}

# Helper Functions
@st.cache_data
def load_and_process_data(uploaded_file, building_name, device_type, floor_name=""):
    """Load and process uploaded CSV data"""
    try:
        df = pd.read_csv(uploaded_file, index_col='id_time', parse_dates=True)
        
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
        
        # Add time-based features
        df['Konsumsi_Energi_Lag_1'] = df[TARGET_VARIABLE].shift(1)
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # Add holidays
        years = df.index.year.unique()
        id_holidays = holidays.Indonesia(years=years)
        df['isHoliday'] = df.index.isin(id_holidays).astype(int)
        
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['week_of_year'] = df.index.isocalendar().week.astype(int)
        df['month_of_year'] = df.index.month
        
        df.dropna(inplace=True)
        df = df[df[TARGET_VARIABLE] > 0].copy()
        
        return df, None
    except Exception as e:
        return None, str(e)

def train_models(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train multiple models and return results"""
    results = {}
    
    # Random Forest
    with st.spinner("Training Random Forest..."):
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
    
    # Gradient Boosting
    with st.spinner("Training Gradient Boosting..."):
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
    
    # LSTM
    with st.spinner("Training LSTM..."):
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
    
    return patterns

def generate_recommendations_analysis(patterns, device_identifier):
    """Generate energy saving recommendations"""
    recommendations = []
    
    peak_hour = patterns['peak_hour']
    off_peak_hour = patterns['off_peak_hour']
    peak_consumption = patterns['peak_consumption']
    off_peak_consumption = patterns['off_peak_consumption']
    
    potential_savings = (peak_consumption - off_peak_consumption) / peak_consumption * 100
    
    recommendations.append(f"🔴 Peak consumption occurs at {peak_hour}:00 with {peak_consumption:.2f} Wh")
    recommendations.append(f"🟢 Lowest consumption at {off_peak_hour}:00 with {off_peak_consumption:.2f} Wh")
    recommendations.append(f"💡 Potential savings: {potential_savings:.1f}% by shifting loads from peak to off-peak hours")
    
    if peak_hour >= 9 and peak_hour <= 17:
        recommendations.append("⏰ Consider load balancing during business hours")
    
    if patterns['daily'][5] > patterns['daily'][0]:  # Weekend vs weekday
        recommendations.append("📅 Weekend consumption is higher than weekdays - review weekend operations")
    
    return recommendations, potential_savings

# Sidebar
st.sidebar.title("📊 Dashboard Controls")

# File upload section
st.sidebar.header("📁 Data Upload")
uploaded_files = st.sidebar.file_uploader(
    "Upload CSV files",
    type=['csv'],
    accept_multiple_files=True,
    help="Upload energy consumption CSV files with 'id_time' column"
)

# Building and device configuration
st.sidebar.header("🏢 Device Configuration")
building_name = st.sidebar.text_input("Building Name", value="Building-A")
device_type = st.sidebar.selectbox(
    "Device Type",
    options=['ahu', 'sdp', 'lift', 'chiller', 'other'],
    index=0
)
floor_name = st.sidebar.text_input("Floor/Location", value="Floor-1")

# Analysis parameters
st.sidebar.header("⚙️ Analysis Parameters")
test_size = st.sidebar.slider("Test Size (%)", min_value=10, max_value=40, value=30)
random_state = st.sidebar.number_input("Random State", value=42)
min_rows = st.sidebar.number_input("Minimum Rows", value=MINIMUM_ROWS)

# Main content area
if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Data Overview", 
        "🤖 Model Training", 
        "📊 Consumption Analysis", 
        "💡 Recommendations", 
        "📋 Reports"
    ])
    
    # Process each uploaded file
    for i, uploaded_file in enumerate(uploaded_files):
        file_name = uploaded_file.name
        device_id = f"{building_name}-{device_type}-{floor_name}-{i+1}"
        
        # Load and process data
        df, error = load_and_process_data(uploaded_file, building_name, device_type, floor_name)
        
        if error:
            st.error(f"❌ Error processing {file_name}: {error}")
            continue
            
        if len(df) < min_rows:
            st.warning(f"⚠️ {file_name}: Insufficient data ({len(df)} rows, minimum {min_rows} required)")
            continue
        
        # Store in session state
        st.session_state.uploaded_files[device_id] = {
            'df': df,
            'file_name': file_name,
            'device_id': device_id
        }
        
        with tab1:
            st.subheader(f"📈 Data Overview - {file_name}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Date Range", f"{df.index.min().date()} to {df.index.max().date()}")
            with col3:
                st.metric("Avg Consumption", f"{df[TARGET_VARIABLE].mean():.2f} Wh")
            with col4:
                st.metric("Peak Consumption", f"{df[TARGET_VARIABLE].max():.2f} Wh")
            
            # Time series plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df[TARGET_VARIABLE],
                mode='lines',
                name='Energy Consumption',
                line=dict(color='#1f77b4', width=1)
            ))
            fig.update_layout(
                title=f"Energy Consumption Over Time - {device_id}",
                xaxis_title="Time",
                yaxis_title="Consumption (Wh)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Data statistics
            st.subheader("📊 Data Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Consumption Statistics:**")
                st.dataframe(df[TARGET_VARIABLE].describe())
            
            with col2:
                st.write("**Data Info:**")
                buffer = io.StringIO()
                df.info(buf=buffer)
                info_str = buffer.getvalue()
                st.text(info_str)
        
        with tab2:
            st.subheader(f"🤖 Model Training - {file_name}")
            
            if st.button(f"Train Models for {device_id}", key=f"train_{i}"):
                # Feature selection
                potential_features = [col for col in df.columns if col not in [TARGET_VARIABLE, 'Konsumsi_Energi_Lag_1']]
                feature_corr_matrix = df[potential_features].corr().abs()
                upper_tri = feature_corr_matrix.where(
                    np.triu(np.ones(feature_corr_matrix.shape), k=1).astype(bool)
                )
                to_drop = [column for column in upper_tri.columns if any(upper_tri[column] >= 0.7)]
                
                independent_features = [f for f in potential_features if f not in to_drop]
                features_for_model = sorted(list(set(['Konsumsi_Energi_Lag_1'] + independent_features)))
                
                st.info(f"Features selected: {len(features_for_model)} (removed {len(to_drop)} due to multicollinearity)")
                
                # Prepare data
                X = df[features_for_model]
                y = df[TARGET_VARIABLE]
                
                X_train, X_temp, y_train, y_temp = train_test_split(
                    X, y, test_size=test_size/100, random_state=random_state
                )
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=0.5, random_state=random_state
                )
                
                # Train models
                model_results = train_models(X_train, y_train, X_val, y_val, X_test, y_test)
                
                # Display results
                st.success("✅ Model training completed!")
                
                # Model comparison
                comparison_data = []
                for model_name, result in model_results.items():
                    comparison_data.append({
                        'Model': model_name,
                        'MAE': result['metrics']['mae'],
                        'RMSE': result['metrics']['rmse'],
                        'R²': result['metrics']['r2']
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df)
                
                # Best model
                best_model = min(model_results, key=lambda k: model_results[k]['metrics']['mae'])
                st.success(f"🏆 Best model: {best_model} (MAE: {model_results[best_model]['metrics']['mae']:.2f})")
                
                # Store results
                st.session_state.analysis_results[device_id] = {
                    'model_results': model_results,
                    'best_model': best_model,
                    'features': features_for_model,
                    'test_data': {'X_test': X_test, 'y_test': y_test}
                }
                
                # Predictions visualization
                best_predictions = model_results[best_model]['predictions']
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(y_test))),
                    y=y_test.values,
                    mode='lines',
                    name='Actual',
                    line=dict(color='blue')
                ))
                fig.add_trace(go.Scatter(
                    x=list(range(len(best_predictions))),
                    y=best_predictions,
                    mode='lines',
                    name='Predicted',
                    line=dict(color='red', dash='dash')
                ))
                fig.update_layout(
                    title=f"Actual vs Predicted - {best_model}",
                    xaxis_title="Sample Index",
                    yaxis_title="Consumption (Wh)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader(f"📊 Consumption Analysis - {file_name}")
            
            # Analyze patterns
            patterns = analyze_consumption_patterns(df, device_id)
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Hourly Pattern', 'Daily Pattern', 'Monthly Pattern', 'Distribution'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Hourly pattern
            fig.add_trace(
                go.Scatter(x=list(patterns['hourly'].index), y=patterns['hourly'].values,
                          mode='lines+markers', name='Hourly Avg'),
                row=1, col=1
            )
            
            # Daily pattern
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            fig.add_trace(
                go.Bar(x=days, y=patterns['daily'].values, name='Daily Avg'),
                row=1, col=2
            )
            
            # Monthly pattern
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            fig.add_trace(
                go.Scatter(x=months[:len(patterns['monthly'])], 
                          y=patterns['monthly'].values,
                          mode='lines+markers', name='Monthly Avg'),
                row=2, col=1
            )
            
            # Distribution
            fig.add_trace(
                go.Histogram(x=df[TARGET_VARIABLE], nbinsx=30, name='Distribution'),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Peak Hour", f"{patterns['peak_hour']}:00")
            with col2:
                st.metric("Peak Consumption", f"{patterns['peak_consumption']:.2f} Wh")
            with col3:
                st.metric("Off-Peak Hour", f"{patterns['off_peak_hour']}:00")
            with col4:
                st.metric("Off-Peak Consumption", f"{patterns['off_peak_consumption']:.2f} Wh")
        
        with tab4:
            st.subheader(f"💡 Recommendations - {file_name}")
            
            patterns = analyze_consumption_patterns(df, device_id)
            recommendations, potential_savings = generate_recommendations_analysis(patterns, device_id)
            
            # Savings potential
            st.info(f"💰 Potential Energy Savings: {potential_savings:.1f}%")
            
            # Recommendations list
            st.write("**📋 Recommendations:**")
            for rec in recommendations:
                st.write(f"• {rec}")
            
            # Operating hours analysis
            device_type_key = device_type.lower()
            if device_type_key in DEVICE_OPERATING_HOURS:
                operating_hours = DEVICE_OPERATING_HOURS[device_type_key]
                st.write(f"**⏰ Device Operating Hours:** {operating_hours[0]}:00 - {operating_hours[1]}:00")
                
                # Filter data for operating hours
                operating_data = df[
                    (df.index.hour >= operating_hours[0]) & 
                    (df.index.hour <= operating_hours[1]) &
                    (df.index.dayofweek < 5)  # Weekdays only
                ]
                
                if not operating_data.empty:
                    total_consumption = operating_data[TARGET_VARIABLE].sum() / 1000  # Convert to kWh
                    avg_consumption = operating_data[TARGET_VARIABLE].mean()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Consumption (Operating Hours)", f"{total_consumption:.2f} kWh")
                    with col2:
                        st.metric("Average Consumption", f"{avg_consumption:.2f} Wh")
            
            # Cost estimation (example rates)
            st.subheader("💸 Cost Analysis")
            electricity_rate = st.number_input("Electricity Rate (per kWh)", value=1500.0, step=100.0)
            
            total_kwh = df[TARGET_VARIABLE].sum() / 1000
            total_cost = total_kwh * electricity_rate
            potential_savings_cost = total_cost * (potential_savings / 100)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Consumption", f"{total_kwh:.2f} kWh")
            with col2:
                st.metric("Total Cost", f"Rp {total_cost:,.0f}")
            with col3:
                st.metric("Potential Savings", f"Rp {potential_savings_cost:,.0f}")
    
    with tab5:
        st.subheader("📋 Summary Report")
        
        if st.session_state.uploaded_files:
            # Aggregate statistics
            total_devices = len(st.session_state.uploaded_files)
            total_consumption = 0
            total_savings_potential = 0
            
            report_data = []
            
            for device_id, data in st.session_state.uploaded_files.items():
                df = data['df']
                patterns = analyze_consumption_patterns(df, device_id)
                _, savings_potential = generate_recommendations_analysis(patterns, device_id)
                
                device_consumption = df[TARGET_VARIABLE].sum() / 1000  # kWh
                total_consumption += device_consumption
                total_savings_potential += (device_consumption * savings_potential / 100)
                
                report_data.append({
                    'Device ID': device_id,
                    'File Name': data['file_name'],
                    'Records': len(df),
                    'Total Consumption (kWh)': device_consumption,
                    'Peak Hour': patterns['peak_hour'],
                    'Savings Potential (%)': savings_potential,
                    'Potential Savings (kWh)': device_consumption * savings_potential / 100
                })
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Devices", total_devices)
            with col2:
                st.metric("Total Consumption", f"{total_consumption:.2f} kWh")
            with col3:
                st.metric("Avg Savings Potential", f"{(total_savings_potential/total_consumption)*100:.1f}%")
            with col4:
                st.metric("Total Potential Savings", f"{total_savings_potential:.2f} kWh")
            
            # Detailed report table
            st.subheader("📊 Detailed Report")
            report_df = pd.DataFrame(report_data)
            st.dataframe(report_df, use_container_width=True)
            
            # Download report
            csv = report_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Report as CSV",
                data=csv,
                file_name=f"energy_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Visualization
            if len(report_df) > 1:
                fig = px.bar(
                    report_df, 
                    x='Device ID', 
                    y='Total Consumption (kWh)',
                    title='Energy Consumption by Device',
                    color='Savings Potential (%)',
                    color_continuous_scale='RdYlGn_r'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("📁 Upload files to generate reports")

else:
    st.info("👆 Please upload CSV files using the sidebar to start the analysis")
    
    # Show example data format
    st.subheader("📋 Expected Data Format")
    st.write("Your CSV files should contain the following columns:")
    
    example_data = {
        'id_time': ['2024-01-01 00:00:00', '2024-01-01 01:00:00', '2024-01-01 02:00:00'],
        'Konsumsi Energi': [1500, 1200, 1800],
        'Temperature': [25.5, 24.8, 26.2],
        'Relative Humidity': [65, 68, 62],
        'id_i1': [5.2, 4.8, 6.1],
        'id_i2': [5.0, 4.6, 5.9],
        'id_i3': [4.8, 4.4, 5.7]
    }
    
    example_df = pd.DataFrame(example_data)
    st.dataframe(example_df)
    
    st.write("**Required columns:**")
    st.write("• `id_time` - Timestamp column (will be used as index)")
    st.write("• `Konsumsi Energi` - Energy consumption values")
    st.write("• Weather data columns (Temperature, Humidity, etc.)")
    st.write("• Current measurement columns (id_i1, id_i2, id_i3) - optional")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    ⚡ Energy Consumption Analysis Dashboard | Built with Streamlit
</div>
""", unsafe_allow_html=True)