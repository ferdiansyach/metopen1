# ==============================================================================
# DASHBOARD SKRIPSI: SISTEM PREDIKSI ENERGI (HYBRID AI)
# Version: 3.0 (Production Ready)
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="Energy AI Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cek Ketersediaan Library ML
try:
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    # Fix Bug Keras di Streamlit
    tf.config.run_functions_eagerly(True)
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    st.error(f"⚠️ Library ML tidak ditemukan: {e}. Beberapa fitur akan dimatikan.")

class Config:
    """Sentralisasi Konfigurasi & Styling"""
    # Palet Warna Professional (Colorblind Friendly)
    COLORS = {
        'primary': '#2c3e50',      # Dark Blue
        'accent': '#3498db',       # Bright Blue
        'success': '#27ae60',      # Green
        'warning': '#f39c12',      # Orange
        'danger': '#c0392b',       # Red
        'background': '#ecf0f1',   # Light Gray
        'chart_actual': '#95a5a6', # Gray (untuk background chart)
        'chart_pred': '#e74c3c'    # Red (untuk highlight prediksi)
    }
    
    # Parameter Default
    DEFAULT_LOOKBACK = 24
    DEFAULT_EPOCHS = 15

def load_custom_css():
    st.markdown(f"""
        <style>
        .main {{ background-color: #f8f9fa; }}
        .stMetric {{
            background-color: white;
            border: 1px solid #e0e0e0;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        h1, h2, h3 {{ color: {Config.COLORS['primary']}; font-family: 'Segoe UI', sans-serif; }}
        .info-box {{
            background-color: #e8f4f8;
            border-left: 4px solid {Config.COLORS['accent']};
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 15px;
        }}
        </style>
    """, unsafe_allow_html=True)

load_custom_css()

# =====================================================================
# 2. DATA MANAGEMENT (ROBUST LOADING)
# =====================================================================

@st.cache_data
def generate_demo_data(days=365):
    """Fallback: Generate data dummy jika file asli tidak ada"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=days*24, freq='H')
    
    # Pola: Sinusoidal Harian + Faktor Mingguan + Trend + Noise
    t = np.linspace(0, days*2*np.pi, days*24)
    daily_seasonality = 100 * np.sin(t * days) # Harian
    
    # Weekly pattern (Weekend lebih rendah)
    day_factor = np.array([1.0 if d < 5 else 0.7 for d in dates.dayofweek])
    
    base_load = 500
    trend = np.linspace(0, 50, len(dates))
    noise = np.random.normal(0, 20, len(dates))
    
    consumption = (base_load + daily_seasonality * day_factor + trend + noise)
    consumption = np.maximum(consumption, 50) # Clip min value
    
    # Tambahkan anomali (Spikes)
    anom_idx = np.random.choice(len(consumption), size=int(len(consumption)*0.01), replace=False)
    consumption[anom_idx] += 300
    
    df = pd.DataFrame({'timestamp': dates, 'energy_consumption': consumption})
    
    # Feature Engineering
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_name'] = df['timestamp'].dt.day_name()
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    return df

def load_data(uploaded_file=None):
    """Fungsi loading data pintar dengan validasi"""
    df = None
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            # Coba deteksi kolom waktu
            time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
            energy_cols = [col for col in df.columns if 'kwh' in col.lower() or 'energy' in col.lower()]
            
            if time_cols and energy_cols:
                df = df.rename(columns={time_cols[0]: 'timestamp', energy_cols[0]: 'energy_consumption'})
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                st.error("Format CSV tidak dikenali. Pastikan ada kolom waktu dan energi.")
                return None
        except Exception as e:
            st.error(f"Error parsing file: {e}")
            return None
    else:
        # Coba load dari folder hasil EDA jika ada
        try:
            df = pd.read_csv('hasil_eda_skripsi/data_eda_bersih_siap_pakai.csv')
            df['timestamp'] = pd.to_datetime(df['id_time'])
            df['energy_consumption'] = df['energy_consumed_kwh']
        except:
            df = generate_demo_data() # Fallback terakhir
            
    # Pastikan fitur waktu ada
    if 'hour' not in df.columns:
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_name'] = df['timestamp'].dt.day_name()
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
    return df

# Initialize Session State
if 'df' not in st.session_state:
    st.session_state.df = load_data()
    st.session_state.models = {}
    st.session_state.training_results = None

df = st.session_state.df

# =====================================================================
# 3. ENGINE AI (BACKEND LOGIC)
# =====================================================================

class AIEngine:
    def prepare_tensors(self, df, lookback):
        """Mengubah data tabel menjadi Tensor 3D untuk LSTM"""
        data = df['energy_consumption'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
            
        X, y = np.array(X), np.array(y)
        
        # Reshape for LSTM [Samples, TimeSteps, Features]
        X_lstm = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Split 80:20
        train_size = int(len(X) * 0.8)
        return (X_lstm[:train_size], y[:train_size]), (X_lstm[train_size:], y[train_size:]), scaler

    def train_hybrid_model(self, X_train, y_train, X_test, y_test, scaler, epochs):
        """Pipeline Training: LSTM + XGBoost Residual"""
        results = {}
        
        # 1. Train LSTM (Main Trend)
        lstm = keras.Sequential([
            keras.layers.LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(50, return_sequences=False),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1)
        ])
        lstm.compile(optimizer='adam', loss='mse')
        
        # Progress bar UI
        prog_bar = st.progress(0, text="🧠 Melatih Neural Network (LSTM)...")
        lstm.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
        prog_bar.progress(50, text="🌲 Melatih XGBoost Residual...")
        
        # 2. Train XGBoost (Residual Correction)
        lstm_pred_train = lstm.predict(X_train, verbose=0).flatten()
        residuals = y_train - lstm_pred_train # Hitung error LSTM
        
        # Flatten X untuk XGBoost
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        
        xgb_resid = GradientBoostingRegressor(n_estimators=100, max_depth=5)
        xgb_resid.fit(X_train_flat, residuals)
        
        prog_bar.progress(80, text="📊 Mengevaluasi Model...")
        
        # 3. Evaluasi pada Data Test
        # Prediksi LSTM
        pred_lstm = lstm.predict(X_test, verbose=0).flatten()
        # Prediksi Residual
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        pred_error = xgb_resid.predict(X_test_flat)
        # Hybrid
        pred_hybrid = pred_lstm + pred_error
        
        # Inverse Scaling
        y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_hybrid = scaler.inverse_transform(pred_hybrid.reshape(-1, 1)).flatten()
        y_pred_hybrid = np.maximum(y_pred_hybrid, 0) # Clip negative
        
        # Calculate Metrics
        mape = np.mean(np.abs((y_true - y_pred_hybrid) / (y_true + 1e-10))) * 100
        rmse = np.sqrt(mean_squared_error(y_true, y_pred_hybrid))
        r2 = r2_score(y_true, y_pred_hybrid)
        
        results['Hybrid'] = {
            'MAPE': mape, 'RMSE': rmse, 'R2': r2,
            'Actual': y_true, 'Predicted': y_pred_hybrid
        }
        
        # Simpan ke session
        st.session_state.models['LSTM'] = lstm
        st.session_state.models['XGB_Resid'] = xgb_resid
        st.session_state.scaler = scaler
        
        prog_bar.progress(100, text="✅ Selesai!")
        return results

engine = AIEngine() if ML_AVAILABLE else None

# =====================================================================
# 4. UI SIDEBAR NAVIGASI
# =====================================================================

with st.sidebar:
    st.title("⚡ Energy AI")
    st.markdown(f"<div style='color:{Config.COLORS['accent']}'><b>Sistem Cerdas Efisiensi Energi</b></div>", unsafe_allow_html=True)
    st.markdown("---")
    
    menu = st.radio("NAVIGASI:", 
        ["🏠 Ringkasan Eksekutif", 
         "📊 Analisis Data (EDA)", 
         "🤖 Training & Evaluasi", 
         "🔮 Simulasi Prediksi"])
    
    st.markdown("---")
    
    # File Uploader Persistent
    uploaded_file = st.file_uploader("📂 Upload Data Baru", type=['csv'], help="Format: CSV dengan kolom timestamp dan energy_consumed_kwh")
    if uploaded_file:
        st.session_state.df = load_data(uploaded_file)
        st.success("Data diperbarui!")
        st.rerun()

# =====================================================================
# HALAMAN 1: EXECUTIVE SUMMARY
# =====================================================================
if menu == "🏠 Ringkasan Eksekutif":
    st.title("📊 Ringkasan Kinerja Energi")
    st.markdown("Gambaran umum konsumsi energi dan indikator kinerja utama (KPI).")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    total_consum = df['energy_consumption'].sum()
    avg_consum = df['energy_consumption'].mean()
    max_consum = df['energy_consumption'].max()
    
    # Calculate Trend (Last 7 days vs Prev 7 days)
    if len(df) > 168:
        curr = df['energy_consumption'].iloc[-168:].sum()
        prev = df['energy_consumption'].iloc[-336:-168].sum()
        delta = ((curr - prev) / prev) * 100
    else:
        delta = 0

    col1.metric("Total Konsumsi", f"{total_consum/1000:,.2f} MWh", f"{delta:+.1f}% (7d Trend)")
    col2.metric("Rata-rata Beban", f"{avg_consum:,.2f} kWh", "Per Jam")
    col3.metric("Beban Puncak", f"{max_consum:,.2f} kWh", "Max Load", delta_color="inverse")
    col4.metric("Efisiensi Data", f"{len(df):,} Rows", "Dataset Size")
    
    st.markdown("---")
    
    # Layout Grafik Atas
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("📈 Tren Konsumsi Energi (Time Series)")
        # Downsample untuk grafik agar ringan (per hari)
        daily_df = df.set_index('timestamp').resample('D')['energy_consumption'].sum().reset_index()
        fig = px.line(daily_df, x='timestamp', y='energy_consumption', template='plotly_white')
        fig.update_traces(line_color=Config.COLORS['accent'], line_width=2.5)
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.subheader("🕒 Pola Jam Puncak")
        hourly = df.groupby('hour')['energy_consumption'].mean().reset_index()
        peak_hour = hourly.loc[hourly['energy_consumption'].idxmax(), 'hour']
        
        fig2 = px.bar(hourly, x='hour', y='energy_consumption', 
                      color='energy_consumption', color_continuous_scale='Blues')
        fig2.add_vline(x=peak_hour, line_dash="dash", line_color="red")
        fig2.update_layout(height=350, showlegend=False, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig2, use_container_width=True)
        
        st.info(f"💡 **Insight:** Beban puncak rata-rata terjadi pada pukul **{peak_hour}:00**. Pertimbangkan load shifting pada jam ini.")

# =====================================================================
# HALAMAN 2: EDA
# =====================================================================
elif menu == "📊 Analisis Data (EDA)":
    st.title("🔍 Exploratory Data Analysis")
    
    # Filter Interaktif
    st.sidebar.markdown("### 🎛️ Filter Data")
    date_range = st.sidebar.date_input("Rentang Tanggal", [df['timestamp'].min(), df['timestamp'].max()])
    
    # Apply Filter
    if len(date_range) == 2:
        mask = (df['timestamp'].dt.date >= date_range[0]) & (df['timestamp'].dt.date <= date_range[1])
        df_eda = df.loc[mask]
    else:
        df_eda = df
        
    tab1, tab2, tab3 = st.tabs(["📉 Distribusi", "🔥 Heatmap Pola", "🚨 Anomali"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Histogram Konsumsi")
            fig = px.histogram(df_eda, x="energy_consumption", nbins=50, color_discrete_sequence=[Config.COLORS['success']])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("##### Boxplot Hari Kerja vs Libur")
            fig = px.box(df_eda, x="is_weekend", y="energy_consumption", color="is_weekend",
                         color_discrete_map={0: Config.COLORS['accent'], 1: Config.COLORS['warning']},
                         labels={'is_weekend': '0=Weekday, 1=Weekend'})
            st.plotly_chart(fig, use_container_width=True)
            
    with tab2:
        st.markdown("##### Heatmap Intensitas Energi (Jam vs Hari)")
        heatmap_data = df_eda.groupby(['day_name', 'hour'])['energy_consumption'].mean().reset_index()
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        fig = px.density_heatmap(heatmap_data, x='hour', y='day_name', z='energy_consumption',
                                 category_orders={'day_name': days_order},
                                 color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
        
    with tab3:
        st.markdown("##### Deteksi Anomali (Metode IQR)")
        Q1 = df_eda['energy_consumption'].quantile(0.25)
        Q3 = df_eda['energy_consumption'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        
        anomalies = df_eda[df_eda['energy_consumption'] > upper_bound]
        
        st.metric("Jumlah Data Anomali", f"{len(anomalies)} Records", f"Threshold > {upper_bound:.0f} kWh", delta_color="inverse")
        
        fig = px.scatter(df_eda, x='timestamp', y='energy_consumption', color=df_eda['energy_consumption'] > upper_bound,
                         color_discrete_map={True: 'red', False: 'blue'}, title="Sebaran Anomali Sepanjang Waktu")
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Lihat Data Tabel Anomali"):
            st.dataframe(anomalies, use_container_width=True)

# =====================================================================
# HALAMAN 3: TRAINING & EVALUASI
# =====================================================================
elif menu == "🤖 Training & Evaluasi":
    st.title("🤖 Training Model AI")
    
    if not ML_AVAILABLE:
        st.warning("Fitur ini dinonaktifkan karena library Machine Learning tidak terinstall.")
        st.stop()
        
    # Panduan Pengguna (User Guidance)
    st.markdown("""
    <div class="info-box">
        ℹ️ <b>Panduan Pelatihan Model:</b><br>
        1. Tentukan parameter <b>Window Size</b> (berapa jam ke belakang yang dilihat model).<br>
        2. Klik tombol <b>Mulai Training</b>.<br>
        3. Sistem akan melatih <b>Hybrid Model (LSTM + XGBoost)</b> secara otomatis.
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.subheader("⚙️ Konfigurasi")
        lookback = st.slider("Lookback Window (Jam)", 12, 48, Config.DEFAULT_LOOKBACK, help="Jumlah jam data historis yang digunakan sebagai input prediksi.")
        epochs = st.slider("Epochs", 5, 50, Config.DEFAULT_EPOCHS, help="Jumlah putaran pelatihan model.")
        
        if st.button("🚀 Mulai Training", type="primary", use_container_width=True):
            with st.status("Sedang melatih model...", expanded=True):
                st.write("📥 Menyiapkan tensor data...")
                (X_train, y_train), (X_test, y_test), scaler = engine.prepare_tensors(df, lookback)
                
                st.write("🧠 Melatih arsitektur Hybrid...")
                results = engine.train_hybrid_model(X_train, y_train, X_test, y_test, scaler, epochs)
                
                st.session_state.training_results = results
                st.success("Training Selesai!")
                
    with c2:
        if st.session_state.training_results:
            res = st.session_state.training_results['Hybrid']
            
            st.subheader("🏆 Hasil Evaluasi (Data Uji)")
            
            k1, k2, k3 = st.columns(3)
            k1.metric("MAPE (Akurasi)", f"{res['MAPE']:.2f}%", "Target < 10%", delta_color="inverse")
            k2.metric("RMSE (Error)", f"{res['RMSE']:.2f} kWh", "Lebih kecil lebih baik", delta_color="inverse")
            k3.metric("R-Squared", f"{res['R2']:.3f}", "Mendekati 1.0 sempurna")
            
            st.subheader("📉 Prediksi vs Aktual")
            
            # Grafik Interaktif Zoomable
            chart_df = pd.DataFrame({
                'Actual': res['Actual'],
                'Predicted': res['Predicted']
            }).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=chart_df['Actual'], name='Data Aktual', line=dict(color=Config.COLORS['chart_actual'], width=2)))
            fig.add_trace(go.Scatter(y=chart_df['Predicted'], name='Prediksi Hybrid', line=dict(color=Config.COLORS['chart_pred'], width=2, dash='dash')))
            
            fig.update_layout(title="Visualisasi Performa Model", xaxis_title="Sample Data Uji", yaxis_title="Energi (kWh)", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("Belum ada model yang dilatih. Silakan klik tombol di samping.")

# =====================================================================
# HALAMAN 4: SIMULASI
# =====================================================================
elif menu == "🔮 Simulasi Prediksi":
    st.title("🔮 Simulasi Prediksi Real-time")
    
    if not st.session_state.training_results:
        st.warning("⚠️ Harap latih model terlebih dahulu di menu 'Training & Evaluasi'.")
        st.stop()
        
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎛️ Parameter Kondisi")
        input_hour = st.slider("Jam Operasional", 0, 23, 12)
        input_temp = st.slider("Suhu Lingkungan (°C)", 20, 35, 28)
        is_holiday = st.checkbox("Hari Libur?")
        
        if st.button("Kalkulasi Beban", type="primary", use_container_width=True):
            # Logika Dummy Cerdas (Placeholder untuk model inference real-time)
            # Menggunakan statistik rata-rata dari data historis
            base = df[df['hour'] == input_hour]['energy_consumption'].mean()
            
            # Faktor koreksi
            if is_holiday: base *= 0.85
            temp_factor = 1 + ((input_temp - 24) * 0.02) # +2% tiap kenaikan 1 derajat
            
            pred_val = base * temp_factor
            
            st.markdown("---")
            st.metric("Estimasi Beban Listrik", f"{pred_val:.2f} kWh")
            
            if pred_val > df['energy_consumption'].quantile(0.90):
                st.error("⚠️ Peringatan: Beban Tinggi!")
            else:
                st.success("✅ Kondisi Normal")
                
    with col2:
        st.subheader("📅 Proyeksi 24 Jam")
        
        # Ambil pola rata-rata per jam dari data historis sebagai baseline
        hourly_profile = df.groupby('hour')['energy_consumption'].mean().values
        
        # Geser grafik agar dimulai dari jam input
        forecast = np.roll(hourly_profile, -input_hour)
        
        fig = px.area(x=range(24), y=forecast, title="Kurva Beban Harian (Baseline Profile)",
                      labels={'x': 'Jam ke Depan', 'y': 'Estimasi kWh'})
        fig.update_traces(line_color=Config.COLORS['accent'])
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.caption("© 2024 Streamlit & TensorFlow Hybrid")