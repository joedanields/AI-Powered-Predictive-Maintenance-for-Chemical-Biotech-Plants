import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import time
import os
import sys

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import our modules
from utils.data_loader import DataLoader
from models.anomaly_detection import AnomalyDetector

# Page configuration
st.set_page_config(
    page_title="AI Predictive Maintenance",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main > div {
    padding-top: 2rem;
}
.stMetric {
    background-color: #f0f2f6;
    border: 1px solid #ddd;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.alert-critical {
    background-color: #ffebee;
    border-left: 4px solid #f44336;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
.alert-warning {
    background-color: #fff3e0;
    border-left: 4px solid #ff9800;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
.alert-normal {
    background-color: #e8f5e8;
    border-left: 4px solid #4caf50;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
.equipment-status {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    border-radius: 1rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Title and header
st.title("🏭 AI-Powered Predictive Maintenance Dashboard")
st.markdown("*Real-time monitoring for Chemical & Biotech Plants*")
st.markdown("---")

# Initialize session state
if 'data_index' not in st.session_state:
    st.session_state.data_index = 0
if 'is_simulation_running' not in st.session_state:
    st.session_state.is_simulation_running = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'detector' not in st.session_state:
    st.session_state.detector = None

# Sidebar controls
st.sidebar.header("🎛️ Control Panel")
plant_section = st.sidebar.selectbox(
    "Plant Section", 
    ["Reactor Unit", "Pump Station", "Heat Exchanger", "Distillation Column", "Compressor Unit"]
)
simulation_speed = st.sidebar.slider("Simulation Speed", 1, 20, 5)

# Equipment type mapping
equipment_params = {
    "Reactor Unit": {"temp_range": (300, 450), "pressure_range": (10, 25), "critical_temp": 400},
    "Pump Station": {"temp_range": (60, 120), "pressure_range": (5, 15), "critical_temp": 100},
    "Heat Exchanger": {"temp_range": (150, 300), "pressure_range": (3, 12), "critical_temp": 250},
    "Distillation Column": {"temp_range": (200, 350), "pressure_range": (1, 8), "critical_temp": 320},
    "Compressor Unit": {"temp_range": (80, 200), "pressure_range": (15, 50), "critical_temp": 150}
}

# Data loading and model initialization
@st.cache_data
def load_data():
    """Load sensor data with caching"""
    try:
        loader = DataLoader()
        df = loader.load_synthetic_sensor_data(8000)  # More data for better demo
        st.sidebar.success("✅ Data loaded successfully!")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def initialize_model():
    """Initialize anomaly detection model with caching"""
    try:
        detector = AnomalyDetector(contamination=0.15)
        return detector
    except Exception as e:
        st.error(f"Error initializing model: {e}")
        return None

# Load data
if st.session_state.df is None:
    with st.spinner("🔄 Loading sensor data..."):
        st.session_state.df = load_data()

df = st.session_state.df

if df is None:
    st.error("❌ Failed to load data. Please check the console for errors.")
    st.stop()

# Model training section
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Model Training")

if st.sidebar.button("🔄 Train Anomaly Detection Model"):
    with st.sidebar:
        with st.spinner("Training anomaly detection model..."):
            detector = AnomalyDetector(contamination=0.12)
            
            # Use first 70% of data for training (unsupervised)
            train_size = int(0.7 * len(df))
            train_data = df.iloc[:train_size]
            
            try:
                detector.train(train_data)
                st.session_state.detector = detector
                st.success("✅ Model trained successfully!")
                st.info(f"📊 Trained on {train_size:,} data points")
            except Exception as e:
                st.error(f"❌ Training failed: {e}")

# Load trained model
if st.session_state.detector is None:
    st.session_state.detector = initialize_model()

detector = st.session_state.detector

# Simulation controls
st.sidebar.markdown("---")
st.sidebar.subheader("▶️ Simulation Controls")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("▶️ Start", key="start_sim"):
        st.session_state.is_simulation_running = True

with col2:
    if st.button("⏸️ Pause", key="pause_sim"):
        st.session_state.is_simulation_running = False

reset_simulation = st.sidebar.button("🔄 Reset")
if reset_simulation:
    st.session_state.data_index = 0
    st.session_state.is_simulation_running = False
    st.rerun()

# Inject fault simulation
st.sidebar.markdown("---")
st.sidebar.subheader("🧪 Fault Injection")
inject_fault = st.sidebar.selectbox(
    "Simulate Fault Type", 
    ["None", "Temperature Spike", "Pressure Drop", "Vibration Increase", "Flow Blockage"]
)

# Main dashboard
window_size = 120
current_index = st.session_state.data_index
end_index = min(current_index + window_size, len(df))
current_data = df.iloc[current_index:end_index].copy()

# Apply fault injection to current data
if inject_fault != "None":
    fault_intensity = st.sidebar.slider("Fault Intensity", 0.1, 3.0, 1.5)
    
    if inject_fault == "Temperature Spike":
        current_data['temperature'] += fault_intensity * 50
    elif inject_fault == "Pressure Drop":
        current_data['pressure'] -= fault_intensity * 3
    elif inject_fault == "Vibration Increase":
        current_data['vibration'] += fault_intensity * 1.5
    elif inject_fault == "Flow Blockage":
        current_data['flow_rate'] -= fault_intensity * 20

# Get current window for analysis
sensor_cols = ['temperature', 'pressure', 'flow_rate', 'vibration']

# Anomaly detection
if len(current_data) > 0 and detector is not None:
    try:
        predictions, scores, raw_predictions = detector.predict(current_data)
    except Exception as e:
        st.sidebar.error(f"❌ Prediction error: {e}")
        predictions = np.zeros(len(current_data))
        scores = np.zeros(len(current_data))
        raw_predictions = np.ones(len(current_data))
else:
    predictions = np.zeros(len(current_data))
    scores = np.zeros(len(current_data))
    raw_predictions = np.ones(len(current_data))

# Current metrics
if len(current_data) > 0:
    current_temp = current_data['temperature'].iloc[-1]
    current_pressure = current_data['pressure'].iloc[-1]
    current_flow = current_data['flow_rate'].iloc[-1]
    current_vibration = current_data['vibration'].iloc[-1]
    current_anomalies = sum(predictions)
    
    # Main dashboard layout
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.subheader("📊 Real-time Sensor Data")
        
        # Create comprehensive sensor plot
        fig = go.Figure()
        
        # Add sensor traces
        colors = {'temperature': '#FF6B6B', 'pressure': '#4ECDC4', 'flow_rate': '#45B7D1', 'vibration': '#96CEB4'}
        
        for i, col in enumerate(sensor_cols):
            fig.add_trace(go.Scatter(
                x=current_data['timestamp'],
                y=current_data[col],
                mode='lines',
                name=col.replace('_', ' ').title(),
                line=dict(color=colors[col], width=2),
                yaxis=f'y{i+1}' if i > 0 else 'y'
            ))
        
        # Add anomaly markers
        anomaly_indices = np.where(predictions == 1)[0]
        if len(anomaly_indices) > 0:
            anomaly_data = current_data.iloc[anomaly_indices]
            fig.add_trace(go.Scatter(
                x=anomaly_data['timestamp'],
                y=anomaly_data['temperature'],
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=10, symbol='x', line=dict(width=2, color='darkred')),
                showlegend=True
            ))
        
        # Update layout with multiple y-axes
        fig.update_layout(
            title=f"{plant_section} - Live Sensor Monitoring",
            xaxis=dict(title="Time", showgrid=True),
            yaxis=dict(title="Temperature (°C)", side="left", color=colors['temperature']),
            yaxis2=dict(title="Pressure (Bar)", side="right", overlaying="y", color=colors['pressure']),
            yaxis3=dict(title="Flow (L/min)", side="left", overlaying="y", position=0.1, color=colors['flow_rate']),
            yaxis4=dict(title="Vibration (mm/s)", side="right", overlaying="y", position=0.9, color=colors['vibration']),
            height=450,
            showlegend=True,
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🚨 System Status")
        
        # Enhanced alert logic
        anomaly_percentage = (current_anomalies / window_size) * 100
        current_params = equipment_params[plant_section]
        
        # Multi-factor risk assessment
        temp_risk = max(0, (current_temp - current_params['critical_temp']) / 50)
        vibration_risk = max(0, (current_vibration - 1.0) / 2.0)
        anomaly_risk = anomaly_percentage / 15
        
        overall_risk = (temp_risk + vibration_risk + anomaly_risk) / 3
        
        if overall_risk > 0.7 or current_anomalies > window_size * 0.15:
            st.markdown('''
            <div class="alert-critical">
                <h3>🔴 CRITICAL ALERT</h3>
                <p>Immediate action required!</p>
                <p><strong>Risk Level:</strong> HIGH</p>
            </div>
            ''', unsafe_allow_html=True)
            risk_level = "CRITICAL"
            risk_color = "red"
        elif overall_risk > 0.4 or current_anomalies > window_size * 0.08:
            st.markdown('''
            <div class="alert-warning">
                <h3>🟡 WARNING</h3>
                <p>Monitoring required</p>
                <p><strong>Risk Level:</strong> ELEVATED</p>
            </div>
            ''', unsafe_allow_html=True)
            risk_level = "ELEVATED"
            risk_color = "orange"
        else:
            st.markdown('''
            <div class="alert-normal">
                <h3>🟢 NORMAL</h3>
                <p>All systems operational</p>
                <p><strong>Risk Level:</strong> LOW</p>
            </div>
            ''', unsafe_allow_html=True)
            risk_level = "LOW"
            risk_color = "green"
        
        # Key metrics
        st.metric("Anomalies Detected", f"{current_anomalies}/{window_size}", f"{anomaly_percentage:.1f}%")
        st.metric("System Health", f"{(1-overall_risk)*100:.0f}%", f"{overall_risk:.2f}")
        
        # Risk breakdown
        st.markdown("**Risk Factors:**")
        st.progress(min(temp_risk, 1.0), text=f"Temperature: {temp_risk*100:.0f}%")
        st.progress(min(vibration_risk, 1.0), text=f"Vibration: {vibration_risk*100:.0f}%")
        st.progress(min(anomaly_risk, 1.0), text=f"Anomalies: {anomaly_risk*100:.0f}%")
    
    with col3:
        st.subheader("⏰ Maintenance Forecast")
        
        # Enhanced RUL calculation
        degradation_factors = {
            'temperature': max(0, (current_temp - current_params['temp_range'][1]) / 100),
            'vibration': max(0, (current_vibration - 1.0) / 3.0),
            'pressure': max(0, abs(current_pressure - np.mean(current_params['pressure_range'])) / 10),
            'anomalies': anomaly_percentage / 20
        }
        
        total_degradation = sum(degradation_factors.values()) / len(degradation_factors)
        
        if total_degradation > 0.8:
            days_to_maintenance = max(1, int(3 * (1 - total_degradation)))
            st.error(f"🔴 URGENT: Maintenance needed in {days_to_maintenance} days!")
            maintenance_status = "URGENT"
        elif total_degradation > 0.5:
            days_to_maintenance = max(3, int(14 * (1 - total_degradation)))
            st.warning(f"🟡 Schedule maintenance in {days_to_maintenance} days")
            maintenance_status = "SCHEDULED"
        elif total_degradation > 0.2:
            days_to_maintenance = max(14, int(60 * (1 - total_degradation)))
            st.info(f"🔵 Maintenance due in {days_to_maintenance} days")
            maintenance_status = "PLANNED"
        else:
            st.success("🟢 No immediate maintenance required")
            days_to_maintenance = "> 90"
            maintenance_status = "NORMAL"
        
        st.metric("Days to Maintenance", days_to_maintenance, f"Status: {maintenance_status}")
        st.metric("Equipment Health", f"{(1-total_degradation)*100:.0f}%", f"-{total_degradation:.2f}")
        
        # Maintenance recommendations
        st.markdown("**🔧 Recommendations:**")
        recommendations = []
        
        if degradation_factors['temperature'] > 0.3:
            recommendations.append("• Check cooling system")
            recommendations.append("• Inspect heat exchangers")
        if degradation_factors['vibration'] > 0.3:
            recommendations.append("• Inspect bearings")
            recommendations.append("• Check alignment")
        if degradation_factors['pressure'] > 0.3:
            recommendations.append("• Check seals & gaskets")
            recommendations.append("• Inspect valves")
        if degradation_factors['anomalies'] > 0.3:
            recommendations.append("• Increase monitoring")
            recommendations.append("• Review procedures")
        
        if not recommendations:
            recommendations = ["• Continue normal operation", "• Maintain regular schedule"]
        
        for rec in recommendations[:4]:  # Show max 4 recommendations
            st.markdown(rec)

    # Equipment status overview
    st.markdown("---")
    st.subheader("🔧 Equipment Health Dashboard")
    
    # Enhanced metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        temp_delta = current_temp - df['temperature'].iloc[max(0, current_index-50):current_index].mean() if current_index > 50 else 0
        temp_status = "🔴" if current_temp > current_params['critical_temp'] else "🟢"
        st.metric(
            f"{temp_status} Reactor Temperature", 
            f"{current_temp:.1f}°C", 
            f"{temp_delta:+.1f}°C"
        )
    
    with col2:
        pressure_delta = current_pressure - df['pressure'].iloc[max(0, current_index-50):current_index].mean() if current_index > 50 else 0
        pressure_status = "🔴" if current_pressure < current_params['pressure_range'][0] else "🟢"
        st.metric(
            f"{pressure_status} System Pressure", 
            f"{current_pressure:.1f} Bar", 
            f"{pressure_delta:+.1f} Bar"
        )
    
    with col3:
        flow_delta = current_flow - df['flow_rate'].iloc[max(0, current_index-50):current_index].mean() if current_index > 50 else 0
        flow_status = "🔴" if current_flow < 70 else "🟢"
        st.metric(
            f"{flow_status} Flow Rate", 
            f"{current_flow:.1f} L/min", 
            f"{flow_delta:+.1f} L/min"
        )
    
    with col4:
        vibration_delta = current_vibration - df['vibration'].iloc[max(0, current_index-50):current_index].mean() if current_index > 50 else 0
        vibration_status = "🔴" if current_vibration > 2.0 else "🟢"
        st.metric(
            f"{vibration_status} Vibration Level", 
            f"{current_vibration:.2f} mm/s", 
            f"{vibration_delta:+.2f} mm/s"
        )

    # Historical analysis
    st.markdown("---")
    st.subheader("📈 Trend Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Equipment degradation trend
        historical_window = min(500, current_index)
        if historical_window > 0:
            hist_start = current_index - historical_window
            hist_data = df.iloc[hist_start:current_index]
            
            # Calculate degradation over time
            degradation_trend = []
            for i in range(0, len(hist_data), 10):
                data_slice = hist_data.iloc[i:i+10]
                temp_deg = max(0, (data_slice['temperature'].mean() - current_params['temp_range'][1]) / 100)
                vib_deg = max(0, (data_slice['vibration'].mean() - 1.0) / 3.0)
                degradation_trend.append((temp_deg + vib_deg) / 2)
            
            if degradation_trend:
                time_points = hist_data['timestamp'].iloc[::10][:len(degradation_trend)]
                
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(
                    x=time_points,
                    y=degradation_trend,
                    mode='lines+markers',
                    name='Equipment Degradation',
                    line=dict(color='orange', width=3),
                    fill='tonexty',
                    fillcolor='rgba(255,165,0,0.1)'
                ))
                
                fig_trend.update_layout(
                    title="Equipment Degradation Trend",
                    xaxis_title="Time",
                    yaxis_title="Degradation Score (0-1)",
                    height=300,
                    showlegend=True
                )
                
                st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        # Anomaly frequency over time
        if historical_window > 0 and detector is not None:
            try:
                hist_predictions, _, _ = detector.predict(hist_data)
                
                # Calculate anomaly frequency in rolling windows
                window_size_freq = 50
                anomaly_freq = []
                freq_times = []
                
                for i in range(0, len(hist_predictions) - window_size_freq, 10):
                    window_anomalies = hist_predictions[i:i+window_size_freq]
                    freq = sum(window_anomalies) / len(window_anomalies) * 100
                    anomaly_freq.append(freq)
                    freq_times.append(hist_data['timestamp'].iloc[i + window_size_freq//2])
                
                if anomaly_freq:
                    fig_freq = go.Figure()
                    fig_freq.add_trace(go.Scatter(
                        x=freq_times,
                        y=anomaly_freq,
                        mode='lines+markers',
                        name='Anomaly Frequency',
                        line=dict(color='red', width=2),
                        marker=dict(size=4)
                    ))
                    
                    fig_freq.update_layout(
                        title="Anomaly Frequency Over Time",
                        xaxis_title="Time",
                        yaxis_title="Anomaly Rate (%)",
                        height=300,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_freq, use_container_width=True)
            except Exception as e:
                st.error(f"Error in anomaly frequency analysis: {e}")

# Auto-refresh simulation
if st.session_state.is_simulation_running:
    if current_index < len(df) - window_size:
        time.sleep(0.1)  # Control simulation speed
        st.session_state.data_index += simulation_speed
        st.rerun()
    else:
        st.session_state.is_simulation_running = False
        st.success("🏁 Simulation completed! Click Reset to restart.")

# Progress indicator
progress_pct = min((current_index / len(df)) * 100, 100)
st.sidebar.progress(progress_pct / 100, text=f"Progress: {progress_pct:.1f}%")

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 0.9em; padding: 20px;'>
    🏭 <strong>AI-Powered Predictive Maintenance Dashboard</strong><br/>
    <em>Monitoring {len(df):,} data points • Current time: {current_data['timestamp'].iloc[-1] if len(current_data) > 0 else 'N/A'}</em><br/>
    Built for Chemical & Biotech Industries • Real-time anomaly detection & predictive maintenance
</div>
""", unsafe_allow_html=True)
