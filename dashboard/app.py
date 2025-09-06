import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import time
import os
import sys
import json

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import DataLoader
from models.anomaly_detection import AnomalyDetector

# Page configuration
st.set_page_config(
    page_title="AI Predictive Maintenance",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
.main > div {
    padding-top: 1rem;
}
.stMetric {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.cost-benefit-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    border-radius: 1rem;
    margin: 1rem 0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.health-score-excellent {
    background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 1rem;
    text-align: center;
    margin: 1rem 0;
}
.health-score-good {
    background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 1rem;
    text-align: center;
    margin: 1rem 0;
}
.health-score-poor {
    background: linear-gradient(135deg, #F44336 0%, #D32F2F 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 1rem;
    text-align: center;
    margin: 1rem 0;
}
.maintenance-schedule {
    background: #f8f9fa;
    border-left: 4px solid #007bff;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 0.25rem;
}
.fault-injection-panel {
    background: #fff3cd;
    border: 1px solid #ffeaa7;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_index' not in st.session_state:
    st.session_state.data_index = 0
if 'is_simulation_running' not in st.session_state:
    st.session_state.is_simulation_running = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'maintenance_schedule' not in st.session_state:
    st.session_state.maintenance_schedule = []
if 'total_cost_savings' not in st.session_state:
    st.session_state.total_cost_savings = 0
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Title
st.title("🏭 AI-Powered Predictive Maintenance Dashboard")
st.markdown("*Advanced Real-time Monitoring for Chemical & Biotech Plants*")

# Enhanced sidebar
st.sidebar.header("🎛️ Control Panel")

# Plant configuration
plant_sections = {
    "Reactor Unit": {"cost_per_hour": 50000, "critical_temp": 400, "critical_pressure": 20},
    "Pump Station": {"cost_per_hour": 15000, "critical_temp": 100, "critical_pressure": 12},
    "Heat Exchanger": {"cost_per_hour": 25000, "critical_temp": 250, "critical_pressure": 15},
    "Distillation Column": {"cost_per_hour": 35000, "critical_temp": 320, "critical_pressure": 8},
    "Compressor Unit": {"cost_per_hour": 40000, "critical_temp": 150, "critical_pressure": 30}
}

plant_section = st.sidebar.selectbox("Plant Section", list(plant_sections.keys()))
plant_config = plant_sections[plant_section]

simulation_speed = st.sidebar.slider("Simulation Speed", 1, 20, 5)

# Enhanced Fault Injection Panel
st.sidebar.markdown("---")
st.sidebar.subheader("🧪 Advanced Fault Injection")

fault_types = {
    "None": {"temp": 0, "pressure": 0, "flow": 0, "vibration": 0},
    "Temperature Spike": {"temp": 60, "pressure": 2, "flow": -5, "vibration": 1},
    "Pressure Drop": {"temp": 10, "pressure": -8, "flow": -15, "vibration": 0.5},
    "Flow Blockage": {"temp": 20, "pressure": 5, "flow": -40, "vibration": 2},
    "Bearing Failure": {"temp": 30, "pressure": 0, "flow": -10, "vibration": 3},
    "Heat Exchanger Fouling": {"temp": 40, "pressure": -3, "flow": -20, "vibration": 1.5},
    "Pump Cavitation": {"temp": 15, "pressure": -6, "flow": -25, "vibration": 2.5},
    "Multiple Failures": {"temp": 50, "pressure": -5, "flow": -30, "vibration": 2.8}
}

inject_fault = st.sidebar.selectbox("Fault Scenario", list(fault_types.keys()))
fault_intensity = st.sidebar.slider("Fault Intensity", 0.1, 3.0, 1.5)

if inject_fault != "None":
    st.sidebar.markdown(f"""
    <div class="fault-injection-panel">
        <strong>🚨 Active Fault: {inject_fault}</strong><br/>
        <small>Intensity: {fault_intensity:.1f}x</small><br/>
        <small>💰 Downtime Cost: ₹{plant_config['cost_per_hour']:,}/hour</small>
    </div>
    """, unsafe_allow_html=True)

# Data loading
@st.cache_data
def load_data():
    loader = DataLoader()
    df = loader.load_synthetic_sensor_data(8000)
    return df

@st.cache_resource
def initialize_model():
    return AnomalyDetector(contamination=0.12)

# Load data
if st.session_state.df is None:
    with st.spinner("🔄 Loading sensor data..."):
        st.session_state.df = load_data()

df = st.session_state.df

# Enhanced Model Training
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI Model Training")

if not st.session_state.model_trained:
    st.sidebar.warning("⚠️ Model not trained yet!")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🔄 Train Model"):
        with st.spinner("🤖 Training AI model..."):
            detector = AnomalyDetector(contamination=0.15)
            train_size = int(0.7 * len(df))
            train_data = df.iloc[:train_size]
            
            # Train the model
            detector.train(train_data)
            
            st.session_state.detector = detector
            st.session_state.model_trained = True
            st.sidebar.success("✅ Model trained successfully!")
            st.sidebar.info(f"📊 Trained on {train_size:,} data points")

with col2:
    if st.button("💾 Save Model"):
        if st.session_state.detector and st.session_state.model_trained:
            st.session_state.detector.save_model()
            st.sidebar.success("✅ Model saved!")
        else:
            st.sidebar.error("❌ Train model first!")

# Load or initialize detector
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

if st.sidebar.button("🔄 Reset Simulation"):
    st.session_state.data_index = 0
    st.session_state.is_simulation_running = False
    st.session_state.total_cost_savings = 0
    st.rerun()

# Main dashboard logic
window_size = 120
current_index = st.session_state.data_index
end_index = min(current_index + window_size, len(df))
current_data = df.iloc[current_index:end_index].copy()

# Apply fault injection
if inject_fault != "None":
    fault_params = fault_types[inject_fault]
    current_data['temperature'] += fault_intensity * fault_params['temp']
    current_data['pressure'] += fault_intensity * fault_params['pressure']
    current_data['flow_rate'] += fault_intensity * fault_params['flow']
    current_data['vibration'] += fault_intensity * fault_params['vibration']

# Anomaly detection
if len(current_data) > 0 and st.session_state.model_trained:
    try:
        predictions, scores, _ = detector.predict(current_data)
    except Exception as e:
        st.error(f"Model prediction error: {e}")
        predictions = np.zeros(len(current_data))
        scores = np.zeros(len(current_data))
else:
    predictions = np.zeros(len(current_data))
    scores = np.zeros(len(current_data))

# Calculate metrics
if len(current_data) > 0:
    current_temp = current_data['temperature'].iloc[-1]
    current_pressure = current_data['pressure'].iloc[-1]
    current_flow = current_data['flow_rate'].iloc[-1]
    current_vibration = current_data['vibration'].iloc[-1]
    current_anomalies = sum(predictions)
    
    # Equipment Health Score Calculation
    temp_score = max(0, 100 - abs(current_temp - 350) / 5)
    pressure_score = max(0, 100 - abs(current_pressure - 15) / 2)
    flow_score = max(0, 100 - abs(current_flow - 100) / 3)
    vibration_score = max(0, 100 - current_vibration * 40)
    anomaly_score = max(0, 100 - (current_anomalies / window_size) * 200)
    
    overall_health = (temp_score + pressure_score + flow_score + vibration_score + anomaly_score) / 5
    
    # Cost-Benefit Analysis
    if current_anomalies > window_size * 0.1:  # High anomaly rate
        predicted_downtime_hours = np.random.randint(4, 24)
        potential_loss = predicted_downtime_hours * plant_config['cost_per_hour']
        maintenance_cost = potential_loss * 0.1  # Maintenance is 10% of downtime cost
        cost_savings = potential_loss - maintenance_cost
        st.session_state.total_cost_savings += cost_savings * 0.001  # Accumulate savings
    
    # Main Layout
    # Top KPI Row
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    
    with kpi_col1:
        st.metric("🌡️ Temperature", f"{current_temp:.1f}°C", 
                 f"{current_temp - plant_config['critical_temp']:+.1f}°C")
    
    with kpi_col2:
        st.metric("📊 Pressure", f"{current_pressure:.1f} Bar",
                 f"{current_pressure - 15:.1f} Bar")
    
    with kpi_col3:
        st.metric("🌊 Flow Rate", f"{current_flow:.1f} L/min",
                 f"{current_flow - 100:.1f} L/min")
    
    with kpi_col4:
        st.metric("📳 Vibration", f"{current_vibration:.2f} mm/s",
                 f"{current_vibration - 0.5:+.2f} mm/s")
    
    with kpi_col5:
        if overall_health >= 80:
            st.metric("🏥 Health Score", f"{overall_health:.0f}%", "Excellent")
        elif overall_health >= 60:
            st.metric("🏥 Health Score", f"{overall_health:.0f}%", "Good")
        else:
            st.metric("🏥 Health Score", f"{overall_health:.0f}%", "Poor")

    st.markdown("---")
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("📊 Real-time Sensor Monitoring")
        
        # Enhanced sensor plot
        fig = go.Figure()
        
        # Add traces with better styling
        colors = {'temperature': '#FF6B6B', 'pressure': '#4ECDC4', 'flow_rate': '#45B7D1', 'vibration': '#96CEB4'}
        
        for col in ['temperature', 'pressure', 'flow_rate', 'vibration']:
            fig.add_trace(go.Scatter(
                x=current_data['timestamp'],
                y=current_data[col],
                mode='lines',
                name=col.replace('_', ' ').title(),
                line=dict(color=colors[col], width=3),
                hovertemplate=f'<b>{col.title()}</b><br>Value: %{{y:.2f}}<br>Time: %{{x}}<extra></extra>'
            ))
        
        # Add anomaly markers
        anomaly_indices = np.where(predictions == 1)[0]
        if len(anomaly_indices) > 0:
            anomaly_data = current_data.iloc[anomaly_indices]
            fig.add_trace(go.Scatter(
                x=anomaly_data['timestamp'],
                y=anomaly_data['temperature'],
                mode='markers',
                name='🚨 Anomalies Detected',
                marker=dict(
                    color='red', 
                    size=12, 
                    symbol='x',
                    line=dict(width=3, color='darkred')
                ),
                showlegend=True
            ))
        
        fig.update_layout(
            title=f"🏭 {plant_section} - Live Sensor Data Stream",
            xaxis_title="Time",
            yaxis_title="Sensor Values",
            height=500,
            showlegend=True,
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
            plot_bgcolor='rgba(0,0,0,0.02)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Equipment Health Dashboard
        st.subheader("🏥 Equipment Health")
        
        if overall_health >= 80:
            st.markdown(f'''
            <div class="health-score-excellent">
                <h2>🟢 EXCELLENT</h2>
                <h3>{overall_health:.0f}%</h3>
                <p>All systems operating optimally</p>
            </div>
            ''', unsafe_allow_html=True)
        elif overall_health >= 60:
            st.markdown(f'''
            <div class="health-score-good">
                <h2>🟡 ATTENTION</h2>
                <h3>{overall_health:.0f}%</h3>
                <p>Monitor closely</p>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="health-score-poor">
                <h2>🔴 CRITICAL</h2>
                <h3>{overall_health:.0f}%</h3>
                <p>Immediate action required!</p>
            </div>
            ''', unsafe_allow_html=True)
        
        # Individual component health
        st.markdown("**Component Health:**")
        st.progress(temp_score/100, text=f"Temperature: {temp_score:.0f}%")
        st.progress(pressure_score/100, text=f"Pressure: {pressure_score:.0f}%")
        st.progress(flow_score/100, text=f"Flow: {flow_score:.0f}%")
        st.progress(vibration_score/100, text=f"Vibration: {vibration_score:.0f}%")
        
        # Alert status
        st.markdown("---")
        if current_anomalies > window_size * 0.15:
            st.error("🚨 CRITICAL ALERT")
            st.markdown("**Immediate Actions Required:**")
            st.markdown("• Reduce production rate")
            st.markdown("• Alert maintenance team")
            st.markdown("• Prepare for emergency shutdown")
        elif current_anomalies > window_size * 0.08:
            st.warning("⚠️ WARNING")
            st.markdown("**Recommended Actions:**")
            st.markdown("• Increase monitoring frequency")
            st.markdown("• Schedule inspection")
            st.markdown("• Review operational parameters")
        else:
            st.success("✅ NORMAL OPERATION")
            st.markdown("**Status:**")
            st.markdown("• All systems nominal")
            st.markdown("• Continue normal operation")

    # Cost-Benefit Analysis Section
    st.markdown("---")
    st.subheader("💰 Cost-Benefit Analysis & ROI")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if current_anomalies > window_size * 0.1:
            predicted_downtime = np.random.randint(4, 24)
            potential_loss = predicted_downtime * plant_config['cost_per_hour']
            
            st.markdown(f'''
            <div class="cost-benefit-card">
                <h4>💸 Potential Loss Prevented</h4>
                <h2>₹{potential_loss:,}</h2>
                <p>Predicted downtime: {predicted_downtime} hours</p>
                <p>Hourly loss rate: ₹{plant_config['cost_per_hour']:,}</p>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="cost-benefit-card">
                <h4>💚 Operational Savings</h4>
                <h2>₹{plant_config['cost_per_hour']:,}/hr</h2>
                <p>Current operational efficiency</p>
                <p>No predicted losses</p>
            </div>
            ''', unsafe_allow_html=True)
    
    with col2:
        maintenance_cost = plant_config['cost_per_hour'] * 2  # 2 hours of maintenance
        st.markdown(f'''
        <div class="cost-benefit-card">
            <h4>🔧 Maintenance Investment</h4>
            <h2>₹{maintenance_cost:,}</h2>
            <p>Proactive maintenance cost</p>
            <p>vs. Reactive repairs</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        total_savings = st.session_state.total_cost_savings
        st.markdown(f'''
        <div class="cost-benefit-card">
            <h4>📈 Cumulative ROI</h4>
            <h2>₹{total_savings:,.0f}</h2>
            <p>Total savings this session</p>
            <p>ROI: {(total_savings/10000)*100:.1f}%</p>
        </div>
        ''', unsafe_allow_html=True)

    # Maintenance Scheduling
    st.markdown("---")
    st.subheader("📅 Intelligent Maintenance Scheduling")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Maintenance predictions based on current conditions
        if current_anomalies > window_size * 0.15:
            maintenance_urgency = "URGENT"
            days_until = 1
            priority = "🔴 CRITICAL"
        elif current_anomalies > window_size * 0.08:
            maintenance_urgency = "SCHEDULED"
            days_until = 7
            priority = "🟡 HIGH"
        elif overall_health < 70:
            maintenance_urgency = "PLANNED"
            days_until = 14
            priority = "🔵 MEDIUM"
        else:
            maintenance_urgency = "ROUTINE"
            days_until = 30
            priority = "🟢 LOW"
        
        next_maintenance = datetime.now() + timedelta(days=days_until)
        
        st.markdown(f'''
        <div class="maintenance-schedule">
            <h4>🔧 Next Maintenance Schedule</h4>
            <p><strong>Priority:</strong> {priority}</p>
            <p><strong>Recommended Date:</strong> {next_maintenance.strftime("%Y-%m-%d")}</p>
            <p><strong>Type:</strong> {maintenance_urgency}</p>
            <p><strong>Estimated Duration:</strong> {2 if maintenance_urgency == "URGENT" else 4} hours</p>
            <p><strong>Equipment:</strong> {plant_section}</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown("**📋 Maintenance Tasks:**")
        if maintenance_urgency == "URGENT":
            tasks = [
                "🔍 Emergency inspection",
                "🔧 Component replacement", 
                "⚡ Safety system check",
                "📊 Performance validation"
            ]
        elif maintenance_urgency == "SCHEDULED":
            tasks = [
                "🔍 Detailed inspection",
                "🧽 Cleaning & lubrication",
                "🔧 Minor adjustments",
                "📊 Calibration check"
            ]
        else:
            tasks = [
                "🔍 Routine inspection",
                "🧽 Standard cleaning",
                "📊 Data backup",
                "📋 Documentation update"
            ]
        
        for task in tasks:
            st.markdown(f"• {task}")
        
        if st.button("📅 Schedule Maintenance"):
            st.session_state.maintenance_schedule.append({
                'date': next_maintenance,
                'type': maintenance_urgency,
                'equipment': plant_section,
                'priority': priority
            })
            st.success("✅ Maintenance scheduled successfully!")

    # Model Performance Dashboard
    st.markdown("---")
    st.subheader("🤖 AI Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.session_state.model_trained:
            st.metric("🎯 Model Status", "TRAINED", "✅ Active")
        else:
            st.metric("🎯 Model Status", "UNTRAINED", "⚠️ Train Required")
    
    with col2:
        accuracy = 85 + (overall_health - 50) * 0.3  # Simulated accuracy based on health
        st.metric("📊 Detection Accuracy", f"{accuracy:.1f}%", f"{accuracy-80:.1f}%")
    
    with col3:
        confidence = np.mean(np.abs(scores)) if len(scores) > 0 else 0
        st.metric("🔍 Confidence Score", f"{confidence:.3f}", "AI Certainty")
    
    with col4:
        total_predictions = current_index + window_size
        st.metric("📈 Predictions Made", f"{total_predictions:,}", "This Session")

# Auto-refresh simulation
if st.session_state.is_simulation_running:
    if current_index < len(df) - window_size:
        time.sleep(0.1)
        st.session_state.data_index += simulation_speed
        st.rerun()
    else:
        st.session_state.is_simulation_running = False
        st.success("🏁 Simulation completed! Click Reset to restart.")

# Progress and status
progress_pct = min((current_index / len(df)) * 100, 100)
st.sidebar.progress(progress_pct / 100, text=f"Progress: {progress_pct:.1f}%")

if st.session_state.maintenance_schedule:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📅 Scheduled Maintenance")
    for i, maintenance in enumerate(st.session_state.maintenance_schedule[-3:]):  # Show last 3
        st.sidebar.markdown(f"**{maintenance['priority']}**")
        st.sidebar.markdown(f"📅 {maintenance['date'].strftime('%Y-%m-%d')}")
        st.sidebar.markdown(f"🏭 {maintenance['equipment']}")

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 0.9em; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px;'>
    🏭 <strong>AI-Powered Predictive Maintenance Dashboard v2.0</strong><br/>
    <em>Monitoring {len(df):,} data points • Model Status: {'✅ Trained' if st.session_state.model_trained else '⚠️ Untrained'} • Total Savings: ₹{st.session_state.total_cost_savings:,.0f}</em><br/>
    Real-time anomaly detection • Predictive maintenance • Cost optimization • Equipment health monitoring
</div>
""", unsafe_allow_html=True)
