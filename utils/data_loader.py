import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

class DataLoader:
    def __init__(self):
        self.scaler = StandardScaler()
        
        # Create data directories if they don't exist
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('models', exist_ok=True)
    
    def load_synthetic_sensor_data(self, n_samples=10000):
        """Generate realistic chemical plant sensor data"""
        np.random.seed(42)
        
        # Time-based patterns for more realistic data
        time_hours = np.linspace(0, 24 * (n_samples / 1440), n_samples)  # Assuming 1 sample per minute
        
        # Normal operation data with daily patterns
        base_temp = 350 + 20 * np.sin(2 * np.pi * time_hours / 24)  # Daily temperature cycle
        temp = base_temp + np.random.normal(0, 5, n_samples)
        
        base_pressure = 15 + 2 * np.sin(2 * np.pi * time_hours / 24 + np.pi/4)
        pressure = base_pressure + np.random.normal(0, 1, n_samples)
        
        base_flow = 100 + 10 * np.sin(2 * np.pi * time_hours / 12)  # Twice daily cycle
        flow_rate = base_flow + np.random.normal(0, 3, n_samples)
        
        vibration = np.random.normal(0.5, 0.1, n_samples)
        
        # Introduce progressive anomalies (last 15% of data)
        anomaly_start = int(0.85 * n_samples)
        anomaly_length = n_samples - anomaly_start
        
        # Progressive degradation
        degradation_factor = np.linspace(0, 1, anomaly_length)
        
        temp[anomaly_start:] += degradation_factor * np.random.normal(30, 10, anomaly_length)
        pressure[anomaly_start:] += degradation_factor * np.random.normal(-3, 1, anomaly_length)
        vibration[anomaly_start:] += degradation_factor * np.random.normal(1.5, 0.3, anomaly_length)
        flow_rate[anomaly_start:] -= degradation_factor * np.random.normal(20, 5, anomaly_length)
        
        # Create labels
        labels = np.zeros(n_samples)
        labels[anomaly_start:] = 1
        
        # Add some random isolated anomalies in normal data (5%)
        normal_anomaly_indices = np.random.choice(anomaly_start, int(0.05 * anomaly_start), replace=False)
        labels[normal_anomaly_indices] = 1
        temp[normal_anomaly_indices] += np.random.normal(40, 10, len(normal_anomaly_indices))
        vibration[normal_anomaly_indices] += np.random.normal(2, 0.5, len(normal_anomaly_indices))
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1min'),
            'temperature': temp,
            'pressure': pressure,
            'flow_rate': flow_rate,
            'vibration': vibration,
            'is_anomaly': labels.astype(int),
            'equipment_id': ['REACTOR_001'] * n_samples,
            'shift': [(i // 480) % 3 + 1 for i in range(n_samples)]  # 8-hour shifts
        })
        
        # Calculate RUL (Remaining Useful Life)
        rul = np.zeros(n_samples)
        for i in range(n_samples):
            if i < anomaly_start:
                rul[i] = anomaly_start - i  # Minutes until degradation starts
            else:
                # During degradation, RUL decreases
                rul[i] = max(0, n_samples - i - 100)  # 100 minutes grace period
        
        df['rul_minutes'] = rul
        df['rul_hours'] = rul / 60
        
        return df
    
    def save_processed_data(self, df, filename='processed_sensor_data.csv'):
        """Save processed data"""
        filepath = f'data/processed/{filename}'
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
        return filepath
    
    def load_processed_data(self, filename='processed_sensor_data.csv'):
        """Load previously saved data"""
        filepath = f'data/processed/{filename}'
        if os.path.exists(filepath):
            return pd.read_csv(filepath, parse_dates=['timestamp'])
        else:
            print(f"File {filepath} not found. Generating new data...")
            return self.load_synthetic_sensor_data()
