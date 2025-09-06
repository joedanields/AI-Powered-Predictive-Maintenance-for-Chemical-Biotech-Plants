import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

class AnomalyDetector:
    def __init__(self, contamination=0.1):
        self.model = IsolationForest(
            contamination=contamination, 
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = None
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
    
    def prepare_features(self, df):
        """Prepare features for anomaly detection"""
        feature_cols = ['temperature', 'pressure', 'flow_rate', 'vibration']
        
        # Add derived features
        df_features = df[feature_cols].copy()
        
        # Rolling statistics (for time series patterns)
        for col in feature_cols:
            df_features[f'{col}_rolling_mean_5'] = df[col].rolling(window=5, min_periods=1).mean()
            df_features[f'{col}_rolling_std_5'] = df[col].rolling(window=5, min_periods=1).std()
            df_features[f'{col}_diff'] = df[col].diff().fillna(0)
        
        # Cross-feature relationships
        df_features['temp_pressure_ratio'] = df['temperature'] / df['pressure']
        df_features['flow_vibration_product'] = df['flow_rate'] * df['vibration']
        
        return df_features.fillna(0)
    
    def train(self, df):
        """Train anomaly detection model"""
        X = self.prepare_features(df)
        self.feature_names = X.columns.tolist()
        
        # Use only normal data for training (unsupervised approach)
        normal_data = df[df['is_anomaly'] == 0] if 'is_anomaly' in df.columns else df
        X_normal = self.prepare_features(normal_data)
        
        # Scale features
        X_normal_scaled = self.scaler.fit_transform(X_normal)
        
        # Train model
        self.model.fit(X_normal_scaled)
        self.is_trained = True
        
        # Save model
        self.save_model()
        
        # Evaluate on full dataset if labels available
        if 'is_anomaly' in df.columns:
            predictions, scores, raw_predictions = self.predict(df)  # FIX: Unpack 3 values!
            self.evaluate(df['is_anomaly'].values, predictions)
        
        return self

    
    def predict(self, df):
        """Predict anomalies"""
        if not self.is_trained:
            self.load_model()
        
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        # Predict (-1 for anomaly, 1 for normal)
        predictions = self.model.predict(X_scaled)
        
        # Get anomaly scores (lower = more anomalous)
        scores = self.model.score_samples(X_scaled)
        
        # Convert to binary (1 for anomaly, 0 for normal)
        anomaly_binary = (predictions == -1).astype(int)
        
        return anomaly_binary, scores, predictions
    
    def evaluate(self, y_true, y_pred):
        """Evaluate model performance"""
        print("Anomaly Detection Performance:")
        print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
    
    def save_model(self):
        """Save trained model"""
        joblib.dump(self.model, 'models/anomaly_model.pkl')
        joblib.dump(self.scaler, 'models/anomaly_scaler.pkl')
        joblib.dump(self.feature_names, 'models/feature_names.pkl')
        print("Models saved successfully!")
    
    def load_model(self):
        """Load trained model"""
        try:
            self.model = joblib.load('models/anomaly_model.pkl')
            self.scaler = joblib.load('models/anomaly_scaler.pkl')
            self.feature_names = joblib.load('models/feature_names.pkl')
            self.is_trained = True
            print("Models loaded successfully!")
        except FileNotFoundError:
            print("No saved models found. Please train the model first.")
