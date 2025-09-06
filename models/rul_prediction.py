import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os

class RULPredictor:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        os.makedirs('models', exist_ok=True)
    
    def prepare_sequences(self, df, target_col='rul_hours'):
        """Prepare time series sequences for LSTM"""
        feature_cols = ['temperature', 'pressure', 'flow_rate', 'vibration']
        
        # Scale features
        X = self.scaler.fit_transform(df[feature_cols])
        y = df[target_col].values
        
        # Create sequences
        X_seq, y_seq = [], []
        
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def build_model(self, n_features):
        """Build LSTM model for RUL prediction"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(self.sequence_length, n_features)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, df):
        """Train RUL prediction model"""
        X, y = self.prepare_sequences(df)
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Build model
        self.model = self.build_model(X.shape[2])
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=50,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"RUL Prediction Performance:")
        print(f"MSE: {mse:.2f}")
        print(f"MAE: {mae:.2f}")
        
        # Save model
        self.save_model()
        
        return history
    
    def predict(self, df):
        """Predict RUL for new data"""
        if not self.is_trained:
            self.load_model()
        
        if len(df) < self.sequence_length:
            # Pad with last available data
            padding = df.iloc[-1:].repeat(self.sequence_length - len(df))
            df_padded = pd.concat([df, padding])
        else:
            df_padded = df
        
        X, _ = self.prepare_sequences(df_padded)
        
        if len(X) == 0:
            return np.array([df['rul_hours'].iloc[-1]])
        
        predictions = self.model.predict(X[-1:])  # Predict for last sequence
        return predictions[0]
    
    def save_model(self):
        """Save trained model"""
        self.model.save('models/rul_model.h5')
        joblib.dump(self.scaler, 'models/rul_scaler.pkl')
        print("RUL model saved successfully!")
    
    def load_model(self):
        """Load trained model"""
        try:
            self.model = tf.keras.models.load_model('models/rul_model.h5')
            self.scaler = joblib.load('models/rul_scaler.pkl')
            self.is_trained = True
            print("RUL model loaded successfully!")
        except (FileNotFoundError, OSError):
            print("No saved RUL model found. Please train the model first.")
