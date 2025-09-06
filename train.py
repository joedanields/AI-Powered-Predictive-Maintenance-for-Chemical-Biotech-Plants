"""
Enhanced model training script with persistence
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import DataLoader
from models.anomaly_detection import AnomalyDetector
import pandas as pd
import numpy as np

def train_all_models():
    """Train and save all models"""
    print("🚀 Starting comprehensive model training...")
    
    # Load data
    print("📊 Loading training data...")
    loader = DataLoader()
    df = loader.load_synthetic_sensor_data(10000)
    print(f"✅ Loaded {len(df):,} data points")
    
    # Train Anomaly Detection
    print("\n🤖 Training Anomaly Detection Model...")
    anomaly_detector = AnomalyDetector(contamination=0.15)
    train_size = int(0.7 * len(df))
    train_data = df.iloc[:train_size]
    
    try:
        anomaly_detector.train(train_data)
        print("✅ Anomaly detection model trained and saved!")
    except Exception as e:
        print(f"❌ Anomaly detection training failed: {e}")
        return False
    
    # Model evaluation
    print("\n📊 Model Evaluation...")
    test_data = df.iloc[train_size:]
    
    try:
        predictions, scores, raw_predictions = anomaly_detector.predict(test_data)
        
        if 'is_anomaly' in test_data.columns:
            from sklearn.metrics import classification_report, accuracy_score
            accuracy = accuracy_score(test_data['is_anomaly'], predictions)
            print(f"🎯 Anomaly Detection Accuracy: {accuracy:.3f}")
            print("\n📋 Classification Report:")
            print(classification_report(test_data['is_anomaly'], predictions))
        
        print(f"\n📈 Model Statistics:")
        print(f"   • Total predictions: {len(predictions):,}")
        print(f"   • Anomalies detected: {sum(predictions):,}")
        print(f"   • Anomaly rate: {(sum(predictions)/len(predictions)*100):.1f}%")
        print(f"   • Average confidence: {np.mean(np.abs(scores)):.3f}")
        
    except Exception as e:
        print(f"⚠️ Model evaluation failed: {e}")
    
    print("\n🏆 Training completed successfully!")
    print("💾 Models saved in /models/ directory")
    print("🚀 Ready for dashboard deployment!")
    
    return True

if __name__ == "__main__":
    success = train_all_models()
    if success:
        print("\n✅ All systems ready! Run 'streamlit run dashboard/app.py' to start dashboard.")
    else:
        print("\n❌ Training failed. Check error messages above.")
