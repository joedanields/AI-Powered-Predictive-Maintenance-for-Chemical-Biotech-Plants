"""
Quick start script for AI Predictive Maintenance Dashboard
"""
import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    packages = [
        'pandas', 'numpy', 'scikit-learn', 'tensorflow', 
        'streamlit', 'plotly', 'seaborn', 'matplotlib', 
        'xgboost', 'lightgbm', 'joblib'
    ]
    
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

def main():
    print("🚀 Starting AI Predictive Maintenance Dashboard...")
    
    # Create necessary directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Install packages if needed
    try:
        import streamlit
        import pandas
        import plotly
        import sklearn
    except ImportError:
        print("Installing required packages...")
        install_requirements()
    
    # Run the dashboard
    subprocess.run(['streamlit', 'run', 'dashboard/app.py'])

if __name__ == "__main__":
    main()
