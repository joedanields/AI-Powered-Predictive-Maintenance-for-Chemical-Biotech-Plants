"""
Simple run script that fixes import issues
"""
import os
import sys
import subprocess

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Create __init__.py files if they don't exist
init_files = [
    '__init__.py',
    'utils/__init__.py',
    'models/__init__.py',
    'dashboard/__init__.py'
]

for init_file in init_files:
    if not os.path.exists(init_file):
        os.makedirs(os.path.dirname(init_file), exist_ok=True)
        with open(init_file, 'w') as f:
            f.write("# Package initialization\n")

print("🚀 Starting AI Predictive Maintenance Dashboard...")
print("📁 Fixed Python path and created package files...")

# Run Streamlit
subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'dashboard/app.py'])
