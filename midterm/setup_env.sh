#!/bin/bash

# Create project structure
mkdir -p src output report/figures

# Create virtual environment
python3 -m venv .venv

# Activate and install dependencies
source .venv/bin/activate

pip install --upgrade pip
pip install numpy pandas matplotlib seaborn scikit-learn kagglehub[pandas-datasets] openpyxl

echo "Environment setup complete."
