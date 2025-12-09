"""
Configuration file for Aktobe Air Pollution Analysis
Contains constants, API endpoints, and WHO standards
"""

import os
from datetime import datetime

# City Information
CITY_NAME = "Aktobe"
CITY_COORDINATES = {
    'latitude': 50.2839,
    'longitude': 57.1670
}

# Date Range for Analysis
START_DATE = "2021-09-10"
END_DATE = "2025-12-31"

# API Configuration
API_KEYS = {
    'iqair': os.getenv('IQAIR_API_KEY', ''),  # Get from environment variable
    'openweathermap': os.getenv('OPENWEATHER_API_KEY', ''),
}

API_ENDPOINTS = {
    'iqair': 'https://api.airvisual.com/v2',
    'openweathermap': 'https://api.openweathermap.org/data/2.5',
    'waqi': 'https://api.waqi.info/feed',
}

# WHO Air Quality Standards (Annual Mean, μg/m³)
WHO_STANDARDS = {
    'PM2.5': {
        'annual': 5,      # Updated 2021 guidelines
        'daily': 15,
        'unit': 'μg/m³'
    },
    'PM10': {
        'annual': 15,     # Updated 2021 guidelines
        'daily': 45,
        'unit': 'μg/m³'
    },
    'NO2': {
        'annual': 10,     # Updated 2021 guidelines
        'daily': 25,
        'unit': 'μg/m³'
    },
    'SO2': {
        'daily': 40,      # Updated 2021 guidelines
        'unit': 'μg/m³'
    },
    'O3': {
        'peak_season': 60,  # Updated 2021 guidelines
        'unit': 'μg/m³'
    },
    'CO': {
        'daily': 4,       # mg/m³
        'unit': 'mg/m³'
    }
}

# AQI Breakpoints (US EPA Standard)
AQI_BREAKPOINTS = {
    'PM2.5': [
        (0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500),
    ],
    'PM10': [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 604, 301, 500),
    ],
}

# Pollutants to analyze
POLLUTANTS = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']

# Seasons definition for Kazakhstan
SEASONS = {
    'Winter': [12, 1, 2],
    'Spring': [3, 4, 5],
    'Summer': [6, 7, 8],
    'Autumn': [9, 10, 11]
}

# Heating season (typically October to April in Kazakhstan)
HEATING_SEASON_MONTHS = [10, 11, 12, 1, 2, 3, 4]

# Data directories
DATA_DIRS = {
    'raw': 'data/raw',
    'processed': 'data/processed',
    'external': 'data/external',
    'figures': 'reports/figures'
}

# Aktobe-specific context
AKTOBE_CONTEXT = {
    'population': 500000,  # Approximate
    'major_industries': ['Oil and Gas', 'Chemical Production', 'Metallurgy'],
    'geography': 'Steppe region, relatively flat terrain',
    'climate': 'Continental - cold winters, hot summers',
    'elevation': 219,  # meters above sea level
}

# Visualization settings
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
COLOR_PALETTE = 'husl'
FIGURE_SIZE = (12, 6)
DPI = 300

# Random seed for reproducibility
RANDOM_SEED = 42
