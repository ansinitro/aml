# Data Sources and Methodology - Aktobe Air Quality Analysis

## Overview

This document explains the data sources used in the Aktobe air quality analysis. Unlike previous attempts that relied on synthetic data, this project utilizes **real historical monitoring data** from two specific stations in Aktobe.

## Data Sources

### 1. Monitoring Stations

We obtained and processed data from two specific air quality monitoring stations in Aktobe:

#### Station 216661 (Eset Batyra 109A)
- **Location:** Eset Batyra 109A, Aktobe
- **Parameters Monitored:**
  - PM2.5 (Fine Particulate Matter)
  - PM10 (Coarse Particulate Matter)
  - PM1
  - Meteorological Data: Temperature, Humidity, Pressure
- **Data Format:** HTML-wrapped CSV files (`Daily pm25`, `Daily met.t`, etc.)

#### Station 517420 (Zhankozha batyr koshesi, 89)
- **Location:** Zhankozha batyr koshesi, 89, Aktobe
- **Parameters Monitored:**
  - PM2.5
  - PM10
  - NO₂ (Nitrogen Dioxide)
  - SO₂ (Sulfur Dioxide)
  - CO (Carbon Monoxide)

### 2. Data Characteristics

- **Source:** Data files were obtained from AQICN/AirKaz archives.
- **Period:** September 2021 – December 2025 (varying coverage per parameter).
- **Resolution:** Daily averages (min, max, median, standard deviation).

## Methodology

### 1. Data Parsing and Cleaning

The raw data files were in a non-standard format (CSV embedded in HTML). We developed a custom parser (`src/process_real_data.py`) to:
1.  **Extract CSV content:** Remove HTML tags and formatting.
2.  **Parse Dates:** Convert date strings to datetime objects.
3.  **Handle Missing Values:** Identified that `0` values in pollutant columns represented missing data, not zero pollution. These were replaced with `NaN`.

### 2. Data Merging

Data from both stations was merged into a single unified dataset:
- **Aggregation:** When both stations reported data for the same parameter on the same day, the values were averaged.
- **Alignment:** All parameters were aligned to a common daily time index.

### 3. Analysis Pipeline

The merged dataset (`data/processed/aktobe_real_merged.csv`) was then fed into our standard analysis pipeline:
1.  **Preprocessing:** Outlier detection and AQI calculation.
2.  **Analysis:** Statistical summaries, seasonal decomposition, and trend analysis.
3.  **Visualization:** Generation of time-series, boxplots, and heatmaps.

## References

1.  **AirKaz.org**: Source of real-time and historical data for Kazakhstan.
2.  **AQICN**: World Air Quality Index project.
3.  **WHO Global Air Quality Guidelines (2021)**: Benchmarks for analysis.
