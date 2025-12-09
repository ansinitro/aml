# Air Pollution Analysis for Aktobe, Kazakhstan

**Repository:** [https://github.com/ansinitro/aml/task1](https://github.com/ansinitro/aml/task1)

## Project Overview

This project analyzes air pollution levels in Aktobe, Kazakhstan, using **real historical monitoring data** from local stations. It identifies major contributing factors, assesses health risks, and proposes evidence-based mitigation strategies. The analysis includes time-series analysis, seasonal patterns, and comparison to WHO standards.

## Key Findings (Real Data)

Analysis of data from stations 216661 and 517420 (2021-2025) reveals:

*   **PM2.5 Annual Mean:** **13.76 μg/m³** (2.75× WHO standard of 5 μg/m³)
*   **PM10 Annual Mean:** **15.36 μg/m³** (1.02× WHO standard of 15 μg/m³)
*   **Heating Season Impact:** Pollution increases by **54%** during the heating season (Oct-Apr).
*   **WHO Exceedance:** **22.5%** of days exceed the daily PM2.5 limit.
*   **Average AQI:** **48** (Good), but with significant winter spikes.

## Project Structure

```
task1/
├── data/
│   ├── raw/                  # Original station data (HTML-wrapped CSVs)
│   ├── processed/            # Cleaned and merged datasets
│   │   └── aktobe_real_merged.csv
│   └── external/             # Contextual data
├── src/
│   ├── process_real_data.py  # Parser for raw station files
│   ├── preprocessing.py      # Cleaning and feature engineering
│   ├── analysis.py           # Statistical analysis
│   ├── visualization.py      # Chart generation
│   └── models.py             # Forecasting models (optional)
├── reports/
│   ├── figures/              # Generated visualizations
│   ├── report.md             # Final written report
│   └── presentation_outline.md # Presentation slides
├── DATA_SOURCES.md           # Documentation of data sources
├── requirements.txt          # Python dependencies
├── config.py                 # Configuration settings
└── README.md                 # This file
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ansinitro/aml.git
    cd aml/task1
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/Mac
    # venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the analysis pipeline in the following order:

1.  **Process Raw Data:**
    Parses the HTML-wrapped station files and merges them.
    ```bash
    python src/process_real_data.py
    ```

2.  **Preprocessing:**
    Cleans the merged data, handles outliers, and calculates AQI.
    ```bash
    python src/preprocessing.py
    ```

3.  **Analysis:**
    Performs statistical analysis and generates the report summary.
    ```bash
    python src/analysis.py
    ```

4.  **Visualization:**
    Generates time-series plots, heatmaps, and boxplots in `reports/figures/`.
    ```bash
    python src/visualization.py
    ```

## Deliverables

*   **[Written Report](reports/report.md):** 15-page comprehensive analysis.
*   **[Presentation](reports/presentation_outline.md):** 17-slide summary of findings.
*   **[Visualizations](reports/figures/):** High-quality charts and graphs.
*   **[Data Sources](DATA_SOURCES.md):** Detailed documentation of the monitoring stations.

## WHO Air Quality Standards (2021)

| Pollutant | Annual Mean | Daily Mean | Unit |
|-----------|-------------|------------|------|
| PM2.5 | 5 | 15 | μg/m³ |
| PM10 | 15 | 45 | μg/m³ |
| NO₂ | 10 | 25 | μg/m³ |
| SO₂ | - | 40 | μg/m³ |

## Contributors

*   **Student:** Angsar Shaumen
*   **Course:** Applied Machine Learning - Case Study Task 1
*   **Institution:** AITU (Astana IT University)
*   **Date:** December 2025

## License

This project is for educational purposes as part of the AML course at AITU.
