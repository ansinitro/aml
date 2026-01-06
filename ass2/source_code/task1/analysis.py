"""
Analysis Module
Statistical analysis, time-series decomposition, seasonal patterns, correlations
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_processed_data(filename='aktobe_processed.csv'):
    """Load processed data"""
    filepath = os.path.join(config.DATA_DIRS['processed'], filename)
    df = pd.read_csv(filepath, parse_dates=['date'])
    return df


def descriptive_statistics(df, pollutants=None):
    """
    Calculate descriptive statistics for pollutants
    
    Returns:
    --------
    DataFrame with mean, median, std, min, max, etc.
    """
    if pollutants is None:
        pollutants = config.POLLUTANTS
    
    # Filter to only existing columns
    pollutants = [p for p in pollutants if p in df.columns]
    
    stats_df = df[pollutants].describe()
    
    # Add additional statistics
    stats_df.loc['skewness'] = df[pollutants].skew()
    stats_df.loc['kurtosis'] = df[pollutants].kurtosis()
    
    return stats_df


def seasonal_analysis(df, pollutant='PM2.5', period=365):
    """
    Perform seasonal decomposition on time series
    
    Parameters:
    -----------
    df : DataFrame
        Data with date index
    pollutant : str
        Pollutant to analyze
    period : int
        Seasonal period (365 for yearly, 30 for monthly)
    
    Returns:
    --------
    Decomposition result with trend, seasonal, and residual components
    """
    if pollutant not in df.columns:
        print(f"Pollutant {pollutant} not found in data")
        return None
    
    # Set date as index
    df_temp = df.set_index('date')
    
    # Ensure regular frequency
    df_temp = df_temp.asfreq('D')
    
    # Fill any gaps
    df_temp[pollutant] = df_temp[pollutant].interpolate()
    
    # Perform decomposition
    try:
        decomposition = seasonal_decompose(
            df_temp[pollutant],
            model='additive',
            period=period,
            extrapolate_trend='freq'
        )
        return decomposition
    except Exception as e:
        print(f"Error in seasonal decomposition: {e}")
        return None


def monthly_seasonal_pattern(df, pollutants=None):
    """
    Analyze monthly patterns for pollutants
    
    Returns:
    --------
    DataFrame with monthly averages
    """
    if pollutants is None:
        pollutants = config.POLLUTANTS
    
    pollutants = [p for p in pollutants if p in df.columns]
    
    monthly_avg = df.groupby('month')[pollutants].mean()
    monthly_avg.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    return monthly_avg


def seasonal_comparison(df, pollutants=None):
    """
    Compare pollution levels across seasons
    
    Returns:
    --------
    DataFrame with seasonal averages
    """
    if pollutants is None:
        pollutants = config.POLLUTANTS
    
    pollutants = [p for p in pollutants if p in df.columns]
    
    seasonal_avg = df.groupby('season')[pollutants].mean()
    
    # Reorder seasons
    season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
    seasonal_avg = seasonal_avg.reindex([s for s in season_order if s in seasonal_avg.index])
    
    return seasonal_avg


def heating_season_comparison(df, pollutants=None):
    """
    Compare pollution during heating vs non-heating season
    """
    if pollutants is None:
        pollutants = config.POLLUTANTS
    
    pollutants = [p for p in pollutants if p in df.columns]
    
    comparison = df.groupby('is_heating_season')[pollutants].mean()
    comparison.index = ['Non-Heating Season', 'Heating Season']
    
    return comparison


def correlation_analysis(df, pollutants=None, weather_vars=None):
    """
    Analyze correlations between pollutants and weather variables
    
    Returns:
    --------
    Correlation matrix
    """
    if pollutants is None:
        pollutants = config.POLLUTANTS
    
    if weather_vars is None:
        weather_vars = ['temperature', 'humidity', 'wind_speed']
    
    # Filter to existing columns
    pollutants = [p for p in pollutants if p in df.columns]
    weather_vars = [w for w in weather_vars if w in df.columns]
    
    all_vars = pollutants + weather_vars
    
    corr_matrix = df[all_vars].corr()
    
    return corr_matrix


def who_exceedance_analysis(df):
    """
    Analyze exceedance of WHO standards
    
    Returns:
    --------
    Dictionary with exceedance statistics
    """
    results = {}
    
    for pollutant in config.POLLUTANTS:
        if pollutant not in df.columns:
            continue
        
        if pollutant not in config.WHO_STANDARDS:
            continue
        
        standards = config.WHO_STANDARDS[pollutant]
        
        # Annual mean comparison
        if 'annual' in standards:
            annual_mean = df[pollutant].mean()
            results[f'{pollutant}_annual_mean'] = annual_mean
            results[f'{pollutant}_WHO_annual'] = standards['annual']
            results[f'{pollutant}_exceeds_annual'] = annual_mean > standards['annual']
            results[f'{pollutant}_annual_ratio'] = annual_mean / standards['annual']
        
        # Daily exceedance count
        if 'daily' in standards:
            exceedance_col = f'{pollutant}_exceeds_WHO_daily'
            if exceedance_col in df.columns:
                exceedance_days = df[exceedance_col].sum()
                total_days = len(df)
                results[f'{pollutant}_exceedance_days'] = exceedance_days
                results[f'{pollutant}_exceedance_percentage'] = (exceedance_days / total_days) * 100
    
    return results


def trend_analysis(df, pollutant='PM2.5'):
    """
    Perform Mann-Kendall trend test
    
    Returns:
    --------
    Dictionary with trend statistics
    """
    if pollutant not in df.columns:
        return None
    
    # Sort by date
    df_sorted = df.sort_values('date')
    data = df_sorted[pollutant].dropna().values
    
    # Mann-Kendall test
    n = len(data)
    s = 0
    
    for i in range(n-1):
        for j in range(i+1, n):
            s += np.sign(data[j] - data[i])
    
    # Variance
    var_s = n * (n - 1) * (2 * n + 5) / 18
    
    # Z-score
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0
    
    # P-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    # Trend direction
    if p_value < 0.05:
        if s > 0:
            trend = 'Increasing'
        else:
            trend = 'Decreasing'
    else:
        trend = 'No significant trend'
    
    return {
        'pollutant': pollutant,
        'Mann-Kendall S': s,
        'Z-score': z,
        'p-value': p_value,
        'trend': trend,
        'significant': p_value < 0.05
    }


def weekend_effect_analysis(df, pollutants=None):
    """
    Analyze difference between weekday and weekend pollution
    """
    if pollutants is None:
        pollutants = config.POLLUTANTS
    
    pollutants = [p for p in pollutants if p in df.columns]
    
    comparison = df.groupby('is_weekend')[pollutants].mean()
    comparison.index = ['Weekday', 'Weekend']
    
    # Calculate percentage difference
    pct_diff = ((comparison.loc['Weekend'] - comparison.loc['Weekday']) / 
                comparison.loc['Weekday'] * 100)
    
    return comparison, pct_diff


def aqi_distribution_analysis(df):
    """
    Analyze AQI distribution and categories
    """
    if 'AQI' not in df.columns:
        return None
    
    results = {
        'mean_AQI': df['AQI'].mean(),
        'median_AQI': df['AQI'].median(),
        'max_AQI': df['AQI'].max(),
        'min_AQI': df['AQI'].min(),
        'std_AQI': df['AQI'].std(),
    }
    
    if 'AQI_category' in df.columns:
        category_counts = df['AQI_category'].value_counts()
        category_pct = (category_counts / len(df) * 100).round(2)
        
        results['category_distribution'] = category_counts.to_dict()
        results['category_percentage'] = category_pct.to_dict()
    
    return results


def generate_analysis_report(df):
    """
    Generate comprehensive analysis report
    """
    print("=" * 70)
    print("AIR QUALITY ANALYSIS REPORT - AKTOBE, KAZAKHSTAN")
    print("=" * 70)
    
    # Descriptive statistics
    print("\n1. DESCRIPTIVE STATISTICS")
    print("-" * 70)
    stats_df = descriptive_statistics(df)
    print(stats_df.round(2))
    
    # Seasonal patterns
    print("\n2. MONTHLY SEASONAL PATTERNS")
    print("-" * 70)
    monthly = monthly_seasonal_pattern(df)
    print(monthly.round(2))
    
    # Seasonal comparison
    print("\n3. SEASONAL COMPARISON")
    print("-" * 70)
    seasonal = seasonal_comparison(df)
    print(seasonal.round(2))
    
    # Heating season
    print("\n4. HEATING SEASON COMPARISON")
    print("-" * 70)
    heating = heating_season_comparison(df)
    print(heating.round(2))
    
    # WHO exceedance
    print("\n5. WHO STANDARDS EXCEEDANCE")
    print("-" * 70)
    who_results = who_exceedance_analysis(df)
    for key, value in who_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # AQI distribution
    print("\n6. AQI DISTRIBUTION")
    print("-" * 70)
    aqi_results = aqi_distribution_analysis(df)
    if aqi_results:
        for key, value in aqi_results.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value:.2f}")
    
    # Trend analysis
    print("\n7. TREND ANALYSIS")
    print("-" * 70)
    for pollutant in ['PM2.5', 'PM10', 'NO2']:
        if pollutant in df.columns:
            trend = trend_analysis(df, pollutant)
            if trend:
                print(f"\n{pollutant}:")
                print(f"  Trend: {trend['trend']}")
                print(f"  Z-score: {trend['Z-score']:.3f}")
                print(f"  p-value: {trend['p-value']:.4f}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Example usage
    df = load_processed_data('aktobe_real_processed.csv')
    generate_analysis_report(df)
