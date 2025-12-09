"""
Machine Learning Models Module
Forecasting and anomaly detection for air quality data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def prepare_time_series_data(df, target_col='PM2.5', lookback=7):
    """
    Prepare data for time series forecasting
    
    Parameters:
    -----------
    df : DataFrame
        Input data with date index
    target_col : str
        Column to forecast
    lookback : int
        Number of past days to use as features
    
    Returns:
    --------
    X, y arrays for training
    """
    df_sorted = df.sort_values('date').copy()
    
    # Create lagged features
    for i in range(1, lookback + 1):
        df_sorted[f'{target_col}_lag_{i}'] = df_sorted[target_col].shift(i)
    
    # Drop rows with NaN (from shifting)
    df_sorted = df_sorted.dropna()
    
    # Features and target
    feature_cols = [f'{target_col}_lag_{i}' for i in range(1, lookback + 1)]
    
    # Add temporal features if available
    if 'month' in df_sorted.columns:
        feature_cols.append('month')
    if 'day_of_week' in df_sorted.columns:
        feature_cols.append('day_of_week')
    if 'is_heating_season' in df_sorted.columns:
        df_sorted['is_heating_season_int'] = df_sorted['is_heating_season'].astype(int)
        feature_cols.append('is_heating_season_int')
    
    X = df_sorted[feature_cols].values
    y = df_sorted[target_col].values
    
    return X, y, df_sorted


def train_arima_model(df, target_col='PM2.5', order=(5, 1, 2)):
    """
    Train ARIMA model for forecasting
    
    Parameters:
    -----------
    df : DataFrame
        Input data
    target_col : str
        Column to forecast
    order : tuple
        ARIMA order (p, d, q)
    
    Returns:
    --------
    Fitted ARIMA model
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        print("statsmodels not installed. Install with: pip install statsmodels")
        return None
    
    # Prepare data
    df_sorted = df.sort_values('date').copy()
    data = df_sorted[target_col].dropna()
    
    # Split train/test
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]
    
    # Fit model
    print(f"Training ARIMA{order} model...")
    model = ARIMA(train, order=order)
    fitted_model = model.fit()
    
    # Forecast
    forecast = fitted_model.forecast(steps=len(test))
    
    # Evaluate
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)
    
    print(f"ARIMA Model Performance:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    
    return fitted_model, forecast, test


def train_prophet_model(df, target_col='PM2.5'):
    """
    Train Prophet model for forecasting with seasonality
    
    Parameters:
    -----------
    df : DataFrame
        Input data with 'date' column
    target_col : str
        Column to forecast
    
    Returns:
    --------
    Fitted Prophet model
    """
    try:
        from prophet import Prophet
    except ImportError:
        print("Prophet not installed. Install with: pip install prophet")
        return None
    
    # Prepare data for Prophet (requires 'ds' and 'y' columns)
    df_prophet = df[['date', target_col]].copy()
    df_prophet.columns = ['ds', 'y']
    df_prophet = df_prophet.dropna()
    
    # Split train/test
    train_size = int(len(df_prophet) * 0.8)
    train = df_prophet[:train_size]
    test = df_prophet[train_size:]
    
    # Initialize and fit model
    print("Training Prophet model...")
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    
    model.fit(train)
    
    # Make predictions
    future = model.make_future_dataframe(periods=len(test))
    forecast = model.predict(future)
    
    # Evaluate on test set
    test_forecast = forecast.iloc[train_size:]['yhat'].values
    test_actual = test['y'].values
    
    rmse = np.sqrt(mean_squared_error(test_actual, test_forecast))
    mae = mean_absolute_error(test_actual, test_forecast)
    
    print(f"Prophet Model Performance:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    
    return model, forecast


def detect_anomalies(df, target_cols=None, contamination=0.05):
    """
    Detect anomalies using Isolation Forest
    
    Parameters:
    -----------
    df : DataFrame
        Input data
    target_cols : list
        Columns to use for anomaly detection
    contamination : float
        Expected proportion of outliers
    
    Returns:
    --------
    DataFrame with anomaly labels
    """
    if target_cols is None:
        target_cols = [p for p in config.POLLUTANTS if p in df.columns]
    
    # Prepare data
    df_clean = df[target_cols].dropna()
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    
    # Fit Isolation Forest
    print(f"Detecting anomalies with contamination={contamination}...")
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=config.RANDOM_SEED,
        n_estimators=100
    )
    
    predictions = iso_forest.fit_predict(X_scaled)
    
    # Add to dataframe
    df_result = df.copy()
    df_result['anomaly'] = 0
    df_result.loc[df_clean.index, 'anomaly'] = predictions
    df_result['is_anomaly'] = df_result['anomaly'] == -1
    
    n_anomalies = (df_result['is_anomaly']).sum()
    print(f"Detected {n_anomalies} anomalies ({n_anomalies/len(df_result)*100:.2f}%)")
    
    return df_result, iso_forest


def evaluate_forecast(actual, predicted):
    """
    Calculate forecast evaluation metrics
    
    Returns:
    --------
    Dictionary with RMSE, MAE, MAPE, R²
    """
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # R²
    r2 = r2_score(actual, predicted)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }


def forecast_future(model, periods=30, model_type='prophet'):
    """
    Forecast future pollution levels
    
    Parameters:
    -----------
    model : fitted model
        Trained forecasting model
    periods : int
        Number of days to forecast
    model_type : str
        'prophet' or 'arima'
    
    Returns:
    --------
    Forecast dataframe
    """
    if model_type == 'prophet':
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    elif model_type == 'arima':
        forecast = model.forecast(steps=periods)
        return forecast
    
    else:
        print(f"Unknown model type: {model_type}")
        return None


class SimpleMLPredictor:
    """
    Simple ML predictor using scikit-learn
    """
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        
    def train(self, X_train, y_train):
        """Train the model"""
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import Ridge
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Initialize model
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=config.RANDOM_SEED,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=config.RANDOM_SEED
            )
        elif self.model_type == 'ridge':
            self.model = Ridge(alpha=1.0)
        
        # Train
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train_scaled, y_train)
        
    def predict(self, X_test):
        """Make predictions"""
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        metrics = evaluate_forecast(y_test, predictions)
        
        print(f"\n{self.model_type.upper()} Model Performance:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.2f}")
        
        return metrics


if __name__ == "__main__":
    print("ML Models module loaded successfully")
    print("Available functions:")
    print("  - train_arima_model()")
    print("  - train_prophet_model()")
    print("  - detect_anomalies()")
    print("  - SimpleMLPredictor class")
