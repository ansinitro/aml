
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR
from statsmodels.datasets import get_rdataset, macrodata
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Define Figure Save Path
FIGURES_DIR = 'figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set style for "serious" academic look
plt.style.use('bmh')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def save_fig(name):
    """Save current figure to the figures directory."""
    path = os.path.join(FIGURES_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    print(f"Saved figure: {path}")
    plt.close()

def adf_test(series, name="Series"):
    """Perform Augmented Dickey-Fuller test."""
    print(f"\n--- ADF Test: {name} ---")
    result = adfuller(series.dropna())
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.3f}')
    
    if result[1] < 0.05:
        print("=> Stationary (Reject H0)")
        return True
    else:
        print("=> Non-Stationary (Fail to reject H0)")
        return False

def evaluate_forecast(y_true, y_pred, model_name):
    """Calculate and print RMSE and MAE."""
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    # Handle potential zero division in MAPE
    y_true_safe = np.where(y_true == 0, 1e-8, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    
    print(f"\nPerformance ({model_name}):")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    return rmse, mae, mape

# ==========================================
# PART A: ARIMA (Univariate)
# ==========================================
def part_a_arima():
    print("\n" + "="*40)
    print("PART A: ARIMA (AirPassengers)")
    print("="*40)
    
    # 1. Load Data
    data = get_rdataset('AirPassengers').data
    # Create valid date index
    data.index = pd.date_range(start='1949-01-01', periods=len(data), freq='M')
    series = data['value']
    
    # 2. Plot Original
    plt.figure()
    plt.plot(series, label='AirPassengers')
    plt.title('AirPassengers Time Series (1949-1960)')
    plt.legend()
    save_fig('part_a_1_series.png')
    
    # 3. Stationarity (ADF)
    adf_test(series, "Original")
    
    # Differencing
    series_diff = series.diff().dropna()
    adf_test(series_diff, "1st Difference")
    
    plt.figure()
    plt.plot(series_diff, color='orange', label='1st Diff')
    plt.title('Differenced Series')
    plt.legend()
    save_fig('part_a_2_diff.png')
    
    # 4. ACF/PACF
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(series_diff, lags=40, ax=ax[0])
    plot_pacf(series_diff, lags=40, ax=ax[1])
    plt.suptitle('ACF and PACF of Differenced Series')
    save_fig('part_a_3_acf_pacf.png')
    
    # 5. Fit ARIMA
    # Using p=2, d=1, q=2 as example
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]
    
    print("\nFitting ARIMA(2,1,2)...")
    model = ARIMA(train, order=(2,1,2))
    model_fit = model.fit()
    print(model_fit.summary())
    
    # Diagnostics
    model_fit.plot_diagnostics(figsize=(12, 8))
    save_fig('part_a_4_diagnostics.png')
    
    # 6. Forecast
    forecast_res = model_fit.get_forecast(steps=len(test))
    pred = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int()
    
    # Evaluate
    evaluate_forecast(test, pred, "ARIMA(2,1,2)")
    
    # Plot forecast
    plt.figure()
    plt.plot(train.index, train, label='Train')
    plt.plot(test.index, test, label='Test', color='green')
    plt.plot(pred.index, pred, label='Forecast', color='red', linestyle='--')
    plt.fill_between(conf_int.index, conf_int.iloc[:,0], conf_int.iloc[:,1], color='pink', alpha=0.3)
    plt.title('ARIMA Forecast vs Actual')
    plt.legend()
    save_fig('part_a_5_forecast.png')

# ==========================================
# PART B: SARIMA (Seasonal)
# ==========================================
def part_b_sarima():
    print("\n" + "="*40)
    print("PART B: SARIMA (AirPassengers)")
    print("="*40)
    
    # Load same data
    data = get_rdataset('AirPassengers').data
    data.index = pd.date_range(start='1949-01-01', periods=len(data), freq='M')
    series = data['value']
    
    # Log transform to stabilize variance
    series_log = np.log(series)
    
    train_size = int(len(series_log) * 0.8)
    train, test = series_log[:train_size], series_log[train_size:]
    
    # Fit SARIMA(1,1,1)(1,1,1,12)
    # Seasonality s=12 for monthly data
    print("\nFitting SARIMA(1,1,1)x(1,1,1,12)...")
    model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
    model_fit = model.fit(disp=False)
    print(model_fit.summary())
    
    # Forecast
    forecast_res = model_fit.get_forecast(steps=len(test))
    pred_log = forecast_res.predicted_mean
    conf_int_log = forecast_res.conf_int()
    
    # Convert back to original scale
    pred = np.exp(pred_log)
    test_orig = np.exp(test)
    train_orig = np.exp(train)
    lb = np.exp(conf_int_log.iloc[:,0])
    ub = np.exp(conf_int_log.iloc[:,1])
    
    evaluate_forecast(test_orig, pred, "SARIMA")
    
    plt.figure()
    plt.plot(train_orig.index, train_orig, label='Train')
    plt.plot(test_orig.index, test_orig, label='Test', color='green')
    plt.plot(pred.index, pred, label='Forecast', color='red', linestyle='--')
    plt.fill_between(pred.index, lb, ub, color='pink', alpha=0.3)
    plt.title('SARIMA Forecast vs Actual (Back-transformed)')
    plt.legend()
    save_fig('part_b_1_forecast.png')

# ==========================================
# PART C: ARIMAX (Exogenous)
# ==========================================
def part_c_arimax():
    print("\n" + "="*40)
    print("PART C: ARIMAX (Macrodata)")
    print("="*40)
    
    # Load Macrodata
    dta = macrodata.load_pandas().data
    dta.index = pd.Index(pd.date_range('1959-01-01', periods=len(dta), freq='Q'))
    
    # Target: Real GDP growth, Exog: Real Consumption
    # We differencing to handle stationarity
    target = dta['realgdp'].diff().dropna()
    exog = dta['realcons'].diff().dropna()
    
    # Align indices
    common_idx = target.index.intersection(exog.index)
    target = target.loc[common_idx]
    exog = exog.loc[common_idx]
    
    train_size = int(len(target) * 0.9)
    train_y, test_y = target[:train_size], target[train_size:]
    train_X, test_X = exog[:train_size], exog[train_size:]
    
    print("\nFitting ARIMAX(1,0,1) with Exogenous variable (Real Cons)...")
    # Using ARIMA with exog argument
    model = ARIMA(train_y, exog=train_X, order=(1,0,1))
    model_fit = model.fit()
    print(model_fit.summary())
    
    # Forecast
    forecast_res = model_fit.get_forecast(steps=len(test_y), exog=test_X)
    pred = forecast_res.predicted_mean
    
    evaluate_forecast(test_y, pred, "ARIMAX")
    
    plt.figure()
    plt.plot(train_y.index[-50:], train_y[-50:], label='Train (Last 50)')
    plt.plot(test_y.index, test_y, label='Test', color='green')
    plt.plot(pred.index, pred, label='Forecast', color='red', linestyle='--')
    plt.title('ARIMAX Forecast (Real GDP Growth)')
    plt.legend()
    save_fig('part_c_1_forecast.png')

# ==========================================
# PART D: VAR (Multivariate)
# ==========================================
def part_d_var():
    print("\n" + "="*40)
    print("PART D: VAR (Macrodata)")
    print("="*40)
    
    # Load data
    dta = macrodata.load_pandas().data
    dta.index = pd.Index(pd.date_range('1959-01-01', periods=len(dta), freq='Q'))
    
    # Select variables: Real GDP, Real Cons, Real Inv
    cols = ['realgdp', 'realcons', 'realinv']
    df = dta[cols]
    
    # Differencing for stationarity (log-diff often better for growth rates, but simple diff here)
    df_diff = np.log(df).diff().dropna()
    
    # Split
    train_size = int(len(df_diff) * 0.9)
    train, test = df_diff[:train_size], df_diff[train_size:]
    
    # Select Order
    model = VAR(train)
    lag_order_res = model.select_order(maxlags=10)
    print(lag_order_res.summary())
    
    # Fit VAR
    best_aic = lag_order_res.selected_orders['aic']
    print(f"\nFitting VAR({best_aic})...")
    results = model.fit(best_aic)
    print(results.summary())
    
    # Forecast
    lag_order = results.k_ar
    forecast_input = train.values[-lag_order:]
    fc = results.forecast(y=forecast_input, steps=len(test))
    df_forecast = pd.DataFrame(fc, index=test.index, columns=cols)
    
    # Plot 'realgdp' forecast
    plt.figure()
    plt.plot(train.index[-50:], train['realgdp'][-50:], label='Train (GDP)')
    plt.plot(test.index, test['realgdp'], label='Test (GDP)', color='green')
    plt.plot(df_forecast.index, df_forecast['realgdp'], label='Forecast (GDP)', color='red', linestyle='--')
    plt.title('VAR Forecast: Scaled Log-Diff GDP')
    plt.legend()
    save_fig('part_d_1_forecast_gdp.png')
    
    # IRF (Impulse Response)
    irf = results.irf(10)
    plt.figure()
    irf.plot(orth=False, figsize=(10,8)) # statsmodels plots it directly
    # The plot() method in statsmodels shows the figure, we need to save the current figure
    # Note: irf.plot() creates a new figure usually. 
    # We'll rely on the fact that it plots.
    # Actually, irf.plot() returns a fig object.
    # Let's handle it carefully.
    plt.close() # Close any previous
    fig_irf = irf.plot(orth=False)
    fig_irf.suptitle('Impulse Response Functions')
    fig_irf.set_size_inches(12, 12)
    plt.tight_layout() 
    plt.savefig(os.path.join(FIGURES_DIR, 'part_d_2_irf.png'))
    print(f"Saved figure: {os.path.join(FIGURES_DIR, 'part_d_2_irf.png')}")
    plt.close(fig_irf)

from statsmodels.tsa.statespace.varmax import VARMAX

def part_d_varmax_comparison():
    print("\n" + "="*40)
    print("PART D (Extension): VARMAX Comparison")
    print("="*40)
    
    # Load same data as VAR
    dta = macrodata.load_pandas().data
    dta.index = pd.Index(pd.date_range('1959-01-01', periods=len(dta), freq='Q'))
    cols = ['realgdp', 'realcons', 'realinv']
    df_diff = np.log(dta[cols]).diff().dropna()
    
    train_size = int(len(df_diff) * 0.9)
    train, test = df_diff[:train_size], df_diff[train_size:]
    
    # Fit VARMAX(1,1) - VAR with Moving Average component
    # CAUTION: VARMA is computationally intensive and hard to identify. 
    # We will try a simple (1,1) order.
    print("\nFitting VARMAX(1,1)... this may take a moment...")
    model = VARMAX(train, order=(1,1))
    
    # maxiter limit to prevent long runtimes in lab
    results = model.fit(maxiter=100, disp=False)
    print(results.summary())
    
    # Forecast
    forecast_res = results.get_forecast(steps=len(test))
    pred = forecast_res.predicted_mean
    
    # Compare with Test
    print("\nPerformance (VARMAX(1,1) - GDP):")
    rmse_gdp = math.sqrt(mean_squared_error(test['realgdp'], pred['realgdp']))
    print(f"RMSE (GDP): {rmse_gdp:.4f}")
    
    plt.figure()
    plt.plot(train.index[-50:], train['realgdp'][-50:], label='Train (GDP)')
    plt.plot(test.index, test['realgdp'], label='Test (GDP)', color='green')
    plt.plot(pred.index, pred['realgdp'], label='Forecast (VARMAX)', color='purple', linestyle='--')
    plt.title('VARMAX(1,1) Forecast: GDP')
    plt.legend()
    save_fig('part_d_3_varmax_gdp.png')

def main():
    print("Starting Lab 2 Implementation...")
    part_a_arima()
    part_b_sarima()
    part_c_arimax()
    part_d_var()
    part_d_varmax_comparison()
    print("\nDone! All figures saved to 'figures/' directory.")

if __name__ == "__main__":
    main()
