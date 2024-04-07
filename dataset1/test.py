import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

# Load the data
data = pd.read_csv('drug.txt', parse_dates=['date'], index_col='date')

# Initial visual inspection of the series
data.plot(figsize=(12, 6))
plt.title('Monthly Anti-Diabetic Drug Sales in Australia')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.savefig('original_series.svg', format='svg')  # Save plot
# plt.show()

# Perform Augmented Dickey-Fuller test for stationarity
adf_test = adfuller(data['value'])
print(f'ADF Statistic: {adf_test[0]}')
print(f'p-value: {adf_test[1]}')

# Determine if transformation is necessary (based on visual inspection)
data['value_log'] = np.log(data['value'])
data['value_log_diff'] = data['value_log'].diff().dropna()  # Differencing

# Plot transformed and differenced data
data[['value_log', 'value_log_diff']].plot(subplots=True, figsize=(12, 8))
plt.savefig('transformed_series.svg', format='svg')  # Save plot
# plt.show()

# Re-check stationarity with ADF on transformed, differenced data
adf_test_log_diff = adfuller(data['value_log_diff'].dropna())
print(f'ADF Statistic (log diff): {adf_test_log_diff[0]}')
print(f'p-value (log diff): {adf_test_log_diff[1]}')

# ACF and PACF plots for differenced, log-transformed series
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(data['value_log_diff'].dropna(), ax=ax[0], lags=40)
plot_pacf(data['value_log_diff'].dropna(), ax=ax[1], lags=40, method='ywm')
plt.savefig('ACF_PACF_transformed.svg', format='svg')  # Save plot
# plt.show()

# Define model configurations
arima_configs = [(1, 1, 0), (0, 1, 1), (1, 1, 1)]
sarima_configs = [((1, 1, 0), (1, 1, 1, 12)), ((0, 1, 1), (1, 1, 0, 12)), ((1, 1, 1), (0, 1, 1, 12))]

# Fit models manually and collect AIC values
results = []
for config in arima_configs:
    try:
        model = ARIMA(data['value_log_diff'], order=config)  # Use log diff series
        model_fit = model.fit()
        results.append(('ARIMA', config, model_fit.aic, model_fit))
    except Exception as e:
        print(f'Error fitting ARIMA{config}: {e}')

for config in sarima_configs:
    try:
        model = SARIMAX(data['value_log_diff'], order=config[0], seasonal_order=config[1])  # Use log diff series
        model_fit = model.fit()
        results.append(('SARIMA', config, model_fit.aic, model_fit))
    except Exception as e:
        print(f'Error fitting SARIMA{config}: {e}')

# Automatic ARIMA model selection using auto_arima on transformed, differenced data
auto_arima_model = pm.auto_arima(data['value_log_diff'], seasonal=True, m=12, trace=False,
                                 error_action='ignore', suppress_warnings=True,
                                 stepwise=True)

# Include auto_arima model in the comparison
results.append(('auto_arima', (auto_arima_model.order, auto_arima_model.seasonal_order), auto_arima_model.aic(), auto_arima_model))

# Select the best model based on AIC
results_df = pd.DataFrame(results, columns=['Model Type', 'Configuration', 'AIC', 'Model Object'])
best_model_details = results_df.sort_values(by='AIC').iloc[0]

# Save tabulated results to CSV
results_df.to_csv('model_selection_results.csv', index=False)

# Display best model information
print(f"Best Model: {best_model_details['Model Type']} - Configuration: {best_model_details['Configuration']} - AIC: {best_model_details['AIC']}")

# Forecast with the best model
forecast_steps = 36  # For 36 months ahead
forecast_index = pd.date_range(data.index[-1], periods=forecast_steps+1, closed='right', freq='MS')

# Forecasting
if best_model_details['Model Type'] == 'auto_arima':
    forecast, conf_int = best_model_details['Model Object'].predict(n_periods=forecast_steps, return_conf_int=True)
else:
    forecast_res = best_model_details['Model Object'].get_forecast(steps=forecast_steps)
    forecast = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int()

# Convert forecast and confidence intervals to original scale
forecast = np.exp(forecast.cumsum() + data['value_log'][-1])
conf_int = np.exp(conf_int.cumsum() + data['value_log'][-1])

# Plot the forecast and the original data
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['value'], label='Observed', marker='o')
plt.plot(forecast_index, forecast, label='Forecast', marker='x')
plt.fill_between(forecast_index, conf_int[:, 0], conf_int[:, 1], color='grey', alpha=0.2)
plt.title('Forecast of Monthly Anti-Diabetic Drug Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.savefig('forecast_plot.svg', format='svg')  # Save plot
# plt.show()