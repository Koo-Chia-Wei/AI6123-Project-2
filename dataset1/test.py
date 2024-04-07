import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

# Load the data
data = pd.read_csv('drug.txt', parse_dates=['date'], index_col='date')

# Visual inspection of the series
data.plot(figsize=(12, 6))
plt.title('Monthly Anti-Diabetic Drug Sales in Australia')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()

# Perform Augmented Dickey-Fuller test
adf_test = adfuller(data['value'])
print('ADF Statistic: %f' % adf_test[0])
print('p-value: %f' % adf_test[1])

# ACF and PACF plots for initial analysis
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(data['value'], ax=ax[0], lags=40)
plot_pacf(data['value'], ax=ax[1], lags=40, method='ywm')
plt.show()

# Define ARIMA and SARIMA configurations
arima_configs = [(1, 1, 0), (0, 1, 1), (1, 1, 1)]
sarima_configs = [((1, 1, 0), (1, 1, 1, 12)), ((0, 1, 1), (1, 1, 0, 12)), ((1, 1, 1), (0, 1, 1, 12))]

# Fit models manually and collect AIC values
results = []
for config in arima_configs:
    try:
        model = ARIMA(data['value'], order=config)
        model_fit = model.fit()
        results.append(('ARIMA', config, model_fit.aic, model_fit))
    except:
        continue

for config in sarima_configs:
    try:
        model = SARIMAX(data['value'], order=config[0], seasonal_order=config[1])
        model_fit = model.fit()
        results.append(('SARIMA', config, model_fit.aic, model_fit))
    except:
        continue

# Automatic ARIMA model selection using auto_arima
auto_arima_model = pm.auto_arima(data['value'], seasonal=True, m=12, trace=False,
                                 error_action='ignore', suppress_warnings=True,
                                 stepwise=True)

# Include auto_arima model in the comparison
results.append(('auto_arima', (auto_arima_model.order, auto_arima_model.seasonal_order), auto_arima_model.aic(), auto_arima_model))

# Select the best model based on AIC
results_df = pd.DataFrame(results, columns=['Model Type', 'Configuration', 'AIC', 'Model Object'])
best_model_details = results_df.sort_values(by='AIC').iloc[0]
best_model = best_model_details['Model Object']

# Display best model information
print(f"Best Model: {best_model_details['Model Type']} - Configuration: {best_model_details['Configuration']} - AIC: {best_model_details['AIC']}")

# Forecast with the best model
forecast_steps = 36  # Example: forecast 36 months ahead
forecast_index = pd.date_range(data.index[-1], periods=forecast_steps+1, closed='right', freq='MS')

# If the best model is from auto_arima, use predict
if best_model_details['Model Type'] == 'auto_arima':
    forecast = best_model.predict(n_periods=forecast_steps)
    conf_int = best_model.predict(n_periods=forecast_steps, return_conf_int=True)[1]
else:  # For ARIMA/SARIMA models
    forecast_res = best_model.get_forecast(steps=forecast_steps)
    forecast = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int()

# Plot the forecast and the original data
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['value'], label='Observed', marker='o')
plt.plot(forecast_index, forecast, label='Forecast', marker='x')
plt.fill_between(forecast_index, conf_int[:, 0], conf_int[:, 1], color='grey', alpha=0.2)
plt.title('Forecast of Monthly Anti-Diabetic Drug Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()
