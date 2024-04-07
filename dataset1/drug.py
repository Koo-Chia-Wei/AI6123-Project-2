import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
import os

# Resolve path of the current file
file_path = os.path.abspath(__file__)

# Extract the directory in which the file is located
file_dir = os.path.dirname(file_path)

# The directory where you want to save the plot
save_dir = os.path.join(file_dir, "image")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Load the dataset
data = pd.read_csv(
    os.path.join(os.path.dirname(file_path), "drug.txt"),
    parse_dates=["date"],
    index_col="date",
)

# Setting the frequency explicitly
data.index.freq = "MS"

# Initial visual inspection of the series
data.plot(figsize=(12, 6))
plt.title("Monthly Anti-Diabetic Drug Sales in Australia")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.savefig(os.path.join(save_dir, "drug_sales.pdf"), format="pdf")  # Save plot
# plt.show()

# Perform Augmented Dickey-Fuller test for stationarity
adf_test = adfuller(data["value"])
print(f"ADF Statistic: {adf_test[0]}")
print(f"p-value: {adf_test[1]}")

# ACF and PACF plots for the original series
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(data["value"], ax=ax[0], lags=40)
plot_pacf(data["value"], ax=ax[1], lags=40, method="ywm")
fig.suptitle("ACF and PACF plots for original Drug Sales data", fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(save_dir, "ACF_PACF_original_drug_sales.pdf"), format="pdf")  # Save plot
# plt.show()

# Determine if transformation is necessary (based on visual inspection)
data["value_log"] = np.log(data["value"])
data["value_log_diff"] = data["value_log"].diff().dropna()  # Differencing
data.dropna(inplace=True)

# Plot transformed and differenced data
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
data["value_log"].plot(ax=axes[0], title="Log-transformed Data")
axes[0].set_ylabel("Log(Sales)")
axes[0].set_xlabel("Date")

data["value_log_diff"].plot(ax=axes[1], title="Differenced Log-transformed Data")
axes[1].set_ylabel("Differenced Log(Sales)")
axes[1].set_xlabel("Date")

plt.tight_layout()  # Adjust layout to make room for the titles
plt.savefig(os.path.join(save_dir, "log_transformed_diff_drug_sales.pdf"), format="pdf")  # Save plot
# plt.show()

# Re-check stationarity with ADF on transformed, differenced data
adf_test_log = adfuller(data["value_log"].dropna())
print(f"ADF Statistic (log): {adf_test_log[0]}")
print(f"p-value (log): {adf_test_log[1]}")
adf_test_log_diff = adfuller(data["value_log_diff"].dropna())
print(f"ADF Statistic (log diff): {adf_test_log_diff[0]}")
print(f"p-value (log diff): {adf_test_log_diff[1]}")

# ACF and PACF plots for log-transformed series
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(data["value_log"].dropna(), ax=ax[0], lags=40)
plot_pacf(data["value_log"].dropna(), ax=ax[1], lags=40, method="ywm")
fig.suptitle("ACF and PACF plots for log-transformed Drug Sales data", fontsize=16)
plt.savefig(
    os.path.join(save_dir, "ACF_PACF_log_transformed_drug_sales.pdf"), format="pdf"
)  # Save plot
# plt.show()

# ACF and PACF plots for differenced, log-transformed series
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(data["value_log_diff"].dropna(), ax=ax[0], lags=40)
plot_pacf(data["value_log_diff"].dropna(), ax=ax[1], lags=40, method="ywm")
fig.suptitle("ACF and PACF plots for differenced, log-transformed Drug Sales data", fontsize=16)
plt.savefig(
    os.path.join(save_dir, "ACF_PACF_diff_log_transformed_drug_sales.pdf"), format="pdf"
)  # Save plot
# plt.show()

# Define ARIMA and SARIMA model configurations
arima_configs = [
    (1, 1, 0),  # AR(1) model with differencing, suggested by PACF cut-off at lag 1
    (0, 1, 1),  # MA(1) model with differencing, suggested by ACF cut-off at lag 1
    (0, 0, 1),  # MA(1) model without differencing, for potential mild non-stationarity
    (1, 1, 1),  # ARMA(1,1) model with differencing, general model incorporating both AR and MA
    (1, 0, 0),  # Simple AR(1) model, capturing potential autoregressive behavior without differencing
    # Add new ARIMA configs here
]

sarima_configs = [
    (
        (1, 1, 0),
        (1, 1, 1, 12),
    ),  # Incorporates both non-seasonal and seasonal components
    (
        (0, 1, 1),
        (1, 1, 0, 12),
    ),  # Seasonal MA component, assumes non-seasonal MA process with seasonal differencing
    (
        (1, 1, 1),
        (0, 1, 1, 12),
    ),  # Non-seasonal ARMA(1,1) with seasonal MA(1), for complex seasonality
    (
        (1, 0, 0),
        (0, 1, 1, 12),
    ),  # Non-seasonal AR(1) model with seasonal MA(1), addresses seasonal autocorrelations
    (
        (1, 0, 0),
        (1, 1, 1, 12),
    ),  # AR(1) with both non-seasonal and seasonal differencing and MA(1), for clear seasonal patterns
    # Add new SARIMA configs here
]

# Fit models manually and collect AIC values
results = []
for config in arima_configs:
    try:
        model = ARIMA(data["value_log_diff"], order=config)  # Use log diff series
        model_fit = model.fit()
        results.append(("ARIMA", config, model_fit.aic, model_fit))
        # Diagnostic plot for the fitted model
        model_fit.plot_diagnostics(figsize=(12, 8))
        plt.suptitle(f'Diagnostic Plot for ARIMA{config} Model', fontsize=16)
        plt.savefig(os.path.join(save_dir, f"diagnostic_plot_ARIMA{config}_drug_sales.pdf"), format="pdf")  # Save plot
        # plt.show()
    except Exception as e:
        print(f"Error fitting ARIMA{config}: {e}")

for config in sarima_configs:
    try:
        model = SARIMAX(
            data["value_log_diff"], order=config[0], seasonal_order=config[1]
        )  # Use log diff series
        model_fit = model.fit()
        results.append(("SARIMA", config, model_fit.aic, model_fit))
        # Diagnostic plot for the fitted model
        model_fit.plot_diagnostics(figsize=(12, 8))
        plt.suptitle(f'Diagnostic Plot for SARIMA{config} Model', fontsize=16)
        plt.savefig(os.path.join(save_dir, f"diagnostic_plot_SARIMA{config}_drug_sales.pdf"), format="pdf")  # Save plot
        # plt.show()
    except Exception as e:
        print(f"Error fitting SARIMA{config}: {e}")

############# do a diagnostic plot for each model

# Automatic ARIMA model selection using auto_arima on transformed, differenced data
auto_arima_model = pm.auto_arima(
    data["value_log_diff"],
    seasonal=True,
    m=12,
    trace=False,
    error_action="ignore",
    suppress_warnings=True,
    stepwise=True,
)

# Include auto_arima model in the comparison
results.append(
    (
        "auto_arima",
        (auto_arima_model.order, auto_arima_model.seasonal_order),
        auto_arima_model.aic(),
        auto_arima_model,
    )
)

# Select the best model based on AIC
results_df = pd.DataFrame(
    results, columns=["Model Type", "Configuration", "AIC", "Model Object"]
)
best_model_details = results_df.sort_values(by="AIC").iloc[0]

# Save tabulated results to CSV
results_df.to_csv(os.path.join(file_dir, "model_selection_results.csv"), index=False)

# Display best model information
print(
    f"Best Model: {best_model_details['Model Type']} - Configuration: {best_model_details['Configuration']} - AIC: {best_model_details['AIC']}"
)

# Forecast with the best model
forecast_steps = 36  # For 36 months ahead
last_date = data.index[-1]
forecast_index = pd.date_range(
    start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq="M"
)

# Forecasting
if best_model_details["Model Type"] == "auto_arima":
    # For auto_arima, directly get the forecast and confidence intervals
    forecast, conf_int = best_model_details["Model Object"].predict(
        n_periods=forecast_steps, return_conf_int=True
    )
    # Adjust the forecast and confidence intervals back to the original scale if necessary
    forecast = np.exp(forecast + data["value_log"].iloc[-1])
    conf_int_lower = np.exp(conf_int[:, 0] + data["value_log"].iloc[-1])
    conf_int_upper = np.exp(conf_int[:, 1] + data["value_log"].iloc[-1])
else:
    # For manual ARIMA/SARIMA models, use get_forecast and adjust the forecast back to the original scale
    forecast_res = best_model_details["Model Object"].get_forecast(steps=forecast_steps)
    forecast = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int()
    # Assuming forecast is on the differenced and logged scale, adjust back to original scale
    forecast = np.exp(forecast.cumsum() + data["value_log"].iloc[-1])
    conf_int_lower = np.exp(conf_int[:, 0].cumsum() + data["value_log"].iloc[-1])
    conf_int_upper = np.exp(conf_int[:, 1].cumsum() + data["value_log"].iloc[-1])

# Plot the forecast and the original data
plt.figure(figsize=(12, 6))
plt.plot(data.index, data["value"], label="Observed", marker="o")
plt.plot(forecast_index, forecast, label="Forecast", marker="x")
plt.fill_between(
    forecast_index, conf_int_lower, conf_int_upper, color="grey", alpha=0.2
)
plt.title("Forecast of Monthly Anti-Diabetic Drug Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.savefig(os.path.join(save_dir, "forecast_drug_sales.pdf"), format="pdf")  # Save plot
# plt.show()
