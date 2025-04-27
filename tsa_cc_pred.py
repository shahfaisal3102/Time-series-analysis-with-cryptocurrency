import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Suppress warnings
warnings.simplefilter('ignore', ConvergenceWarning)

# Load data
df = pd.read_csv("coin_Ethereum.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df.set_index('Date', inplace=True)

# Feature Engineering for Random Forest
df['Close_lag1'] = df['Close'].shift(1)
df['Close_rolling7'] = df['Close'].rolling(window=7).mean()
df['Close_rolling14'] = df['Close'].rolling(window=14).mean()
df['Day'] = df.index.day
df['Month'] = df.index.month
df['Year'] = df.index.year
df_model = df.dropna()

# Features and Target
features = ['Close_lag1', 'Close_rolling7', 'Close_rolling14', 'Day', 'Month', 'Year']
target = 'Close'
X = df_model[features]
y = df_model[target]
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

# ARIMA Model
ts_data = df['Close']
train_size = int(len(ts_data) * 0.8)
train_arima, test_arima = ts_data.iloc[:train_size], ts_data.iloc[train_size:]
arima_model = ARIMA(train_arima, order=(5, 1, 0))
arima_result = arima_model.fit()
forecast_arima = arima_result.forecast(steps=len(test_arima))
mae_arima = mean_absolute_error(test_arima, forecast_arima)
rmse_arima = np.sqrt(mean_squared_error(test_arima, forecast_arima))

# Streamlit Dashboard
st.title("Cryptocurrency Time Series Analysis Dashboard")
st.markdown("This dashboard analyzes historical Ethereum prices using Random Forest Regression and ARIMA models.")

st.subheader("Model Performance")
st.write("**Random Forest Regression:**")
st.write(f"- MAE: {mae_rf:.2f}")
st.write(f"- RMSE: {rmse_rf:.2f}")
st.write("**ARIMA Model:**")
st.write(f"- MAE: {mae_arima:.2f}")
st.write(f"- RMSE: {rmse_arima:.2f}")

st.subheader("Predicted vs Actual Close Prices")
fig_rf, ax_rf = plt.subplots(figsize=(12, 5))
ax_rf.plot(y_test.index, y_test, label='Actual', color='blue')
ax_rf.plot(y_test.index, y_pred_rf, label='Random Forest Prediction', color='green')
ax_rf.set_title('Random Forest: Actual vs Predicted Close Prices')
ax_rf.set_xlabel('Date')
ax_rf.set_ylabel('Close Price')
ax_rf.legend()
st.pyplot(fig_rf)
st.markdown("Random Forest captures patterns well, but might miss sudden sharp movements due to its ensemble averaging.")

fig_arima, ax_arima = plt.subplots(figsize=(12, 5))
ax_arima.plot(test_arima.index, test_arima, label='Actual', color='blue')
ax_arima.plot(test_arima.index, forecast_arima, label='ARIMA Forecast', linestyle='--', color='red')
ax_arima.set_title('ARIMA: Forecast vs Actual Close Prices')
ax_arima.set_xlabel('Date')
ax_arima.set_ylabel('Close Price')
ax_arima.legend()
st.pyplot(fig_arima)
st.markdown("ARIMA model works well for linear and short-term predictions, capturing the trend based on past prices.")

st.subheader("Historical Ethereum Price Chart")
st.line_chart(df['Close'])
st.markdown("The historical closing price chart gives an overview of Ethereum's market trend over time.")

# Additional Visualizations
st.subheader("Price Distribution and Trends")
fig_dist, ax_dist = plt.subplots(figsize=(10, 4))
sns.histplot(df['Close'], bins=50, kde=True, ax=ax_dist, color='skyblue')
ax_dist.set_title('Distribution of Closing Prices')
st.pyplot(fig_dist)
st.markdown("This histogram shows the most common closing price ranges. The KDE curve helps visualize the probability density.")

fig_vol, ax_vol = plt.subplots(figsize=(12, 5))
ax_vol.plot(df.index, df['Volume'], color='orange')
ax_vol.set_title('Trading Volume Over Time')
ax_vol.set_xlabel('Date')
ax_vol.set_ylabel('Volume')
st.pyplot(fig_vol)
st.markdown("Volume trends often precede price movements. Peaks in volume can signal upcoming volatility.")

fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
corr_matrix = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Marketcap']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax_corr)
ax_corr.set_title('Correlation Heatmap')
st.pyplot(fig_corr)
st.markdown("The heatmap shows relationships between variables. Close price has a strong positive correlation with High and Low prices.")
