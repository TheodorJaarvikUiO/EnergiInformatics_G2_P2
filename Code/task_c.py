import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import tensorflow as tf
import keras
from keras import layers

# In some situations, we may not always have weather data, e.g., wind speed data, at the
# wind farm location. In this question, we will make wind power production forecasting
#
# 4
#
# when we only have wind power generation data; and we do not have other data. That is,
# in the training data file TrainData.csv, the following columns should be removed: U10,
# V10, WS10, U100, V100, WS100. In the new training data file, we only have two columns:
# TIMESTAMP and POWER, which is called as time-series data. We will apply the linear
# regression, SVR, ANN, and recurrent neural network (RNN) techniques to predict wind
# power generation. You predict the wind power generation in 11.2013 and save in the
# files: ForecastTemplate3-LR.csv for the linear regression model, ForecastTemplate3-
# SVR.csv for the supported vector regression model, ForecastTemplate3-ANN.csv for the
# artificial neural network model, and ForecastTemplate3-RNN.csv for the RNN model.
# Finally, you compare the predicted wind power and the true wind power measurements
# (in the file Solution.csv). You evaluate the prediction accuracy using RMSE. You may use
# a table to compare the prediction accuracy among these four machine learning
# approaches.


# ------------------------------------------- Data processing ----------------------------------------------------------

# Load and clean training data
df = pd.read_csv('../Data/TrainData.csv')
df = df.drop(['U10', 'V10', 'WS10', 'U100', 'V100', 'WS100'], axis=1)
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])

# Extract time-based features
df['hour'] = df['TIMESTAMP'].dt.hour
df['day'] = df['TIMESTAMP'].dt.day
df['month'] = df['TIMESTAMP'].dt.month
df['weekday'] = df['TIMESTAMP'].dt.weekday

X = df[['hour', 'day', 'month', 'weekday']]
y = df['POWER']

# Load prediction timestamps
df_prediction = pd.read_csv('../Data/ForecastTemplate.csv')
df_prediction['TIMESTAMP'] = pd.to_datetime(df_prediction['TIMESTAMP'])

# Extract same time-based features
df_prediction['hour'] = df_prediction['TIMESTAMP'].dt.hour
df_prediction['day'] = df_prediction['TIMESTAMP'].dt.day
df_prediction['month'] = df_prediction['TIMESTAMP'].dt.month
df_prediction['weekday'] = df_prediction['TIMESTAMP'].dt.weekday

X_pred = df_prediction[['hour', 'day', 'month', 'weekday']]

# ------------------------------------------- Linear Regression Model --------------------------------------------------

# Fit and predict
lr_model = LinearRegression()
lr_model.fit(X, y)
predicted_power = lr_model.predict(X_pred)

# Save forecast, RMSE and plots
lr_results = pd.DataFrame({'TIMESTAMP': df_prediction['TIMESTAMP'], 'POWER': predicted_power})
lr_results.to_csv('../Results/Task3/ForecastTemplate3-LR.csv', index=False)

solution = pd.read_csv('../Data/Solution.csv')
rmse_lr = np.sqrt(mean_squared_error(solution['POWER'], predicted_power))

solution['TIMESTAMP'] = pd.to_datetime(solution['TIMESTAMP'])
plt.plot(solution['TIMESTAMP'], solution['POWER'], label='Actual Power', alpha=0.5)
plt.plot(lr_results['TIMESTAMP'], lr_results['POWER'], label='Predicted Power')
plt.xlabel('Timestamp')
plt.ylabel('Power')
plt.legend()
plt.savefig('../Results/Task3/LR_Predicted.png')
plt.close()

# ------------------------------------------- Support Vector Regression ------------------------------------------------

svr_model = SVR(kernel='rbf')
svr_model.fit(X, y)
predicted_power_svr = svr_model.predict(X_pred)

svr_results = pd.DataFrame({'TIMESTAMP': df_prediction['TIMESTAMP'], 'POWER': predicted_power_svr})
svr_results.to_csv('../Results/Task3/ForecastTemplate3-SVR.csv', index=False)

rmse_svr = np.sqrt(mean_squared_error(solution['POWER'], predicted_power_svr))

solution['TIMESTAMP'] = pd.to_datetime(solution['TIMESTAMP'])
plt.plot(solution['TIMESTAMP'], solution['POWER'], label='Actual Power', alpha=0.5)
plt.plot(svr_results['TIMESTAMP'], svr_results['POWER'], label='Predicted Power')
plt.xlabel('Timestamp')
plt.ylabel('Power')
plt.legend()
plt.savefig('../Results/Task3/SVR_Predicted.png')
plt.close()

# ------------------------------------------- Artificial Neural Network ------------------------------------------------

ann_model = MLPRegressor(hidden_layer_sizes=(25, 25, 25, 25), max_iter=100, activation='relu')
ann_model.fit(X, y)
predicted_power_ann = ann_model.predict(X_pred)

ann_results = pd.DataFrame({'TIMESTAMP': df_prediction['TIMESTAMP'], 'POWER': predicted_power_ann})
ann_results.to_csv('../Results/Task3/ForecastTemplate3-ANN.csv', index=False)

rmse_ann = np.sqrt(mean_squared_error(solution['POWER'], predicted_power_ann))

solution['TIMESTAMP'] = pd.to_datetime(solution['TIMESTAMP'])
plt.plot(solution['TIMESTAMP'], solution['POWER'], label='Actual Power', alpha=0.5)
plt.plot(ann_results['TIMESTAMP'], ann_results['POWER'], label='Predicted Power')
plt.xlabel('Timestamp')
plt.ylabel('Power')
plt.legend()
plt.savefig('../Results/Task3/ANN_Predicted.png')
plt.close()

# ------------------------------------------- Recurring Neural Network -------------------------------------------------

# Scale target variable
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Scale input features
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X.values)
X_rnn = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

#RNN model
rnn_model = keras.Sequential()
rnn_model.add(layers.SimpleRNN(20, activation='tanh', input_shape=(1, X.shape[1])))
rnn_model.add(layers.Dense(1))
rnn_model.compile(optimizer='adam', loss='mse')
rnn_model.fit(X_rnn, y_scaled, epochs=50, verbose=2)

X_pred_scaled = scaler_X.transform(X_pred.values)
X_pred_rnn = X_pred_scaled.reshape((X_pred_scaled.shape[0], 1, X_pred_scaled.shape[1]))

predicted_power_rnn_scaled = rnn_model.predict(X_pred_rnn)
predicted_power_rnn = scaler_y.inverse_transform(predicted_power_rnn_scaled)

rnn_results = pd.DataFrame({'TIMESTAMP': df_prediction['TIMESTAMP'],'POWER': predicted_power_rnn.flatten()})
rnn_results.to_csv('../Results/Task3/ForecastTemplate3-RNN.csv', index=False)

rmse_rnn = np.sqrt(mean_squared_error(solution['POWER'], predicted_power_rnn))

plt.plot(solution['TIMESTAMP'], solution['POWER'], label='Actual Power', alpha=0.5)
plt.plot(rnn_results['TIMESTAMP'], rnn_results['POWER'], label='Predicted Power (RNN)')
plt.xlabel('Timestamp')
plt.ylabel('Power')
plt.legend()
plt.savefig('../Results/Task3/RNN_Predicted.png')
plt.close()

print(f"Linear Regression RMSE: {rmse_lr:.4f}")
print(f"SVR RMSE: {rmse_svr:.4f}")
print(f"Artificial Neural Network  RMSE: {rmse_ann:.4f}")
print(f"RNN RMSE: {rmse_rnn:.4f}")



