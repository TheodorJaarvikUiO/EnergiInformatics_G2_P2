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


# ------------------------------------------- Data processing ----------------------------------------------------------

# Load and clean training data
df = pd.read_csv('Data/TrainData.csv')
df = df.drop(['U10', 'V10', 'WS10', 'U100', 'V100', 'WS100'], axis="columns")


X = df['POWER'][1:16079].values.reshape(-1, 1)
y = df['POWER'][2:16080]


# Load prediction timestamps
df_prediction = pd.read_csv('Data/solution.csv')

X_pred = df_prediction['POWER'].values.reshape(-1, 1)

# ------------------------------------------- Linear Regression Model --------------------------------------------------

# Fit and predict
lr_model = LinearRegression()
lr_model.fit(X, y)
predicted_power = lr_model.predict(X_pred)

# Save forecast, RMSE and plots
lr_results = pd.DataFrame({'TIMESTAMP': df_prediction['TIMESTAMP'], 'POWER': predicted_power})
lr_results.to_csv('Results/Task3/ForecastTemplate3-LR.csv', index=False)

solution = pd.read_csv('Data/Solution.csv')
rmse_lr = np.sqrt(mean_squared_error(solution['POWER'], predicted_power))

solution['TIMESTAMP'] = pd.to_datetime(solution['TIMESTAMP'])
lr_results['TIMESTAMP'] = pd.to_datetime(lr_results['TIMESTAMP'])

plt.plot(solution['TIMESTAMP'], solution['POWER'], label='Actual Power', alpha=0.5)
plt.plot(lr_results['TIMESTAMP'], lr_results['POWER'], label='Predicted Power')
plt.xlabel('Timestamp')
plt.ylabel('Power')
plt.legend()
plt.savefig('Results/Task3/LR_Predicted.png')
plt.close()

# ------------------------------------------- Support Vector Regression ------------------------------------------------

svr_model = SVR(kernel='rbf')
svr_model.fit(X, y)
predicted_power_svr = svr_model.predict(X_pred)

svr_results = pd.DataFrame({'TIMESTAMP': df_prediction['TIMESTAMP'], 'POWER': predicted_power_svr})
svr_results.to_csv('Results/Task3/ForecastTemplate3-SVR.csv', index=False)

rmse_svr = np.sqrt(mean_squared_error(solution['POWER'], predicted_power_svr))

solution['TIMESTAMP'] = pd.to_datetime(solution['TIMESTAMP'])
svr_results['TIMESTAMP'] = pd.to_datetime(svr_results['TIMESTAMP'])

plt.plot(solution['TIMESTAMP'], solution['POWER'], label='Actual Power', alpha=0.5)
plt.plot(svr_results['TIMESTAMP'], svr_results['POWER'], label='Predicted Power')
plt.xlabel('Timestamp')
plt.ylabel('Power')
plt.legend()
plt.savefig('Results/Task3/SVR_Predicted.png')
plt.close()

# ------------------------------------------- Artificial Neural Network ------------------------------------------------

ann_model = MLPRegressor(hidden_layer_sizes=(25, 25, 25, 25), max_iter=50, activation='relu')
ann_model.fit(X, y)
predicted_power_ann = ann_model.predict(X_pred)

ann_results = pd.DataFrame({'TIMESTAMP': df_prediction['TIMESTAMP'], 'POWER': predicted_power_ann})
ann_results.to_csv('Results/Task3/ForecastTemplate3-ANN.csv', index=False)

rmse_ann = np.sqrt(mean_squared_error(solution['POWER'], predicted_power_ann))
solution['TIMESTAMP'] = pd.to_datetime(solution['TIMESTAMP'])
ann_results['TIMESTAMP'] = pd.to_datetime(ann_results['TIMESTAMP'])

plt.plot(solution['TIMESTAMP'], solution['POWER'], label='Actual Power', alpha=0.5)
plt.plot(ann_results['TIMESTAMP'], ann_results['POWER'], label='Predicted Power')
plt.xlabel('Timestamp')
plt.ylabel('Power')
plt.legend()
plt.savefig('Results/Task3/ANN_Predicted.png')
plt.close()

# ------------------------------------------- Recurring Neural Network -------------------------------------------------

# Scale target variable
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))


#RNN model
rnn_model = keras.Sequential()
rnn_model.add(layers.SimpleRNN(200, activation='relu', input_shape=(1, X.shape[1])))
rnn_model.add(layers.Dense(1))
rnn_model.compile(optimizer='adam', loss='mse')
rnn_model.fit(X, y_scaled, epochs=10, verbose=2)

predicted_power_rnn_scaled = rnn_model.predict(X_pred)
predicted_power_rnn = scaler_y.inverse_transform(predicted_power_rnn_scaled)

rnn_results = pd.DataFrame({'TIMESTAMP': df_prediction['TIMESTAMP'],'POWER': predicted_power_rnn.flatten()})
rnn_results.to_csv('Results/Task3/ForecastTemplate3-RNN.csv', index=False)

solution['TIMESTAMP'] = pd.to_datetime(solution['TIMESTAMP'])
rnn_results['TIMESTAMP'] = pd.to_datetime(rnn_results['TIMESTAMP'])

rmse_rnn = np.sqrt(mean_squared_error(solution['POWER'], predicted_power_rnn))

plt.plot(solution['TIMESTAMP'], solution['POWER'], label='Actual Power', alpha=0.5)
plt.plot(rnn_results['TIMESTAMP'], rnn_results['POWER'], label='Predicted Power (RNN)')
plt.xlabel('Timestamp')
plt.ylabel('Power')
plt.legend()
plt.savefig('Results/Task3/RNN_Predicted.png')
plt.close()

print(f"Linear Regression RMSE: {rmse_lr:.4f}")
print(f"SVR RMSE: {rmse_svr:.4f}")
print(f"Artificial Neural Network  RMSE: {rmse_ann:.4f}")
print(f"RNN RMSE: {rmse_rnn:.4f}")

# Plot RMSE for each model
models = ['Linear Regression', 'SVR', 'Artificial Neural Network', 'RNN']
rmse_values = [rmse_lr, rmse_svr, rmse_ann, rmse_rnn]
plt.bar(models, rmse_values)
plt.ylabel('RMSE')
plt.title('RMSE for Each Model')
plt.savefig('Results/Task3/RMSE.png')
plt.close()



