import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import loguniform, uniform
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import keras
from keras import layers
from sklearn.model_selection import RandomizedSearchCV

# ------------------------------------------- Parameters -------------------------------------------------------------
df = pd.read_csv('Data/TrainData.csv')
df = df.drop(['U10', 'V10', 'WS10', 'U100', 'V100', 'WS100'], axis="columns")
forecast_horizon = len(pd.read_csv('Data/solution.csv'))

n_steps = 500
power = df['POWER'].values
X = np.array([power[i:i + n_steps] for i in range(len(power) - n_steps)])
X_pred_input = power[-n_steps:]

df_prediction = pd.read_csv('Data/solution.csv')
solution = pd.read_csv('Data/solution.csv')

# ------------------------------------------- Linear Regression Model --------------------------------------------------
print("(LR) started training")
y = np.array([power[i + n_steps] for i in range(len(power) - n_steps)])

lr_model = LinearRegression()
lr_model.fit(X, y)

print("(LR) started predicting")
recursive_input = X_pred_input.reshape(1, -1)
recursive_predictions = []
for _ in range(forecast_horizon):
    next_pred = lr_model.predict(recursive_input)[0]
    recursive_predictions.append(next_pred)
    recursive_input = np.roll(recursive_input, -1)
    recursive_input[0, -1] = next_pred

lr_results = pd.DataFrame({'TIMESTAMP': df_prediction['TIMESTAMP'], 'POWER': recursive_predictions})
lr_results.to_csv('Results/Task3/ForecastTemplate3-LR.csv', index=False)

solution = pd.read_csv('Data/Solution.csv')
rmse_lr = np.sqrt(mean_squared_error(solution['POWER'], recursive_predictions))

print(f"Linear Regression RMSE: {rmse_lr:.4f}")

solution['TIMESTAMP'] = pd.to_datetime(solution['TIMESTAMP'])
lr_results['TIMESTAMP'] = pd.to_datetime(lr_results['TIMESTAMP'])

plt.figure(figsize=(12, 4))
plt.plot(solution['TIMESTAMP'], solution['POWER'], label='Actual Power', alpha=0.5)
plt.plot(lr_results['TIMESTAMP'], lr_results['POWER'], label='Predicted Power')
plt.xlabel('Timestamp')
plt.ylabel('Power')
plt.title('Linear Regression')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('Results/Task3/LR_Predicted.png')
plt.close()

# ------------------------------------------- Support Vector Regression ------------------------------------------------
print("(SVR) started training")

svr_model = SVR(kernel='rbf')
svr_param_distributions = {
    'C': loguniform(1e-2, 1e3),
    'epsilon': uniform(0, 0.5),
    'gamma': ['scale', 'auto', loguniform(1e-4, 1e-1)]
}

svr_random_search = RandomizedSearchCV(
    estimator=svr_model,
    param_distributions=svr_param_distributions,
    n_iter=10,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

svr_random_search.fit(X, y)
print("(SVR) Best parameters found:", svr_random_search.best_params_)

print("(SVR) started predicting")
recursive_input = X_pred_input.reshape(1, -1)
recursive_predictions = []
for _ in range(forecast_horizon):
    pred = svr_random_search.predict(recursive_input)[0]
    recursive_predictions.append(pred)
    recursive_input = np.roll(recursive_input, -1)
    recursive_input[0, -1] = pred

svr_results = pd.DataFrame({'TIMESTAMP': df_prediction['TIMESTAMP'], 'POWER': recursive_predictions})
svr_results.to_csv('Results/Task3/ForecastTemplate3-SVR.csv', index=False)

rmse_svr = np.sqrt(mean_squared_error(solution['POWER'], recursive_predictions))

print(f"Support Vector Regression RMSE: {rmse_svr:.4f}")

solution['TIMESTAMP'] = pd.to_datetime(solution['TIMESTAMP'])
svr_results['TIMESTAMP'] = pd.to_datetime(svr_results['TIMESTAMP'])

plt.figure(figsize=(12, 4))
plt.plot(solution['TIMESTAMP'], solution['POWER'], label='Actual Power', alpha=0.5)
plt.plot(svr_results['TIMESTAMP'], svr_results['POWER'], label='Predicted Power')
plt.xlabel('Timestamp')
plt.ylabel('Power')
plt.title('Support Vector Regression')
plt.legend()
plt.tight_layout()
plt.savefig('Results/Task3/SVR_Predicted.png')
plt.close()

# ------------------------------------------- Artificial Neural Network ------------------------------------------------
print("(ANN) started training")

y = np.array([power[i + n_steps:i + n_steps + forecast_horizon]
                     for i in range(len(power) - n_steps - forecast_horizon + 1)])

X = np.array([power[i:i + n_steps]
                     for i in range(len(power) - n_steps - forecast_horizon + 1)])

ann_model = MLPRegressor(activation='relu', max_iter=500)

ann_param_distributions = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50)],
    'solver': ['adam', 'lbfgs'],
    'alpha': loguniform(1e-5, 1e-2),
    'learning_rate': ['constant', 'adaptive']
}

ann_random_search = RandomizedSearchCV(
    estimator=ann_model,
    param_distributions=ann_param_distributions,
    n_iter=15,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

ann_random_search.fit(X, y)
print("(ANN) Best parameters found:", ann_random_search.best_params_)

print("(ANN) started predicting")
direct_predictions = ann_random_search.predict(X_pred_input.reshape(1, -1))[0]
ann_results = pd.DataFrame({'TIMESTAMP': df_prediction['TIMESTAMP'], 'POWER': direct_predictions})
ann_results.to_csv('Results/Task3/ForecastTemplate3-ANN.csv', index=False)

rmse_ann = np.sqrt(mean_squared_error(solution['POWER'], direct_predictions))

print(f"Artificial Neural Network RMSE: {rmse_ann:.4f}")

solution['TIMESTAMP'] = pd.to_datetime(solution['TIMESTAMP'])
ann_results['TIMESTAMP'] = pd.to_datetime(ann_results['TIMESTAMP'])

plt.figure(figsize=(12, 4))
plt.plot(solution['TIMESTAMP'], solution['POWER'], label='Actual Power', alpha=0.5)
plt.plot(ann_results['TIMESTAMP'], ann_results['POWER'], label='Predicted Power')
plt.xlabel('Timestamp')
plt.ylabel('Power')
plt.title('Artificial Neural Network')
plt.legend()
plt.tight_layout()
plt.savefig('Results/Task3/ANN_Predicted.png')
plt.close()

# ------------------------------------------- Recurring Neural Network -------------------------------------------------
print("(RNN) started training")

def build_model(units=256, learning_rate=0.001):
    model = keras.Sequential([
        layers.Input(shape=(n_steps, 1)),
        layers.LSTM(units, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(units),
        layers.Dropout(0.2),
        layers.Dense(forecast_horizon)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

X_rnn = np.array([power[i:i + n_steps]
                  for i in range(len(power) - n_steps - forecast_horizon + 1)])
X_rnn = X_rnn.reshape((X_rnn.shape[0], X_rnn.shape[1], 1))

y_rnn = np.array([power[i + n_steps:i + n_steps + forecast_horizon]
                  for i in range(len(power) - n_steps - forecast_horizon + 1)])

model = build_model(units=128, learning_rate=0.001)
model.fit(
    X_rnn, y_rnn,
    batch_size=32,
    epochs=10,
    verbose=1,
    validation_split=0.1,
    callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
)


print("(RNN) started predicting")
X_rnn_pred_input = df['POWER'].values[-n_steps:].reshape(1, n_steps, 1)
rnn_direct_preds = model.predict(X_rnn_pred_input)[0]

rnn_results = pd.DataFrame({'TIMESTAMP': df_prediction['TIMESTAMP'], 'POWER': rnn_direct_preds})
rnn_results.to_csv('Results/Task3/ForecastTemplate3-RNN.csv', index=False)

rmse_rnn = np.sqrt(mean_squared_error(solution['POWER'], rnn_direct_preds))
print(f"Recurrent Neural Network RMSE: {rmse_rnn:.4f}")

solution['TIMESTAMP'] = pd.to_datetime(solution['TIMESTAMP'])
rnn_results['TIMESTAMP'] = pd.to_datetime(rnn_results['TIMESTAMP'])

plt.figure(figsize=(12, 4))
plt.plot(solution['TIMESTAMP'], solution['POWER'], label='Actual Power', alpha=0.5)
plt.plot(rnn_results['TIMESTAMP'], rnn_results['POWER'], label='Predicted Power')
plt.xlabel('Timestamp')
plt.ylabel('Power')
plt.title('Recurrent Neural Network')
plt.legend()
plt.tight_layout()
plt.savefig('Results/Task3/RNN_Predicted.png')
plt.close()

# ------------------------------------------- Plotting -----------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.bar(['Linear Regression', 'SVR', 'Artificial Neural Network', 'RNN'],
        [rmse_lr, rmse_svr, rmse_ann, rmse_rnn])
plt.ylabel('RMSE')
plt.title('RMSE Comparison')
plt.grid()
plt.tight_layout()
plt.savefig('Results/Task3/RMSE.png')
plt.close()

print(f"Linear Regression RMSE: {rmse_lr:.4f}")
print(f"SVR RMSE: {rmse_svr:.4f}")
print(f"Artificial Neural Network RMSE: {rmse_ann:.4f}")
print(f"RNN RMSE: {rmse_rnn:.4f}")




