import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import keras
from keras import layers

n_steps = 500
df = pd.read_csv('../Data/TrainData.csv')
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])

df = df.drop(['U10', 'V10', 'WS10', 'U100', 'V100', 'WS100'], axis="columns")
power = df['POWER'].values

df_prediction = pd.read_csv('../Data/solution.csv')
X_pred_len = len(df_prediction)

X = []
y = []

y = np.array([power[i + n_steps:i + n_steps + X_pred_len]
              for i in range(len(power) - n_steps - X_pred_len + 1)])

X = np.array([power[i:i + n_steps]
              for i in range(len(power) - n_steps - X_pred_len + 1)])

solution = pd.read_csv('../Data/Solution.csv')
solution['TIMESTAMP'] = pd.to_datetime(solution['TIMESTAMP'])

def save_and_plot(name, preds):
    df_pred = pd.DataFrame({'TIMESTAMP': df_prediction['TIMESTAMP'], 'POWER': preds})
    df_pred['TIMESTAMP'] = pd.to_datetime(df_pred['TIMESTAMP'])
    df_pred.to_csv(f'../Results/Task3/Direct/ForecastTemplate3-{name}-direct.csv', index=False)

    rmse = np.sqrt(mean_squared_error(solution['POWER'], preds))
    print(f"{name} RMSE: {rmse:.4f}")
    plt.figure(figsize=(10, 4))
    plt.plot(solution['TIMESTAMP'], solution['POWER'], label='Actual', alpha=0.5)
    plt.plot(df_pred['TIMESTAMP'], df_pred['POWER'], label=f'{name} Forecast')
    plt.xlabel('Timestamp')
    plt.ylabel('Power')
    plt.legend()
    plt.title(f"{name} Direct Prediction, n_steps={n_steps}", fontsize=12)
    plt.tight_layout()
    plt.savefig(f'../Results/Task3/Direct/{name}_Predicted-direct.png')
    plt.close()
    return rmse

# ---------------------- Linear Regression ----------------------------------------------------------------------------
X_flat = X.reshape((X.shape[0], -1))
lr = LinearRegression()
lr.fit(X_flat, y)
lr_preds = lr.predict(X_flat[-1].reshape(1, -1)).flatten()
rmse_lr = save_and_plot("LR", lr_preds)

# ---------------------- SVR (Direct) -----------------------------------------------------------------------------------
# # Train one SVR model per forecast step
# svr_models = []
# for step in range(y.shape[1]):
#     print(f"Training SVR model for step {step}")
#     svr = SVR(C=10, epsilon=0.1, kernel='rbf')
#     svr.fit(X, y[:, step])
#     svr_models.append(svr)
#
# # Predict all steps in one go using the last input window
# svr_preds = [model.predict(X[-1].reshape(1, -1))[0] for model in svr_models]
#
# rmse_svr = save_and_plot("SVR", svr_preds)

# ---------------------- MLP ------------------------------------------------------------------------------------------
mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500)
mlp.fit(X_flat, y)
mlp_preds = mlp.predict(X_flat[-1].reshape(1, -1)).flatten()
rmse_ann = save_and_plot("ANN", mlp_preds)

# ---------------------- RNN ------------------------------------------------------------------------------------------
def build_model(n_steps, output_steps):
    model = keras.Sequential([
        layers.Input(shape=(n_steps, 1)),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(64),
        layers.Dropout(0.2),
        layers.Dense(output_steps)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

X_rnn = X.reshape((X.shape[0], X.shape[1], 1))
model = build_model(n_steps, X_pred_len)
model.fit(X_rnn, y, epochs=10, batch_size=64, validation_split=0.1,
          callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
          verbose=1)
rnn_preds = model.predict(X_rnn[-1].reshape(1, n_steps, 1), verbose=0).flatten()
rmse_rnn = save_and_plot("RNN", rnn_preds)

# ---------------------- RMSE Bar Chart --------------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.bar(['LR', 'ANN', 'RNN'], [rmse_lr, rmse_ann, rmse_rnn])
plt.ylabel('RMSE')
plt.title('Direct Forecasting RMSE Comparison')
plt.grid()
plt.tight_layout()
plt.savefig('../Results/Task3/Direct/RMSE_direct.png')
plt.close()


