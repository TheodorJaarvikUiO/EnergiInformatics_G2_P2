import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from keras import Sequential, layers
import keras

# ------------------------------------------- Data Preparation -------------------------------------------------------
n_steps = 500
df_prediction = pd.read_csv('../Data/solution.csv')
forecast_horizon = len(df_prediction)

df = pd.read_csv('../Data/TrainData.csv')
df = df.drop(['U10', 'V10', 'WS10', 'U100', 'V100', 'WS100'], axis="columns")
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])

X, y = [], []
for i in range(n_steps, len(df)):
    power_window = df['POWER'].values[i - n_steps:i]
    X.append(power_window)
    y.append(df['POWER'].iloc[i])
X = np.array(X)
y = np.array(y)
X_train, y_train = X, y

# ------------------------------------------- Train Traditional Models ----------------------------------------------
models = {
    "LinearRegression": LinearRegression(),
    "SVR": SVR(kernel="rbf", C=100, epsilon=0.1),
    "MLP": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)

    # Recursive Forecasting
    print(f"Forecasting {name}...")
    last_input_power = df['POWER'].values[-n_steps:].tolist()
    preds = []

    for _ in range(forecast_horizon):
        model_input = np.array(last_input_power).reshape(1, -1)
        pred = model.predict(model_input)[0]
        preds.append(pred)
        last_input_power = last_input_power[1:] + [pred]

    pred_df = pd.DataFrame({
        'TIMESTAMP': df_prediction['TIMESTAMP'],
        'POWER': preds
    })

    pred_df.to_csv(f'../Results/Task3/Recursive/ForecastTemplate3-{name}-recursive.csv', index=False)
    solution = pd.read_csv('../Data/Solution.csv')
    rmse = np.sqrt(mean_squared_error(solution['POWER'], preds))
    print(f"{name} RMSE: {rmse:.4f}")

    pred_df['TIMESTAMP'] = pd.to_datetime(pred_df['TIMESTAMP'])
    solution['TIMESTAMP'] = pd.to_datetime(solution['TIMESTAMP'])

    plt.figure(figsize=(10, 4))
    plt.plot(solution['TIMESTAMP'], solution['POWER'], label='Actual Power', alpha=0.6)
    plt.plot(pred_df['TIMESTAMP'], pred_df['POWER'], label=f'{name} Prediction')
    plt.title(f'{name} Forecast, n_steps={n_steps}', fontsize=12)
    plt.xlabel('Timestamp')
    plt.ylabel('Power')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'../Results/Task3/Recursive/{name}_Predicted-recursive.png')
    plt.close()

    results[name] = {
        "rmse": rmse,
        "forecast": pred_df
    }

# ------------------------------------------- RNN Recursive ----------------------------------------------------------

X_rnn, y_rnn = [], []
for i in range(len(df) - n_steps - 1):
    X_rnn.append(df['POWER'].values[i:i + n_steps].reshape(-1, 1))
    y_rnn.append(df['POWER'].values[i + n_steps])
X_rnn = np.array(X_rnn)
y_rnn = np.array(y_rnn)

def build_recursive_rnn(n_steps):
    model = Sequential([
        layers.Input(shape=(n_steps, 1)),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(64),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mae')
    return model

print("Training recursive RNN...")
model = build_recursive_rnn(n_steps)
model.fit(
    X_rnn, y_rnn,
    batch_size=64,
    epochs=30,
    validation_split=0.1,
    verbose=1,
    callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
)

print("Forecasting with recursive RNN...")
input_seq = df['POWER'].values[-n_steps:].reshape(1, n_steps, 1)
preds_rnn = []

for _ in range(forecast_horizon):
    next_pred = model.predict(input_seq, verbose=0)[0, 0]
    preds_rnn.append(next_pred)

    input_seq = np.append(input_seq[:, 1:, :], [[[next_pred]]], axis=1)

rnn_results = pd.DataFrame({'TIMESTAMP': df_prediction['TIMESTAMP'], 'POWER': preds_rnn})
rnn_results.to_csv('../Results/Task3/Recursive/ForecastTemplate3-RNN-recursive.csv', index=False)

solution = pd.read_csv('../Data/Solution.csv')
rmse_rnn = np.sqrt(mean_squared_error(solution['POWER'], preds_rnn))
print(f"RNN RMSE: {rmse_rnn:.4f}")

solution['TIMESTAMP'] = pd.to_datetime(solution['TIMESTAMP'])
rnn_results['TIMESTAMP'] = pd.to_datetime(rnn_results['TIMESTAMP'])

plt.figure(figsize=(10, 4))
plt.plot(solution['TIMESTAMP'], solution['POWER'], label='Actual Power', alpha=0.5)
plt.plot(rnn_results['TIMESTAMP'], rnn_results['POWER'], label='Predicted Power (RNN)')
plt.xlabel('Timestamp')
plt.ylabel('Power')
plt.legend()
plt.title(f"Recurrent Neural Network, n_steps={n_steps}", fontsize=12)
plt.tight_layout()
plt.savefig('../Results/Task3/Recursive/RNN_Predicted-recursive.png')
plt.close()

results['RNN'] = {
    "rmse": rmse_rnn,
    "forecast": rnn_results
}

plt.figure(figsize=(10, 4))
plt.bar(['Linear Regression', 'SVR', 'Artificial Neural Network', 'RNN'], [results['Linear Regression']['rmse'], results['SVR']['rmse'], results['MLP']['rmse'], results['RNN']['rmse']])
plt.ylabel('RMSE')
plt.title('RMSE Comparison')
plt.grid()
plt.tight_layout()
plt.savefig('../Results/Task3/Recursive/RMSE_recursive.png')
plt.close()



