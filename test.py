import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import keras
from keras import layers
from keras import Sequential
from sklearn.model_selection import RandomizedSearchCV

# ------------------------------------------- Parameters -------------------------------------------------------------
n_steps = 1000  # Number of previous time steps to consider
df_prediction = pd.read_csv('Data/solution.csv')
forecast_horizon = len(df_prediction)

# ------------------------------------------- Data processing ----------------------------------------------------------
df = pd.read_csv('Data/TrainData.csv')
df = df.drop(['U10', 'V10', 'WS10', 'U100', 'V100', 'WS100'], axis="columns")

# Add time-based features (unscaled)
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
df['HOUR'] = df['TIMESTAMP'].dt.hour
df['DAYOFWEEK'] = df['TIMESTAMP'].dt.dayofweek
df['MONTH'] = df['TIMESTAMP'].dt.month
features = ['POWER', 'HOUR', 'DAYOFWEEK', 'MONTH']

# No scaling
df_unscaled = df[features]

# Create supervised learning dataset for multi-step output
X_rnn, y_rnn = [], []
for i in range(len(df_unscaled) - n_steps - forecast_horizon):
    X_rnn.append(df_unscaled.iloc[i:i + n_steps].values)
    y_rnn.append(df_unscaled.iloc[i + n_steps:i + n_steps + forecast_horizon]['POWER'].values)
X_rnn = np.array(X_rnn)
y_rnn = np.array(y_rnn)

# ------------------------------------------- Recurrent Neural Network -------------------------------------------------

def build_model(n_steps, n_features, output_steps):
    model = Sequential([
        layers.Input(shape=(n_steps, n_features)),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(64),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(output_steps)
    ])
    model.compile(optimizer='adam', loss='mae')
    return model

model = build_model(n_steps, X_rnn.shape[2], forecast_horizon)
model.fit(
    X_rnn, y_rnn,
    batch_size=64,
    epochs=30,
    validation_split=0.1,
    verbose=1,
    callbacks=[keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)]
)

# Predict directly the next 870 steps
X_input = df_unscaled.iloc[-n_steps:][features].values.reshape(1, n_steps, len(features))
pred_power = model.predict(X_input)[0]

# Save results
rnn_results = pd.DataFrame({'TIMESTAMP': df_prediction['TIMESTAMP'], 'POWER': pred_power})
rnn_results.to_csv('Results/Task3/ForecastTemplate3-RNN.csv', index=False)

# Evaluate
solution = pd.read_csv('Data/Solution.csv')
rmse_rnn = np.sqrt(mean_squared_error(solution['POWER'], pred_power))
print(f"RNN RMSE: {rmse_rnn:.4f}")

solution['TIMESTAMP'] = pd.to_datetime(solution['TIMESTAMP'])
rnn_results['TIMESTAMP'] = pd.to_datetime(rnn_results['TIMESTAMP'])

plt.plot(solution['TIMESTAMP'], solution['POWER'], label='Actual Power', alpha=0.5)
plt.plot(rnn_results['TIMESTAMP'], rnn_results['POWER'], label='Predicted Power (RNN)')
plt.xlabel('Timestamp')
plt.ylabel('Power')
plt.legend()
plt.savefig('Results/Task3/RNN_Predicted.png')
plt.close()
