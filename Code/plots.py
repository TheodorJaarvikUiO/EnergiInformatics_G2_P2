import pandas as pd
import matplotlib.pyplot as plt

df_lr_hr = pd.read_csv('../Results/Task3/Day-Ahead/ForecastTemplate3-LR.csv')
df_lr_hr['TIMESTAMP'] = pd.to_datetime(df_lr_hr['TIMESTAMP'])

df_nn_hr = pd.read_csv('../Results/Task3/Day-Ahead/ForecastTemplate3-ANN.csv')
df_nn_hr['TIMESTAMP'] = pd.to_datetime(df_nn_hr['TIMESTAMP'])

df_svr_hr = pd.read_csv('../Results/Task3/Day-Ahead/ForecastTemplate3-SVR.csv')
df_svr_hr['TIMESTAMP'] = pd.to_datetime(df_svr_hr['TIMESTAMP'])

df_rnn_hr = pd.read_csv('../Results/Task3/Day-Ahead/ForecastTemplate3-RNN.csv')
df_rnn_hr['TIMESTAMP'] = pd.to_datetime(df_rnn_hr['TIMESTAMP'])

solution = pd.read_csv('../Data/Solution.csv')
solution['TIMESTAMP'] = pd.to_datetime(solution['TIMESTAMP'])

df_lr = pd.read_csv('../Results/Task3/ForecastTemplate3-LR.csv')
df_lr['TIMESTAMP'] = pd.to_datetime(df_lr['TIMESTAMP'])

df_nn = pd.read_csv('../Results/Task3/ForecastTemplate3-ANN.csv')
df_nn['TIMESTAMP'] = pd.to_datetime(df_nn['TIMESTAMP'])

df_svr = pd.read_csv('../Results/Task3/ForecastTemplate3-SVR.csv')
df_svr['TIMESTAMP'] = pd.to_datetime(df_svr['TIMESTAMP'])

df_rnn = pd.read_csv('../Results/Task3/ForecastTemplate3-RNN.csv')
df_rnn['TIMESTAMP'] = pd.to_datetime(df_rnn['TIMESTAMP'])

plt.figure(figsize=(12, 4))
plt.plot(solution['TIMESTAMP'], solution['POWER'], label='Actual Power', alpha=0.5)
plt.plot(df_lr_hr['TIMESTAMP'], df_lr_hr['POWER'], label='Predicted Power (LR)')
plt.plot(df_svr_hr['TIMESTAMP'], df_svr_hr['POWER'], label='Predicted Power (SVR)')
plt.xlabel('Timestamp')
plt.ylabel('Power')
plt.title('Predicted power for 1hr-Ahead')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../Results/Task3/Day-Ahead/LR_SVR_Predicted_hour.png')
plt.close()

plt.figure(figsize=(12, 4))
plt.plot(solution['TIMESTAMP'], solution['POWER'], label='Actual Power', alpha=0.5)
plt.plot(df_rnn_hr['TIMESTAMP'], df_rnn_hr['POWER'], label='Predicted Power (RNN)')
plt.plot(df_nn_hr['TIMESTAMP'], df_nn_hr['POWER'], label='Predicted Power (ANN)')
plt.xlabel('Timestamp')
plt.ylabel('Power')
plt.title('Predicted power for 1hr-Ahead')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../Results/Task3/Day-Ahead/RNN_ANN_Predicted_hour.png')
plt.close()

plt.figure(figsize=(12, 4))
plt.plot(solution['TIMESTAMP'], solution['POWER'], label='Actual Power', alpha=0.5)
plt.plot(df_lr['TIMESTAMP'], df_lr['POWER'], label='Predicted Power (LR)')
plt.plot(df_svr['TIMESTAMP'], df_svr['POWER'], label='Predicted Power (SVR)')
plt.xlabel('Timestamp')
plt.ylabel('Power')
plt.title('Predicted power 1 month')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../Results/Task3/LR_SVR_Predicted_Month.png')
plt.close()

plt.figure(figsize=(12, 4))
plt.plot(solution['TIMESTAMP'], solution['POWER'], label='Actual Power', alpha=0.5)
plt.plot(df_rnn['TIMESTAMP'], df_rnn['POWER'], label='Predicted Power (RNN)')
plt.plot(df_nn['TIMESTAMP'], df_nn['POWER'], label='Predicted Power (ANN)')
plt.xlabel('Timestamp')
plt.ylabel('Power')
plt.title('Predicted power 1 month')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../Results/Task3/RNN_ANN_Predicted_Month.png')
plt.close()



