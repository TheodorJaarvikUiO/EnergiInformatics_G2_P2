from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from scipy.stats import uniform, loguniform

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#font for plots
plt.rcParams["font.family"] = "Arial"

# read data:
wind_training_data_df  = pd.read_csv(("TrainData.csv"))
wind_input_data_df = pd.read_csv(('WeatherForecastInput.csv'))
wind_solution_df = pd.read_csv(('Solution.csv'))

# parse the training data
wind_training_data_df['TIMESTAMP'] = pd.to_datetime(wind_training_data_df['TIMESTAMP'], format='%Y%m%d %H:%M')
wind_training_data_df = wind_training_data_df.set_index('TIMESTAMP')

# parse the input data
wind_input_data_df['TIMESTAMP'] = pd.to_datetime(wind_input_data_df['TIMESTAMP'], format='%Y%m%d %H:%M')
wind_input_data_df = wind_input_data_df.set_index('TIMESTAMP')

# parse the solution data
wind_solution_df['TIMESTAMP'] = pd.to_datetime(wind_solution_df['TIMESTAMP'], format='%Y%m%d %H:%M')
wind_solution_df = wind_solution_df.set_index('TIMESTAMP')

# splitting the data into training and prediction sets 80% training and 20% testing
# so that we can adjust some paramneters and reduce the final error
x = wind_training_data_df[['WS10']]  # independent variable
y = wind_training_data_df[['POWER']] # dependent variable
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

x_predict = wind_input_data_df[['WS10']]
'''
#---------------- k-NN cross-validation --------------------------------------------------
# establish the parameter 
param_knn = {'n_neighbors': range(5, 1000)}
knn = KNeighborsRegressor()
# Set up GridSearchCV for cross-validation, performs k-fold cross-validation for each value in param_grid
# and evaluates the model using the specified scoring metric.
grid_search = GridSearchCV(knn,                              # The model to tune (knn in this case)
                           param_knn,                        # dictionary with the hyperparameter
                           cv= 5,                            # number of cross-validation folds
                           scoring= 'neg_mean_squared_error' # metric to evaluate the model's performance
                           )
# Fit GridSearchCV to the training data
grid_search.fit(x_train, y_train)
# Examine the results of the cross-validation
best_k = grid_search.best_params_
cross_validation = grid_search.best_score_
# performance for all tested 'k' values
k_results_df = pd.DataFrame(grid_search.cv_results_)
'''
'''
#---------------- SVR cross-validation ----------------------------------------------------
# establish the parameters
param_svr = {
    'C': [1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'epsilon': [0.01, 0.1, 0.2],
    'gamma': ['scale', 'auto']
    }
svr = SVR()
# Set up GridSearchCV for Cross-Validation 
grid_search = GridSearchCV(svr,                               # The model to tune (SVR in this case)
                           param_grid= param_svr,             # dictionary of hyperparameters
                           cv= 3,                             # number of cross-validation folds
                           scoring= 'neg_mean_squared_error', # metric to evaluate the model's performance
                           verbose= 2,                        # verbosity of the output during training
                           n_jobs= -1                         # CPU cores to use (-1 uses all available cores)
                           )
grid_search.fit(x_train, y_train['POWER'])
best_params = grid_search.best_params_
'''
'''
#---------------- ANN cross-validation ----------------------------------------------------
nn_param_distributions = {
    'hidden_layer_sizes': [(50,), (100,), (200,), (50, 50), (100, 50), (100, 100)],
    'solver': ['adam', 'lbfgs'],
    'alpha': loguniform(1e-5, 1e-2),  # regularization
    'learning_rate': ['constant', 'adaptive']
    }

nn_model = MLPRegressor(activation='relu', max_iter=5000)

nn_random_search = RandomizedSearchCV(
    estimator= nn_model,                  
    param_distributions= nn_param_distributions,        
    n_iter= 50,
    scoring='neg_mean_squared_error',
    cv= 5,
    verbose= 1,
    random_state= 42,
    n_jobs= -1
)
nn_random_search.fit(x_train, y_train['POWER'])
best_nn_params = nn_random_search.best_params_
#-------------------------------------------------------------------------------------------
'''
# create and train the models using the training datasets
LR_model = LinearRegression()
LR_model.fit(x_train, y_train)

kNN_model =  KNeighborsRegressor(n_neighbors= 586)
kNN_model.fit(x_train, y_train)

SVR_model = SVR(kernel='rbf', C=1, gamma='scale', epsilon=0.1)
SVR_model.fit(x_train, y_train['POWER'])

ANN_model = MLPRegressor(hidden_layer_sizes=(200,), activation='relu', solver='adam', learning_rate= 'constant', max_iter=500, random_state=42)          
ANN_model.fit(x_train, y_train['POWER'])

# predictions for november 2013 using the testing data set
y_LR_test = LR_model.predict(x_test)
y_kNN_test = kNN_model.predict(x_test)
y_SVR_test = SVR_model.predict(x_test)
y_ANN_test = ANN_model.predict(x_test)

#evaluate the test error
mse_LR = mean_squared_error(y_test, y_LR_test)
rmse_LR = np.sqrt(mse_LR)

mse_kNN = mean_squared_error(y_test, y_kNN_test)
rmse_kNN = np.sqrt(mse_kNN)

mse_SVR = mean_squared_error(y_test, y_SVR_test)
rmse_SVR = np.sqrt(mse_SVR)

mse_ANN = mean_squared_error(y_test, y_ANN_test)
rmse_ANN = np.sqrt(mse_ANN)

# preparing the csv files to store the forecast
forecast_data = pd.read_csv('ForecastTemplate.csv')
y_LR_predict = pd.read_csv('ForecastTemplate.csv')
y_kNN_predict = pd.read_csv('ForecastTemplate.csv')
y_SVR_predict = pd.read_csv('ForecastTemplate.csv')
y_ANN_predict = pd.read_csv('ForecastTemplate.csv')

# predictions for november 2013 using the prediction data set
y_LR_predict['FORECAST'] = LR_model.predict(x_predict)
y_kNN_predict['FORECAST'] = kNN_model.predict(x_predict)
y_SVR_predict['FORECAST'] = SVR_model.predict(x_predict)
y_ANN_predict['FORECAST'] = ANN_model.predict(x_predict)

# sending the results to a csv file
y_LR_predict.to_csv('ForecastTemplate1-LR.csv', index=False)
y_kNN_predict.to_csv('ForecastTemplate1-kNN.csv', index=False)
y_SVR_predict.to_csv('ForecastTemplate1-SVR.csv', index=False)
y_ANN_predict.to_csv('ForecastTemplate1-NN.csv', index=False)

# setting the TIMESTAMP as index to perform the RMSE calulations with wind_solution_df
y_LR_predict = y_LR_predict.set_index(y_LR_predict.columns[0], drop=True)
y_kNN_predict = y_kNN_predict.set_index(y_kNN_predict.columns[0], drop=True)
y_SVR_predict = y_SVR_predict.set_index(y_SVR_predict.columns[0], drop=True)
y_ANN_predict = y_ANN_predict.set_index(y_ANN_predict.columns[0], drop=True)                                        
                                  
# evaluate the prediction accuracy using RMSE
mse_LR = mean_squared_error(wind_solution_df, y_LR_predict)
rmse_LR = np.sqrt(mse_LR)

mse_kNN = mean_squared_error(wind_solution_df, y_kNN_predict)
rmse_kNN = np.sqrt(mse_kNN)

mse_SVR = mean_squared_error(wind_solution_df, y_SVR_predict)
rmse_SVR = np.sqrt(mse_SVR)

mse_ANN = mean_squared_error(wind_solution_df, y_ANN_predict)
rmse_ANN = np.sqrt(mse_ANN)

# plot the power value from the solution vs the power prediction with each model 
fig, ax = plt.subplots(1,1, figsize=(14, 7))
plt.plot(wind_solution_df.index, y_LR_predict, label = 'LR prediction')
plt.plot(wind_solution_df, label = 'True measurment')
ax.legend(loc='upper center')
ax.grid(linestyle=':', alpha=0.75, color = 'grey')
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
plt.xticks(rotation=45, ha='right')
plt.xlim(wind_solution_df.index.min(), wind_solution_df.index.max())
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1,1, figsize=(14, 7))
plt.plot(wind_solution_df.index, y_kNN_predict, label = 'kNN prediction')
plt.plot(wind_solution_df, label = 'True measurment')
ax.legend()
ax.grid(linestyle=':', alpha=0.75, color = 'grey')
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
plt.xticks(rotation=45, ha='right')
plt.xlim(wind_solution_df.index.min(), wind_solution_df.index.max())
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1,1, figsize=(14, 7))
plt.plot(wind_solution_df.index, y_SVR_predict, label = 'SVR prediction')
plt.plot(wind_solution_df, label = 'True measurment')
ax.legend()
ax.grid(linestyle=':', alpha=0.75, color = 'grey')
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
plt.xticks(rotation=45, ha='right')
plt.xlim(wind_solution_df.index.min(), wind_solution_df.index.max())
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1,1, figsize=(14, 7))
plt.plot(wind_solution_df.index, y_ANN_predict, label = 'ANN prediction')
plt.plot(wind_solution_df, label = 'True measurment')
ax.legend()
ax.grid(linestyle=':', alpha=0.75, color = 'grey')
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
plt.xticks(rotation=45, ha='right')
plt.xlim(wind_solution_df.index.min(), wind_solution_df.index.max())
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1,1, figsize=(14, 7))
plt.plot(wind_solution_df.index, y_LR_predict, label = 'LR prediction')
plt.plot(wind_solution_df.index, y_kNN_predict, label = 'kNN prediction')
plt.plot(wind_solution_df.index, y_SVR_predict, label = 'SVR prediction')
plt.plot(wind_solution_df.index, y_ANN_predict, label = 'ANN prediction')
plt.plot(wind_solution_df.index, wind_solution_df, label = 'True measurment')
ax.legend()
ax.grid(linestyle=':', alpha=0.75, color = 'grey')
ax.set_title('Comparison between all prediction models')
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
plt.xticks(rotation=45, ha='right')
plt.xlim(wind_solution_df.index.min(), wind_solution_df.index.max())
plt.tight_layout()
plt.show()


#--------------------------------- SVR Model ----------------------------------------------------------
# C--> Regularization parameter: Controls the trade-off between achieving a low training error and having a "smoother" model that generalizes well to unseen data
           # Low C emphasizes a simpler, smoother model. It might tolerate more training errors. This can help prevent overfitting if your training data is noisy or if you want the model to generalize better.
           # High C tries to fit the training data more closely, aiming for a lower training error. This can lead to a more complex model that might overfit the training data, especially if it's noisy.
# gamma --> Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
           # Influences the "reach" of a single training example
           # Low gamma: A larger radius of influence. Faraway points are considered when making predictions, leading to a smoother decision boundary.
           # High gamma: A smaller radius of influence. Only points close to the prediction point have a significant effect. This can lead to a more complex and potentially wiggly decision boundary that might overfit the training data.
           # gamma='scale' (default): Uses 1/(n_features*X.var()).
           # gamma='auto': Uses 1/n_features.
# ϵ-tube
           # Defines a margin of tolerance where no penalty is given for errors within this range
           # Small epsilon: The model aims to fit the training data with very little error. This can lead to a more complex model and potentially overfitting.
           # Large epsilon: The model is more tolerant of errors. This can result in a simpler model and might improve generalization, especially if the data has some inherent noise.

#--------------------------------- ANN Model ----------------------------------------------------------
# Two hidden layers with 100 and 50 neurons
# activation --> Rectified Linear Unit activation
# solver --> Adam optimization algorithm
# max_iter --> Maximum number of iterations
# random_state --> For reproducibility