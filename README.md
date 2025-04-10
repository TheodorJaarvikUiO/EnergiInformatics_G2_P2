# EnergyInformatics - Project 2
GENERAL CONSIDERATION 
Assignment 2 is about wind energy forecasting by using different machine learning 
techniques. The work to undertake involves data pre-processing, training model building, 
and implementing the algorithm and discussion of the results.  - - 
You can choose the programming language/tool you prefer e.g., 
Python/R/Java/Matlab/etc.,  
Each group should prepare a detailed report of the above tasks. 
The expected outcome of Assignment 2 includes:  
• a report  
• all ForecastTemplate.csv files 
• code as supplementary material 
Assignment 2 is to be performed in groups, where a group can consist of 3-4 students (Max 3 
students in the group(s) with PhD student(s)). 
The evaluation of Assignment 2 will count for 15% of the grade. All students in the same 
group receive the same grade. 
Description of the Assignment 
Wind power forecasts are critical for decision-making for load balancing, electricity price 
computation (electricity markets), to power grid stability. It is therefore crucial to understand 
how these forecasts can be generated.  
You will generate wind power forecasts for a given period based on a long history of past cases 
(ca. 2 years) to learn from. The historical data includes the weather data, and the power 
observations collected from the wind farm. You can build a training model to learn the 
relationship between weather and the power produced at the wind farm. You also have 
weather forecasts as input. Based on the relationship learned from the past, you can combine 
it with the weather forecasts input to forecast wind power generation over the evaluation 
period.  
The data for the assignment includes real wind power data (normalized by the wind farm 
nominal capacity, to preserve anonymity) for a wind farm in Australia. The weather input 
information is from the European Centre for Medium-range Weather Forecasts (ECMWF - 
ecmwf.int), the world-leading research and operational weather forecasting center. In the 
wind energy forecasting, the input data consists of wind forecasts at 2 heights (10m and 100m 
above ground level). Wind forecasts are given in terms of their zonal and meridional 
components, which correspond to the projection of the wind vector on West-East and South
North axes, respectively. The data and the other files provided for the assignment are described below. 

DESCRIPTION OF DATA FILES 
The files to be used for this assignment have all the necessary input weather forecasts, past 
wind power measurements, as well as templates for submitting the forecasts, for the 
assignment. The files include:  
TrainData.csv: This file gives the set of data that can be used to find a relationship between 
weather forecasts inputs (wind speed forecasts at 10m and 100m above ground level) and 
observed power generation. These data cover the period from 1.1.2012 to 31.10.2013.  
Variables (columns):  
• TIMESTAMP: Date and time of the hourly wind power measurements in following 
columns. For instance, "20120708 13:00" is for the 8th of July 2012 at 13:00  
• POWER: Measured power values (normalized)  
• U10: Zonal component of the wind forecast (West-East projection) at 10m above 
ground level 
• V10: Meridional component of the wind forecast (South-North projection) at 10m 
above ground level  
• WS10: Wind speed at 10m above ground level  
• U100: Zonal component of the wind forecast (West-East projection) at 100m above 
ground level  
• V100: Meridional component of the wind forecast (South-North projection) at 100m 
above ground level  
• WS100: Wind speed at 100m above ground level 
WeatherForecastInput.csv: This file gives the set of input weather that can be used as input 
to predict wind power generation for the whole month of 11.2013. These include wind 
speed forecasts at 10m and 100m above ground level. These data cover the period from 
1.11.2013 to 30.11.2013.  
Variables (columns):  
• TIMESTAMP: Time stamps for the wind forecasts  
• U10: Zonal component of the wind forecast (West-East projection) at 10m above 
ground level  
• V10: Meridional component of the wind forecast (South-North projection) at 10m 
above ground level  
• WS10: Wind speed at 10m above ground level  
• U100: Zonal component of the wind forecast (West-East projection) at 100m above 
ground level  
• V100: Meridional component of the wind forecast (South-North projection) at 100m 
above ground level 
• WS100: Wind speed at 100m above ground level 
Solution.csv: This file gives the true wind power measurements for the whole month of 
11.2013, which will be used to calculate the error between your forecasts and the true 
measured wind power.  
Variables (columns):  
• TIMESTAMP: Time stamps for the wind power measurements, corresponding to the 
forecasts to be compiled in ForecastTemplate.csv  
• POWER: True wind power measurement (normalized) 
ForecastTemplate.csv: This file gives the template for submitting your forecasts for the 
whole month of 11.2013. For each question, this file should be accordingly renamed. 
Variables (columns):  
• TIMESTAMP: Time stamps for the wind power forecast values to be generated  
• FORECAST: Your forecast values 

TASKS 
1. We focus on the relationship between wind power generation and wind speed. Based on 
the training data from 1.1.2012 to 31.10.2013 in the file TrainData.csv, you apply 
machine learning techniques to find the relationship between wind power generation 
and wind speed. Here, we only use the wind speed at 10m above ground level. Note 
that, through this project assignment, we only use weather data forecasting at 10m 
above ground level. The machine learning techniques include: linear regression, k
nearest neighbor (kNN), supported vector regression (SVR), and artificial neural 
networks (ANN). Each machine learning technique has a different training model. Next, 
you can find the wind speed for the whole month of 11.2013 in the file 
WeatherForecastInput.csv. For each training model and the wind speed data, you 
predict the wind power generation in 11.2013 and save the predicted results in the files: 
ForecastTemplate1-LR.csv for the linear regression model; ForecastTemplate1-kNN.csv 
for the kNN model; ForecastTemplate1-SVR.csv for the SVR model and 
ForecastTemplate1-NN.csv for the neural networks model. Finally, you evaluate the 
prediction accuracy  by comparing the predicted wind power and the true wind power 
measurements (in the file Solution.csv). Please use the error metric RMSE to evaluate 
and compare the prediction accuracy among the machine learning approaches. 
2. Wind power generation may not be only dependent on wind speed, it may be also 
related to wind direction, temperature, and pressure. In this question, we focus on the 
relationship between wind power generation and two weather parameters (i.e., wind 
speed and wind direction). First, you may have noticed the zonal component U10 and 
the meridional component V10 of the wind forecast in the file TrainData.csv. You can 
calculate the wind direction based on the zonal component and meridional component. 
Then, you build Multiple Linear Regression (MLR) model between wind power 
generation and two weather parameters (i.e., wind speed and wind direction). Finally, 
you can predict the wind power production for the whole month 11.2013; based on the 
MLR model and weather forecasting data in the file WeatherForecastInput.csv. The 
predicted wind power production is saved in the file ForecastTemplate2.csv. You 
compare the predicted wind power and the true wind power measurements (in the file 
Solution.csv) by using the metric RMSE. You may also compare the prediction accuracy 
with the linear regression where only wind speed is considered.  
3. In some situations, we may not always have weather data, e.g., wind speed data, at the 
wind farm location. In this question, we will make wind power production forecasting 
4. Wind power generation may be not only dependent on wind speed, it may be also 
related to wind direction, temperature, pressure and other parameters. For example, we 
may have the following neural network to forecast wind power generation with two 
inputs, one hidden layer with two nodes and one output. Training in a neural network 
mainly includes forward propagation and back propagation. In this neural network, 
understanding the principle to update the weights is important. 
Let us consider  
• Input data: 𝑥=0.04, 𝑥=0. 20 
• Two nodes in the hidden layer: ℎ ,	ℎ 
• Output data: 𝑦=0.50 
• Initial random weight: 𝑤, 𝑤, 𝑤, 𝑤, 𝑤, 𝑤 
• Activation function: sigmoid function 
• Learning rate: 𝛼 =0.4 
For the input 𝑥 and 𝑥 (0.04 and 0.20), we need to train the neural network to find the 
weight 𝑤(𝑖 = 1,2…6) to produce the output that is as close as possible to the actual 
output 𝑦=0.50. The training process in the first round mainly includes: 
1) Generate initial random weight between (0 1);  
2) In the forward propagation, calculate the output of the node in the output layer 
3) Calculate the error function 
4) In the back propagation, use gradient descent to update all weights 
This task includes both mathematical calculation and programming.  
Mathematical calculation in the first round: In the forward propagation, please write 
equations to calculate the output. Please calculate the error with the initial random weight. 
In the back propagation, please write all equations to calculate/update the weights via 
gradient descent. 
Please calculate the error with the update weights, and then compare with the error when 
we use the initial random weights.  
Programming: please program and run the training process. The stopping condition: the 
difference between the error in a round and the error in the previous round is lower than a 
certain threshold. 
Structure and contents of the report to be delivered 
The report for the assignment should include:  
• For task 1, use a table to compare the value of RMSE error metric among all four 
machine learning techniques. Please elaborate on the comparison e.g., how different 
and why such a difference between the results;  
• For task 1, for each machine learning technique, plot a figure for the whole month 
11.2013 to compare the true wind energy measurement and your predicted results. In 
each figure, there should be two curves such that one curve shows the true wind energy 
measurement and the other curve shows the wind power forecasts results; 
• For task 2, plot a time-series figure for 11.2013 with three curves. One curve should 
show the true wind energy measurement, the 2nd curve should show the wind power 
forecasts results by using linear regression, and the 3rd curve should show the wind 
power forecasts results by using multiple linear regression. In addition, use a table to 
compare the prediction accuracy by using linear regression and multiple linear 
regression, and elaborate on whether wind direction plays an important role in wind 
power prediction; 
• For task 3, please explain the training data for the linear regression model, the SVR 
model, the ANN model, and the RNN model. Describe how you encode the data as the 
input and the output in these models. Plot a time-series figure for the whole month 
11.2013 with three curves such that one curve shows the true wind energy 
measurement, the 2nd curve shows the wind power forecasts results by using linear 
regression, and the 3rd curve shows the wind power forecasts results by using SVR. Plot 
another time-series figure for the whole month 11.2013 with three curves such that one 
curve shows the true wind energy measurement, the 2nd curve shows the wind power 
forecasts results by using ANN, and the 3rd curve shows the wind power forecasts results 
by using RNN. Then, use a table to compare the forecasting accuracy.  
• For task 4, show via mathematical calculation whether the error has been decreased by 
using the updated weight, compared to the error in the 1st round. Use a table (or a 
screenshot) to show the error in the first 10 rounds and also the error in the last round 
until the training process stops.  
• The code should be provided separately. 
Delivery of the Assignment  
Assignment 2 is to be sent to the following email  
Email: sabita@ifi.uio.no and poushals@ifi.uio.no  
Submission form: the submission should be in a ZIP file with naming convention “IN5410
Assignment2 - GroupX.zip", where “X” is the group number.  
Email subject: “[IN5410] Assignment 2 submission by Group X” 
Firm deadline: 5th May 2024. Please be reminded that this is firm deadline. Groups that send 
reports after this deadline will receive the grade “F”. 
Questions? please contact Poushali Sengupta. Email: poushals@ifi.uio.no; office: 4448 
Reference  
1. Example source codes in R for three machine learning techniques: linear regression, kNN 
and SVR. Two files: LR-kNN-SVR-forHousePrice.R builds the three different training 
models based on the data in the file HousePriceData.csv. Then, we can use the model to 
predict the prices of other houses with different size. 
2. Examples source code in R for neural network. Two files: NNforBostonHousePrice.R 
builds the training model based on the data in the file Boston_House.csv. Then, we use 
the model to predict the prices of other houses in Boston. 
3. Reference for task 3: page 237-240 in the book “Deep Learning and Neural Networks” by 
Jeff Heaton. You can find these pages in the file ReferenceforQuestion3.pdf. 
4. Reference for task 4, slide 22-26 in the lecture “Deep Learning for Renewable Energy 
Forecasting” for mathematical calculation