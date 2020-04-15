# Forecasting

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # 
import statsmodels.graphics.tsaplots as tsa_plots
import statsmodels.tsa.statespace as tm_models
from datetime import datetime,time

#import dataset
AirlineData=pd.read_excel('Airlines+Data.xlsx')

print (AirlineData.head())
print ('\n Data Types:')
print (AirlineData.dtypes)

# Converting the normal index of AirlineData to time stamp
con=AirlineData['Month']
AirlineData['Month']=pd.to_datetime(AirlineData['Month'])
AirlineData.set_index('Month', inplace=True)
#check datatype of index
AirlineData.index
AirlineData.Passengers.plot() # time series plot 

# Creating a Date column to store the actual Date format for the given Month column
AirlineData["Date"] = pd.to_datetime(AirlineData.index,format="%b-%y")

# Extracting Day, weekday name, month name, year from the Date column using 
# Date functions from pandas 
AirlineData["month"] = AirlineData.Date.dt.strftime("%b") # month extraction
#AirlineData["Day"] = AirlineData.Date.dt.strftime("%d") # Day extraction
#AirlineData["wkday"] = AirlineData.Date.dt.strftime("%A") # weekday extraction
AirlineData["year"] = AirlineData.Date.dt.strftime("%Y") # year extraction

# Some EDA on Time series data 
# Heat map visualization 
heatmap_y_month = pd.pivot_table(data=AirlineData,values="Passengers",index="year",columns="month",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g")

# Boxplot for ever
sns.boxplot(x="month",y="Passengers",data=AirlineData)
sns.boxplot(x="year",y="Passengers",data=AirlineData)
# sns.factorplot("month","Passengers",data=AirlineData,kind="box")

# Line plot for Passengers based on year  and for each month
sns.lineplot(x="year",y="Passengers",hue="month",data=AirlineData)


# moving average for the time series to understand better about the trend character in AirlineData
AirlineData.Passengers.plot(label="org")
for i in range(2,24,6):
    AirlineData["Passengers"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=3)
    
# Time series decomposition plot 
decompose_ts_add = seasonal_decompose(AirlineData.Passengers,model="additive")
decompose_ts_add.plot()
decompose_ts_mul = seasonal_decompose(AirlineData.Passengers,model="multiplicative")
decompose_ts_mul.plot()

# ACF plots and PACF plots on Original data sets 
tsa_plots.plot_acf(AirlineData.Passengers,lags=12)
tsa_plots.plot_pacf(AirlineData.Passengers)

#Checking Stationarity
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
        #Determing rolling statistics
    rolmean = pd.Series(timeseries).rolling(window=12).mean()
    rolstd = pd.Series(timeseries).rolling(window=12).std()
    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
   
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

test_stationarity(AirlineData.Passengers)

# splitting the data into Train and Test data and considering the last 12 months data as 
# Test data and left over data as train data 
Train = AirlineData.head(84)
Test = AirlineData.tail(12)

# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org))*100/org
    return np.mean(temp)

# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train['Passengers'].astype('double')).fit()
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
ses_model_MAPE=MAPE(pred_ses,Test.Passengers) #14.235

# Holt method 
hw_model = Holt(Train["Passengers"].astype('double')).fit()
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
hw_model_MAPE=MAPE(pred_hw,Test.Passengers) # 11.840


# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["Passengers"].astype('double'),seasonal="add",trend="add",seasonal_periods=12,damped=True).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
hwe_add_add_MAPE=MAPE(pred_hwe_add_add,Test.Passengers) # 1.61

# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["Passengers"].astype('double'),seasonal="mul",trend="add",seasonal_periods=12).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
hwe_mul_add_MAPE= MAPE(pred_hwe_mul_add,Test.Passengers) # 2.819


# Lets us use auto_arima from pmdarima
from pmdarima import auto_arima
auto_arima_model = auto_arima(Train["Passengers"],start_p=0,
                              start_q=0,max_p=10,max_q=10,
                              m=12,start_P=0,seasonal=True,
                              d=1,D=1,trace=True,error_action="ignore",
                              suppress_warnings= True,
                              stepwise=False)
                
            
auto_arima_model.summary() #  SARIMAX(0, 1, 3)x(1, 1, 1, 12)
# AIC ==> 531.370
# BIC ==> 544.946

# For getting Fitted values for train data set we use 
# predict_in_sample() function 
auto_arima_model.predict_in_sample( )

# For getting predictions for future we use predict() function 
pred_test = pd.Series(auto_arima_model.predict(n_periods=12))
# Adding the index values of Test Data set to predictions of Auto Arima
pred_test.index = Test.index
ARIMA_MAPE= MAPE(pred_test,Test.Passengers)  # 2.65


# Using Sarimax from statsmodels 
# As we do not have automatic function in indetifying the 
# best p,d,q combination 
# iterate over multiple combinations and return the best the combination
# For sarimax we require p,d,q and P,D,Q 
import itertools
#set parameter range
p = range(1,7)
q = range(1,3)
d = range(1,2)
s = range(12,13)
# list of all parameter combos
pdq = list(itertools.product(p, d, q))
seasonal_pdq = list(itertools.product(p, d, q, s))
results_sarima = []
best_aic = float("inf")
# SARIMA model pipeline
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(Train["Passengers"], order=param, seasonal_order=param_seasonal)
            results = mod.fit(max_iter = 50, method = 'powell')
            print('SARIMA{},{} - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
        aic = results.aic
        if aic < best_aic:
            best_model = mod
            best_aic = aic
            best_pdq = param
            best_PDQS = param_seasonal
        results_sarima.append([param,param_seasonal,results.aic])

result_sarima_table = pd.DataFrame(results_sarima)
result_sarima_table.columns = ["paramaters_l","parameters_j","aic"]
result_sarima_table = result_sarima_table.sort_values(by="aic",ascending=True).reset_index(drop=True)


best_fit_model = sm.tsa.statespace.SARIMAX(Train["Passengers"],
                                                     order = (1,1,1),seasonal_order = (5,1,1,12)).fit(disp=-1)
best_fit_model.summary()
best_fit_model.aic # 531.39
srma_pred = best_fit_model.predict(start = Test.index[0],end = Test.index[-1])
AirlineData["srma_pred"] = srma_pred
#MAPE
SARIMA_MAPE=MAPE(srma_pred,Test.Passengers)  #3.82

#visualizing ACF and PACF plot for residuals
tsa_plots.plot_acf(best_fit_model.resid,lags=12)
tsa_plots.plot_pacf(best_fit_model.resid)

# Visualization of Forecasted values for Test data set using different methods 
plt.plot(Train.index, Train["Passengers"], label='Train',color="black")
plt.plot(Test.index, Test["Passengers"], label='Test',color="blue")
plt.plot(pred_ses.index, pred_ses, label='SimpleExponential',color="green")
plt.plot(pred_hw.index, pred_hw, label='Holts_winter',color="red")
plt.plot(pred_hwe_add_add.index,pred_hwe_add_add,label="HoltsWinterExponential_1",color="brown")
plt.plot(pred_hwe_mul_add.index,pred_hwe_mul_add,label="HoltsWinterExponential_2",color="yellow")
plt.plot(pred_hwe_mul_add.index,pred_hwe_mul_add,label="Auto_Arima",color="grey")
plt.plot(pred_hwe_mul_add.index,srma_pred,label="Auto_Sarima",color="purple")
plt.legend(loc='best')

# Models and their MAPE values
model_mapes = {"MODEL":pd.Series(["SimpleExponential","Holts_winter","HoltsWinterExponential(add_add)","HoltsWinterExponential(mul_add)","Auto_Arima","Auto_Sarima"]),
               "MAPE_Values":pd.Series([ses_model_MAPE,hw_model_MAPE,hwe_add_add_MAPE,hwe_mul_add_MAPE,ARIMA_MAPE,SARIMA_MAPE])}
table_mape=pd.DataFrame(model_mapes)
table_mape

# so, Mape for Holts winter exponential smoothing with additive seasonality and additive trend has least value 
# so selecting that model for final compuatation

final_model = ExponentialSmoothing(AirlineData["Passengers"].astype('double'),seasonal="add",trend="add",seasonal_periods=12,damped=True).fit()
forecast=pd.DataFrame(final_model.forecast(12))
forecast
plt.plot(AirlineData.index, AirlineData["Passengers"], label='Train',color="black")
plt.plot(forecast.index, forecast[0], label='Test',color="blue")

