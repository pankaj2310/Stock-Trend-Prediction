import numpy as np
import pandas as pd
import math 
import matplotlib.pyplot as plt
import pandas_datareader as data
from sklearn.metrics import mean_squared_error
from datetime import datetime
import datetime as dt
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import math, random
import streamlit as st
from keras.models import load_model
from statsmodels.tsa.arima_model import ARIMA
import streamlit.components.v1 as components  # Import Streamlit
import csv
import nsepy
from nsepy import get_history
from datetime import date
from nsetools import Nse
from sklearn.svm import SVR


dates = [] 
prices = []


series = "EQ"

st.title('Stock Trend Prediction')



user_input = st.text_input('Enter Stock Ticker','SBIN')

quote = user_input



end = datetime.now()
start = datetime(end.year-3,end.month,end.day)

data = get_history(symbol=  quote, start=start, end=end, index = True)

df = pd.DataFrame(data=data)
df.to_csv(''+quote+'.csv')
if(df.empty):
            ts = TimeSeries(key='N6A6QT6IBFJOPJ70',output_format='pandas')
            data, meta_data = ts.get_daily_adjusted(symbol='NSE:'+quote, outputsize='full')
            #Format df
            #Last 2 yrs rows => 502, in ascending order => ::-1
            data=data.head(757).iloc[::-1]
            data=data.reset_index()
            #Keep Required cols only
            df=pd.DataFrame()
            df['Date']=data['date']
            df['Open']=data['1. open']
            df['High']=data['2. high']
            df['Low']=data['3. low']
            df['Close']=data['4. close']
            df['Adj Close']=data['5. adjusted close']
            df['Volume']=data['6. volume']
            df.to_csv(''+quote+'.csv',index=False)
            
            


        
def ARIMA_ALGO(df):
    uniqueVals = df["Code"].unique()  
    len(uniqueVals)
    df=df.set_index("Code")
        #for daily basis
    def parser(x):
        return datetime.strptime(x, '%Y-%m-%d')
    def arima_model(train, test):
        history = [x for x in train]
        predictions = list()
        for t in range(len(test)):
           model = ARIMA(history, order=(6,1 ,1))
           model_fit = model.fit(disp=0)
           output = model_fit.forecast()
           yhat = output[0]
           predictions.append(yhat[0])
           obs = test[t]
           history.append(obs)
        return predictions
    for company in uniqueVals[:10]:
        data=(df.loc[company,:]).reset_index()
        data['Price'] = data['Close']
        Quantity_date = data[['Price','Date']]
        Quantity_date.index = Quantity_date['Date'].map(lambda x: parser(x))
        Quantity_date['Price'] = Quantity_date['Price'].map(lambda x: float(x))
        Quantity_date = Quantity_date.fillna(Quantity_date.bfill())
        Quantity_date = Quantity_date.drop(['Date'],axis =1)
        fig = plt.figure(figsize=(7.2,4.8),dpi=65)
        plt.plot(Quantity_date)
        plt.savefig('static/Trends.png')
        plt.close(fig)
            
        quantity = Quantity_date.values
        size = int(len(quantity) * 0.80)
        train, test = quantity[0:size], quantity[size:len(quantity)]
            #fit in model
        predictions = arima_model(train, test)
        st.subheader('Predictions vs Original ARIMA model')
        fig = plt.figure(figsize=(7.2,4.8),dpi=65)
        
        
        plt.plot(test,'b',label = 'Original Price')
        plt.plot(predictions,'r',label = 'Predicted Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend(loc='upper left')
        st.pyplot(fig)
        
        plt.savefig('static/LSTM.png')
        plt.close(fig)
        arima_pred=predictions[-2]
        st.write("Tomorrow's",quote," Closing Price Prediction by ARIMA:",arima_pred)
            #rmse calculation
        error_arima = math.sqrt(mean_squared_error(test, predictions))
        st.write("ARIMA RMSE:",error_arima)
        return arima_pred, error_arima
            
def LSTM_ALGO(df):
        #Split data into training set and test set
        dataset_train=df.iloc[0:int(0.8*len(df)),:]
        dataset_test=df.iloc[int(0.8*len(df)):,:]
        ############# NOTE #################
        #TO PREDICT STOCK PRICES OF NEXT N DAYS, STORE PREVIOUS N DAYS IN MEMORY WHILE TRAINING
        # HERE N=7
        ###dataset_train=pd.read_csv('Google_Stock_Price_Train.csv')
        training_set=df.iloc[:,4:5].values# 1:2, to store as numpy array else Series obj will be stored
        #select cols using above manner to select as float64 type, view in var explorer

        #Feature Scaling
        from sklearn.preprocessing import MinMaxScaler
        sc=MinMaxScaler(feature_range=(0,1))#Scaled values btween 0,1
        training_set_scaled=sc.fit_transform(training_set)
        #In scaling, fit_transform for training, transform for test
        
        #Creating data stucture with 7 timesteps and 1 output. 
        #7 timesteps meaning storing trends from 7 days before current day to predict 1 next output
        X_train=[]#memory with 7 days from day i
        y_train=[]#day i
        for i in range(7,len(training_set_scaled)):
            X_train.append(training_set_scaled[i-7:i,0])
            y_train.append(training_set_scaled[i,0])
        #Convert list to numpy arrays
        X_train=np.array(X_train)
        y_train=np.array(y_train)
        X_forecast=np.array(X_train[-1,1:])
        X_forecast=np.append(X_forecast,y_train[-1])
        
        from sklearn.preprocessing import StandardScaler
        
        
        #Reshaping: Adding 3rd dimension
        X_train=np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))#.shape 0=row,1=col
        X_forecast=np.reshape(X_forecast, (1,X_forecast.shape[0],1))
        #For X_train=np.reshape(no. of rows/samples, timesteps, no. of cols/features)
        
        #Building RNN
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import Dropout
        from keras.layers import LSTM
        
        regressor = load_model('lstm_model.h5')
        #For lstm, batch_size=power of 2
        
        #Testing
        ###dataset_test=pd.read_csv('Google_Stock_Price_Test.csv')
        real_stock_price=dataset_test.iloc[:,4:5].values
        
        #To predict, we need stock prices of 7 days before the test set
        #So combine train and test set to get the entire data set
        dataset_total=pd.concat((dataset_train['Close'],dataset_test['Close']),axis=0) 
        testing_set=dataset_total[ len(dataset_total) -len(dataset_test) -7: ].values
        testing_set=testing_set.reshape(-1,1)
        #-1=till last row, (-1,1)=>(80,1). otherwise only (80,0)
        
        #Feature scaling
        testing_set=sc.transform(testing_set)
        
        #Create data structure
        X_test=[]
        for i in range(7,len(testing_set)):
            X_test.append(testing_set[i-7:i,0])
            #Convert list to numpy arrays
        X_test=np.array(X_test)
        
        #Reshaping: Adding 3rd dimension
        X_test=np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

        #Testing Prediction
        predicted_stock_price=regressor.predict(X_test)
        
        #Getting original prices back from scaled values
        predicted_stock_price=sc.inverse_transform(predicted_stock_price)
        
        
        
        st.subheader('Predictions vs Original LSTM')
        fig = plt.figure(figsize=(7.2,4.8),dpi=65)
        
        
        plt.plot(real_stock_price,'b',label = "Original Price")
        plt.plot(predicted_stock_price,'r',label = "Predicted Price")
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend(loc='upper left')
        st.pyplot(fig)
        
        plt.savefig('static/LSTM.png')
        plt.close(fig)
        
        
        error_lstm = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
        
        
         
        
        #Forecasting Prediction
        forecasted_stock_price=regressor.predict(X_forecast)
        
        #Getting original prices back from scaled values
        forecasted_stock_price=sc.inverse_transform(forecasted_stock_price)
        
        
        
        lstm_pred=forecasted_stock_price[0,0]
        
        

        
        
        st.write("LSTM RMSE:",error_lstm)
        
        
        

        
        
        return lstm_pred,error_lstm
        
        
        
        

df = pd.read_csv(''+quote+'.csv')

st.write("Latest",quote,"Stock Data: ")
today_stock=df.iloc[-1:]
st.write(today_stock)

df = df.dropna()
code_list=[]
for i in range(0,len(df)):
  code_list.append(quote)
df2=pd.DataFrame(code_list,columns=['Code'])
df2 = pd.concat([df2, df], axis=1)
df=df2              

def LIN_REG_ALGO(df):
        #No of days to be forcasted in future
        forecast_out = int(7)
        #Price after n days
        df['Close after n days'] = df['Close'].shift(-forecast_out)
        #New df with only relevant data
        df_new=df[['Close','Close after n days']]

        #Structure data for train, test & forecast
        #lables of known data, discard last 35 rows
        y =np.array(df_new.iloc[:-forecast_out,-1])
        y=np.reshape(y, (-1,1))
        #all cols of known data except lables, discard last 35 rows
        X=np.array(df_new.iloc[:-forecast_out,0:-1])
        #Unknown, X to be forecasted
        X_to_be_forecasted=np.array(df_new.iloc[-forecast_out:,0:-1])
        
        #Traning, testing to plot graphs, check accuracy
        X_train=X[0:int(0.8*len(df)),:]
        X_test=X[int(0.8*len(df)):,:]
        y_train=y[0:int(0.8*len(df)),:]
        y_test=y[int(0.8*len(df)):,:]
        
        # Feature Scaling===Normalization
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        X_to_be_forecasted=sc.transform(X_to_be_forecasted)
        
        #Training
        clf = LinearRegression(n_jobs=-1)
        clf.fit(X_train, y_train)
        
        #Testing
        y_test_pred=clf.predict(X_test)
        y_test_pred=y_test_pred*(1.04)
        
        
        error_lr = math.sqrt(mean_squared_error(y_test, y_test_pred))
        
        
        #Forecasting
        forecast_set = clf.predict(X_to_be_forecasted)
        forecast_set=forecast_set*(1.04)
        mean=forecast_set.mean()
        lr_pred=forecast_set[0,0]
        
        st.write("Tomorrow's ",quote," Closing Price Prediction BY LSTM: ",lr_pred)
        
        
        
        
        return df, lr_pred, forecast_set, mean, error_lr
        


st.subheader('Stock Data for last 3 years')
st.write(df.describe())


st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12, 6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12, 6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

from sklearn.linear_model import LinearRegression
lstm_pred, error_lstm= LSTM_ALGO(df)
arima_pred, error_arima=ARIMA_ALGO(df)



st.success("Forecasted Prices for Next 7 days (LSTM Model):")
st.write(forecast_set)