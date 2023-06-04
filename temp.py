import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import tensorflow as tf
from tensorflow import keras

import os
from datetime import datetime
 
import warnings
warnings.filterwarnings("ignore")


# grabs stock ticker data
import yfinance as yf

# various libraries for UI, we can use seaborn instead of plotly for now
from plotly import graph_objs as go
import streamlit as st

# Define functions
# creates a pandas dataframe for one given stock
def load_data(ticker, startdate, enddate):
    # not sure if this works?
    data = yf.download(ticker, startdate, enddate)
    # puts the date in the first column
    data.reset_index(inplace=True)
    # data is returned in a pandas dataframe
    return data




#eventually these methods will be defined by user input
companies = ['NVDA']
selected_stock = 'AAPL'
START = "2015-01-01"
TODAY = datetime.today().strftime("%Y-%m-%d")


#data = pd.read_csv('all_stocks_5yr.csv')
data = load_data(companies,START,TODAY)
print(data.shape)
print(data.sample(7))
data.info()
# for debugging, this line will write a csv of the downloaded data
#data.to_csv('dummydata.csv')

# these two lines aren't needed right now
# by default yfinance will give us the csv with dates in datetime64 format
# therefore, code to convert date formats to the proper format is not needed
# data['date'] = pd.to_datetime(data['date'])
# data.info()

# date vs open
# date vs close
plt.figure(figsize=(20, 12))
for index, company in enumerate(companies, 1):
    plt.subplot(3, 3, index)
    plt.plot(data['Date'], data['Close'], c="r", label="close", marker="+")
    plt.plot(data['Date'], data['Open'], c="g", label="open", marker="^")
    plt.title(company)
    plt.legend()
    plt.tight_layout()
    
    
# plt.figure(figsize=(25, 12))
# for index, company in enumerate(companies, 1):
#     plt.subplot(3, 3, index)
#     plt.plot(data['Date'], data['Volume'], c='purple', marker='*')
#     plt.title("Volume")
#     plt.tight_layout()


# plt.figure(figsize=(15, 8))    
# apple = data[data['Name'] == 'AAPL']
# prediction_range = apple.loc[(apple['date'] > datetime(2013,1,1))
#   & (apple['date']<datetime(2018,1,1))]
# plt.plot(apple['date'],apple['close'])
# plt.xlabel("Date")
# plt.ylabel("Close")
# plt.title("Apple Stock Prices")
# plt.show()   


# # sdataset creation
# close_data = apple.filter(['close'])
# dataset = close_data.values
# training = int(np.ceil(len(dataset) * .95))
# print(training)

# # preprocessing
# from sklearn.preprocessing import MinMaxScaler
 
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_data = scaler.fit_transform(dataset)
 
# train_data = scaled_data[0:int(training), :]
# # prepare feature and labels
# x_train = []
# y_train = []
 
# for i in range(60, len(train_data)):
#     x_train.append(train_data[i-60:i, 0])
#     y_train.append(train_data[i, 0])
 
# x_train, y_train = np.array(x_train), np.array(y_train)
# x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# model = keras.models.Sequential()
# model.add(keras.layers.LSTM(units=64,
#                             return_sequences=True,
#                             input_shape=(x_train.shape[1], 1)))
# model.add(keras.layers.LSTM(units=64))
# model.add(keras.layers.Dense(32))
# model.add(keras.layers.Dropout(0.5))
# model.add(keras.layers.Dense(1))
# model.summary

# model.compile(optimizer='adam',
#               loss='mean_squared_error')
# history = model.fit(x_train,
#                     y_train,
#                     epochs=10)

# test_data = scaled_data[training - 60:, :]
# x_test = []
# y_test = dataset[training:, :]
# for i in range(60, len(test_data)):
#     x_test.append(test_data[i-60:i, 0])
 
# x_test = np.array(x_test)
# x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
 
# # predict the testing data
# predictions = model.predict(x_test)
# predictions = scaler.inverse_transform(predictions)
 
# # evaluation metrics
# mse = np.mean(((predictions - y_test) ** 2))
# print("MSE", mse)
# print("RMSE", np.sqrt(mse))

# train = apple[:training]
# test = apple[training:]
# test['Predictions'] = predictions
 
# plt.figure(figsize=(10, 8))
# plt.plot(train['date'], train['close'])
# plt.plot(test['date'], test[['close', 'Predictions']])
# plt.title('Apple Stock Close Price')
# plt.xlabel('Date')
# plt.ylabel("Close")
# plt.legend(['Train', 'Test', 'Predictions'])
