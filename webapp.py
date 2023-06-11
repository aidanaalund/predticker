import streamlit as st
from datetime import date, timedelta
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay


import yfinance as yf
from plotly import graph_objs as go
import time
import random
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from collections import deque

START = "2022-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
YEAR = date.today().strftime("%Y")
startOfYear = "f'{YEAR}-01-01'"

# A markdown hack that makes things look cooler
# m = st.markdown("""
# <style>
# div.stButton > button:first-child {
#     border: 2px solid rgba(221, 199, 237);
#     background-color: rgb(255, 255, 255);   
# # }
# # </style>""", unsafe_allow_html=True)


# color1 = st.color_picker('选择渐变起始颜色', '#1aa3ff',key=1)
# color2 = st.color_picker('选择渐变结尾颜色', '#00ff00',key=2)
# color3 = st.color_picker('选择文字颜色', '#ffffff',key=3)
# content = "Predticker - A magic stock predictor! :crystal_ball:"
# st.markdown(f'<p style="text-align:center;background-image: linear-gradient(to right,{color1}, {color2});color:{color3};font-size:24px;border-radius:2%;">{content}</p>', unsafe_allow_html=True)
# title, emoji, sub = st.columns([2,2,3])
# with title:
title, loadingtext = st.columns([1,1])
with title:
   st.title("Predticker:crystal_ball:")
# with loadingtext:
#     #data_load_state = st.markdown("(_Loading data..._)")

st.subheader("A magic stock predictor & dashboard") 
    
if 'stocks' not in st.session_state:
    st.session_state.stocks = set(["AAPL", "GOOG", "MSFT", "GME"])
if 'predictiontext' not in st.session_state:
    #make this set to what the selector is currently set to
    st.session_state.predictiontext = ''

# User Input
col1, col2, col3 = st.columns([6,3,3])

def addstock(newstock):    
    st.session_state.stocks.add(newstock)
    st.session_state.search_1 = newstock

with col1:
    selected_stock = st.selectbox("Select ticker to display", 
                                  st.session_state.stocks,
                                  key='search_1')
with col2:
    newstock = st.text_input(label='Add a ticker...', 
                         placeholder="Type a ticker and press enter to add to the list", 
                         max_chars=4,
                         value = 'AAPL')
with col3:
    st.write('')
    st.write('')
    adder = st.button('Add stock', on_click=addstock,
                 args=(newstock, ))

# Loading data
# we can use st.cache_resource for ML models later on!
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    st.session_state.currentdataframe = data
    return data

data = load_data(selected_stock)
# data_load_state.text("_Finished!_")
# data_load_state.empty()

if 'currentdataframe' not in st.session_state:
    #make this set to what the selector is currently set to
    st.session_state.currentdataframe = data

# Data preprocessing
# grab first and last observations from df.date and make a continuous date range from that
dt_all = pd.date_range(start=data['Date'].iloc[0],end=data['Date'].iloc[-1], freq = 'D')

# check which dates from your source that also accur in the continuous date range
dt_obs = [d.strftime("%Y-%m-%d") for d in data['Date']]

# isolate missing timestamps
dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d").tolist() if not d in dt_obs]

# For debugging, this will display the last 5 rows of our dataframe
#st.subheader("Raw Data")
#st.write(data.tail())

#TODO: determine if our model is even worth anything...
@st.cache_resource(show_spinner=False)
def predict(stockdataframe):
    
    # To put the data set in the correct form for training, 'Prepare_Data' function is implemented
    def Prepare_Data(dataframe, days):

      df = dataframe.copy()
      df['future'] = df['scaled_close'].shift(-days)
      last_sequence = np.array(df[['scaled_close']].tail(days))
      df.dropna(inplace=True)
      
      sequence_data = []
      sequences = deque(maxlen=NUMBER_of_STEPS_BACK)

      for entry, target in zip(df[['scaled_close','Date']].values, df['future'].values):
          sequences.append(entry)
          if len(sequences) == NUMBER_of_STEPS_BACK:
              sequence_data.append([np.array(sequences), target])

      last_sequence = list([s[:1] for s in sequences]) + list(last_sequence)
      last_sequence = np.array(last_sequence).astype(np.float32)

      # build X and Y training set
      X, Y = [], []
      for seq, target in sequence_data:
          X.append(seq)
          Y.append(target)

      # convert X and Y to numpy arrays for compatibility
      X = np.array(X)
      Y = np.array(Y)

      return last_sequence, X, Y

    # To train the one-of-a-kind LSTM model with set hyperparameters, 'Train_Model' function is implemented
    def Train_Model(x_train, y_train, NUMBER_of_STEPS_BACK, BATCH_SIZE, UNITS, EPOCHS, DROPOUT, OPTIMIZER, LOSS):

      model = Sequential()

      model.add(LSTM(UNITS, return_sequences=True, input_shape=(NUMBER_of_STEPS_BACK, 1)))
      model.add(Dropout(DROPOUT))
      model.add(LSTM(UNITS, return_sequences=False))
      model.add(Dropout(DROPOUT))
      model.add(Dense(1)) # Makes sure that for each day, there is only one prediction

      model.compile(loss=LOSS, optimizer=OPTIMIZER)

      model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

      model.summary()

      return model

    # Check for a null input
    if st.session_state.predictiontext == '':
        st.session_state.predictiontext = 'Please input the stock again!'  
        

    #take the dataframe, chop it the amount back specified internally
    NUMBER_of_STEPS_BACK = 7 # Number of days back that the model will be trained for
    #TODO: allow PREDICTION_STEPS to be modified based on the slider.
    #There is some kind of method built in with range() to do this
    PREDICTION_STEPS = [1] # Number of days that the model will predict. To predict the next three days, modify it as follows: [1,2,3]
    BATCH_SIZE = 16 # Number of training samples that will be passed to the network in one epoch
    DROPOUT = 0.25 # Probability to exclude the input and recurrent connections to improve performance by regularization (25%)
    UNITS = 60 # Number of neurons connected to the layer
    EPOCHS = 15 # Number of times that the learning algorithm will work through the entire training set 
    LOSS='mean_squared_error' # Methodology to measure the inaccuracy
    OPTIMIZER='adam' # Optimizer used to iterate to better states
    scaler = RobustScaler()
    stockdataframe['scaled_close'] = scaler.fit_transform(np.expand_dims(stockdataframe['Close'].values, axis=1))
    
    predictions = []

    for step in PREDICTION_STEPS:
      last_sequence, x_train, y_train = Prepare_Data(stockdataframe, step)
      x_train = x_train[:, :, :1].astype(np.float32)

      model = Train_Model(x_train, y_train, NUMBER_of_STEPS_BACK, BATCH_SIZE, UNITS, EPOCHS, DROPOUT, OPTIMIZER, LOSS)
      
      last_sequence = last_sequence[-NUMBER_of_STEPS_BACK:]
      last_sequence = np.expand_dims(last_sequence, axis=0)
      prediction = model.predict(last_sequence)
      predicted_price = scaler.inverse_transform(prediction)[0][0]

      predictions.append(round(float(predicted_price), 2))
      
    # Print Prediction
    if len(predictions) > 0:
      predictions_list = [str(d)+'$' for d in predictions]
      predictions_str = ', '.join(predictions_list)
      message = f":sparkles: {selected_stock}'s share price prediction for the next day(s) is {predictions_str} :sparkles:"
      st.session_state.predictiontext = message
          


# PREDICTION UI
col3, col4, col5 = st.columns([6,3,3])
with col3:
    n_years = st.slider(label="Select how many days ahead you'd like to predict the closing price:", 
                        min_value = 1, 
                        max_value = 4)
with col4:
    st.write('')
    st.write('')
    adder = st.button(f'Predict **{n_years}** day(s) ahead... :male_mage:')
with col5:
    if adder:
        st.write('')
        st.write('')
        st.write('')
        with st.spinner('Predicting...'):
            predict(st.session_state.currentdataframe)
    
#TODO: make this into some kind of popup? Right now it is '' before predicting
# This takes up valuable screen space   
prediction = st.write(st.session_state.predictiontext)



# Define the plot types and the default layout to them
candlestick = go.Candlestick(x=data['Date'], open=data['Open'], 
                             high=data['High'], low=data['Low'], 
                             close=data['Close'],
                             increasing_line_color= '#2ca02c', 
                             decreasing_line_color= '#ff4b4b')

volume = go.Scatter(x=data['Date'], y=data['Volume'])

stocklayout = dict(
    
    yaxis=dict(fixedrange = False),
    xaxis_rangeslider_visible = True,
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="1D",
                     step="day",
                     stepmode="backward"),
                dict(count=7,
                     label="1W",
                     step="day",
                     stepmode="backward"),
                dict(count=1,
                     label="1M",
                     step="month",
                     stepmode="backward"),
                dict(count=6,
                     label="6M",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="YTD",
                     step="year",
                     stepmode="todate"),
                dict(count=1,
                     label="1Y",
                     step="year",
                     stepmode="backward"),
                dict(count=5,
                     label="5Y",
                     step="year",
                     stepmode="backward"),
                dict(step="all")
                ])
            ),
        rangebreaks=[
        dict(bounds=["sat", "mon"]), #hide weekends
        dict(values=dt_breaks)
        ]
        )
    )

# Display candlestick and volume plots
fig = go.Figure(data=candlestick, layout=stocklayout)
fig.update_layout(title = 'Candlestick Plot', yaxis_title='Share Price ($)')
fig2 = go.Figure(data=volume, layout=stocklayout)
fig2.update_layout(title = 'Volume Plot',yaxis_title='Number of Shares')

plotcontainer = st.container()
volume, candle = plotcontainer.columns([4,4])
#TODO: have some statistics like the google dashboard.
with volume:
    plotcontainer.plotly_chart(fig)
with candle:
    plotcontainer.plotly_chart(fig2)