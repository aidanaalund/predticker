from datetime import date
from collections import deque

import streamlit as st

import pandas as pd
# from pandas.tseries.holiday import USFederalHolidayCalendar
# from pandas.tseries.offsets import CustomBusinessDay

import yfinance as yf
from plotly import graph_objs as go
import numpy as np

from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
YEAR = date.today().strftime("%Y")
STARTOFYEAR = "f'{YEAR}-01-01'"

title, loadingtext = st.columns([1, 1])
with title:
    st.title("Predticker:crystal_ball:")

st.caption("A magic stock predictor & dashboard")

if 'stocks' not in st.session_state:
    st.session_state.stocks = set(["AAPL", "GOOG", "MSFT", "GME"])
if 'predictiontext' not in st.session_state:
    # make this set to what the selector is currently set to
    st.session_state.predictiontext = ''
if 'stocktoadd' not in st.session_state:
    st.session_state.stocktoadd = ''

# User Input
col1, col2, col3 = st.columns([6, 3, 3])

# def addstock2(newstock):
#     if newstock != '':
#         st.session_state.stocks.add(newstock)
#         st.session_state.search_1 = newstock

# TODO: refactor to check if input is a real ticker and remove addstock()


def addstock():
    if st.session_state.textbox != '':
        st.session_state.stocks.add(st.session_state.textbox)
        st.session_state.search_1 = st.session_state.textbox


with col1:
    selected_stock = st.selectbox("Select a ticker from your list:",
                                  st.session_state.stocks,
                                  key='search_1')
with col2:
    newstock = st.text_input(label='Add a ticker to the list...',
                             placeholder="Type a ticker to add",
                             max_chars=4,
                             on_change=addstock,
                             key='textbox',
                             help='Please input a valid US ticker in all caps!')
# with col3:
#     st.write('')
#     st.write('')
#     adder = st.button('Add stock', on_click=addstock2,
#                  args=(newstock, ))


# Load correctly formatted data
@st.cache_data(show_spinner=False)
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    st.session_state.currentdataframe = data
    return data


data = load_data(selected_stock)

if 'currentdataframe' not in st.session_state:
    # make this set to what the selector is currently set to
    st.session_state.currentdataframe = data

# Data preprocessing
# grab first and last observations from df.date and make a continuous date range from that
dt_all = pd.date_range(
    start=data['Date'].iloc[0], end=data['Date'].iloc[-1], freq='D')
# check which dates from your source that also accur in the continuous date range
dt_obs = [d.strftime("%Y-%m-%d") for d in data['Date']]
# isolate missing timestamps
dt_breaks = [d for d in dt_all.strftime(
    "%Y-%m-%d").tolist() if not d in dt_obs]

# For debugging, this will display the last 5 rows of our dataframe
# st.subheader("Raw Data")
# st.write(data.tail())

# TODO: determine if our model is even worth anything...


@st.cache_resource(show_spinner=False)
def predict(stockdataframe):

    # TODO: start a timer that gives some info on how long it took to predict?
    # To put the data set in the correct form for training, 'Prepare_Data' function is implemented
    def Prepare_Data(dataframe, days):

        df = dataframe.copy()
        df['future'] = df['scaled_close'].shift(-days)
        last_sequence = np.array(df[['scaled_close']].tail(days))
        df.dropna(inplace=True)

        sequence_data = []
        sequences = deque(maxlen=NUMBER_of_STEPS_BACK)

        for entry, target in zip(df[['scaled_close', 'Date']].values, df['future'].values):
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

        model.add(LSTM(UNITS, return_sequences=True,
                  input_shape=(NUMBER_of_STEPS_BACK, 1)))
        model.add(Dropout(DROPOUT))
        model.add(LSTM(UNITS, return_sequences=False))
        model.add(Dropout(DROPOUT))
        # Makes sure that for each day, there is only one prediction
        model.add(Dense(1))

        model.compile(loss=LOSS, optimizer=OPTIMIZER)

        model.fit(x_train, y_train, epochs=EPOCHS,
                  batch_size=BATCH_SIZE, verbose=1)

        model.summary()

        return model

    # Check for a null input
    if st.session_state.predictiontext == '':
        st.session_state.predictiontext = 'Please input the stock again!'

    # take the dataframe, chop it the amount back specified internally
    NUMBER_of_STEPS_BACK = 7  # Number of days back that the model will be trained for
    # TODO: allow PREDICTION_STEPS to be modified based on the slider.
    # There is some kind of method built in with range() to do this
    # Number of days that the model will predict. To predict the next three days, modify it as follows: [1,2,3]
    PREDICTION_STEPS = [1]
    BATCH_SIZE = 16  # Number of training samples that will be passed to the network in one epoch
    # Probability to exclude the input and recurrent connections to improve performance by regularization (25%)
    DROPOUT = 0.25
    UNITS = 60  # Number of neurons connected to the layer
    EPOCHS = 15  # Number of times that the learning algorithm will work through the entire training set
    LOSS = 'mean_squared_error'  # Methodology to measure the inaccuracy
    OPTIMIZER = 'adam'  # Optimizer used to iterate to better states
    scaler = RobustScaler()
    stockdataframe['scaled_close'] = scaler.fit_transform(
        np.expand_dims(stockdataframe['Close'].values, axis=1))

    predictions = []

    for step in PREDICTION_STEPS:
        last_sequence, x_train, y_train = Prepare_Data(stockdataframe, step)
        x_train = x_train[:, :, :1].astype(np.float32)

        model = Train_Model(x_train, y_train, NUMBER_of_STEPS_BACK,
                            BATCH_SIZE, UNITS, EPOCHS, DROPOUT, OPTIMIZER, LOSS)

        last_sequence = last_sequence[-NUMBER_of_STEPS_BACK:]
        last_sequence = np.expand_dims(last_sequence, axis=0)
        prediction = model.predict(last_sequence)
        predicted_price = scaler.inverse_transform(prediction)[0][0]

        predictions.append(round(float(predicted_price), 2))

    # Print Prediction
    if len(predictions) > 0:
        predictions_list = [str(d)+'$' for d in predictions]
        predictions_str = ', '.join(predictions_list)
        message = f":sparkles: {selected_stock}'s closing price prediction(s): {predictions_str} :sparkles:"
        st.session_state.predictiontext = message


# PREDICTION UI
col3, col4, col5 = st.columns([6, 3, 3])
with col3:
    n_years = st.slider(label="Select how many days ahead you'd like to predict the closing price:",
                        min_value=1,
                        max_value=10)
with col4:
    st.write('')
    st.write('')
    adder = st.button(f'Predict **{n_years}** day(s) ahead... :male_mage:')
with col5:
    st.write('')
    # prediction = st.write(st.session_state.predictiontext)
    if adder:
        doneindicator = st.write('')
        st.write('')
        with st.spinner('Predicting...'):
            predict(st.session_state.currentdataframe)

prediction = st.write(st.session_state.predictiontext)

# Define the plot types and the default layouts
candlestick = go.Candlestick(x=data['Date'], open=data['Open'],
                             high=data['High'], low=data['Low'],
                             close=data['Close'],
                             increasing_line_color='#2ca02c',
                             decreasing_line_color='#ff4b4b')

volume = go.Scatter(x=data['Date'], y=data['Volume'])

# Returns an x/y range that shows a scaled 1W graph
# The range is returned in a format [min, max]
# There are two different y ranges returned for candlestick vs scatter


def defaultRanges(df):

    bf = -25

    x = [df['Date'].iloc[bf], df['Date'].iloc[-1]]

    cbuffer = (df['High'].iloc[bf:-1].max() -
               df['Low'].iloc[bf:-1].min())*0.25
    ycandle = [df['Low'].iloc[bf:-1].min()-cbuffer,
               df['High'].iloc[bf:-1].max()+cbuffer]
    vbuffer = (df['Volume'].iloc[bf:-1].max() -
               df['Volume'].iloc[bf:-1].min())*0.25
    yvolume = [df['Volume'].iloc[bf:-1].min()-vbuffer,
               df['Volume'].iloc[bf:-1].max()+vbuffer]

    return x, ycandle, yvolume


xrange, ycandlerange, yvolumerange = defaultRanges(data)

stocklayout = dict(

    yaxis=dict(fixedrange=False,
               ),
    xaxis_rangeslider_visible=True,
    xaxis=dict(
        fixedrange=True,
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
            dict(bounds=["sat", "mon"]),  # hide weekends
            dict(values=dt_breaks)
        ],

        range=xrange
    )
)

# Display candlestick and volume plots
# TODO: determine how to limit panning
fig = go.Figure(data=candlestick, layout=stocklayout)
fig.update_layout(title='',
                  yaxis=dict(
                      range=ycandlerange

                  ),
                  yaxis_title='Share Price ($)',
                  autosize=False,
                  width=700,
                  height=350,
                  margin=dict(
                      l=10,
                      r=0,
                      b=0,
                      t=0,
                      pad=0
                  ),)

fig2 = go.Figure(data=volume, layout=stocklayout)
fig2.update_layout(title='Volume', yaxis_title='Number of Shares',
                   autosize=False,
                   yaxis=dict(
                       range=yvolumerange
                   ),
                   width=700,
                   height=400,
                   margin=dict(
                       l=0,
                       r=10,
                       b=100,
                       t=75,
                       pad=4
                   ),)
header, subinfo = st.columns([10, 1])
change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
# TODO: this logic might be incorrect. Is it day over day?
with header:
    price = data['Close'].iloc[-1]
    # DoD increase, percentage increase, arrow

    percentage = (float(data['Close'].iloc[-1] -
                  data['Close'].iloc[-2])/abs(data['Close'].iloc[-2]))*100.00
    if change > 0:
        st.header('${:0.2f}'.format(price)+':chart_with_upwards_trend:')
        string = '+${:0.2f}'.format(change) + \
            '  (+{:0.2f}'.format(percentage)+'%)'
        st.caption(f':green[{string}]')
    elif change == 0:
        st.header('${:0.2f}'.format(price)+':arrow_right:')
        st.caption('+${:0.2f}'.format(change) +
                   '  (+{:0.2f}'.format(percentage)+'%)')
    else:
        st.header('${:0.2f}'.format(price)+':chart_with_downwards_trend:')
        string = '-${:0.2f}'.format(abs(change)) + \
            '  (-{:0.2f}'.format(abs(percentage))+'%)'
        st.caption(f':red[{string}]')

st.plotly_chart(fig, use_container_width=False)
st.plotly_chart(fig2, use_container_width=True)
