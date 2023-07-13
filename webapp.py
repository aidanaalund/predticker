# Streamlit imports
import streamlit as st
from streamlit_extras.badges import badge as badge

# Data scraping/visualization
import pandas as pd
import pandas_ta as ta
import requests
import yfinance as yf
from plotly import graph_objs as go
import numpy as np
from datetime import date
from collections import deque
from datetime import datetime
import datetime
# from sklearn.linear_model import LinearRegression
# import pwlf

# Prediction imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras import metrics
import keras_tuner
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# News/NLP imports
from transformers import pipeline
import json
from newspaper import Article
# from newsapi import NewsApiClient
import newsapi
import nltk
import re
from heapq import nlargest
# import torch

START = "2016-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
YEAR = int(date.today().strftime("%Y"))
STARTOFYEAR = "f'{YEAR}-01-01'"

st.set_page_config(page_title='Predticker', page_icon=':magic_wand:', layout="centered",
                   initial_sidebar_state="auto", menu_items={
                       'Get Help': 'https://github.com/aidanaalund/predticker',
                       'Report a bug': "https://github.com/aidanaalund/predticker",
                       'About': "A dashboard that speculates future values of publicy traded US companies using an LSTM neural network model. Check out the repo for more information on how this works! Stock data pulled may not be accurate/up to date and this is not financial advice."
                   })
title, loadingtext = st.columns([1, 1])
with title:
    st.title(":crystal_ball: Predticker")

st.caption("A magic stock dashboard")
# nltk.download('punkt')
if 'stocks' not in st.session_state:
    st.session_state.stocks = set(["AAPL", "GOOG", "TSLA", "MSFT"])
if 'predictiontext' not in st.session_state:
    st.session_state.predictiontext = ''
if 'currentlayoutbutton' not in st.session_state:
    st.session_state.currentlayoutbutton = None
if 'predictionary' not in st.session_state:
    st.session_state.predictionary = {}
if 'newsdictionary' not in st.session_state:
    st.session_state.newsdictionary = {}
if 'metrics' not in st.session_state:
    st.session_state.metrics = ''
if 'hptuning' not in st.session_state:
    st.session_state.hptuning = False
if 'costfunctioncheck' not in st.session_state:
    st.session_state.costfunctioncheck = False
if 'bbandcheck' not in st.session_state:
    st.session_state.bbandcheck = False
if 'volumecheck' not in st.session_state:
    st.session_state.volumecheck = False
if 'modelhistory' not in st.session_state:
    st.session_state.modelhistory = None
if 'currentstockmetadata' not in st.session_state:
    st.session_state.currentstockmetadata = None
if 'newgraph' not in st.session_state:
    st.session_state.newgraph = True
if 'currentdataframe' not in st.session_state:
    # make this set to what the selector is currently set to
    st.session_state.currentdataframe = None
# User Input
col1, col2, col3 = st.columns([6, 3, 3])

# Callback that adds an inputted stock string to a list of stocks in the state
# Checks for an invalid ticker by attempting to get the first value in a column


def addstock():
    if st.session_state.textinput:
        try:
            temp = yf.download(st.session_state.textinput, START, TODAY)
            test = temp['Close'].iloc[0]
            st.session_state.textinput = st.session_state.textinput.upper()
            st.session_state.stocks.add(st.session_state.textinput)
            st.session_state.selectbox = st.session_state.textinput
        except IndexError:
            with col3:
                st.error(
                    f'Error: "{st.session_state.textinput}" is an invalid ticker.')
                st.session_state.textinput = ''


def newgraph():
    st.session_state.newgraph = True


with col1:
    selected_stock = st.selectbox("Select a ticker from your list:",
                                  st.session_state.stocks,
                                  key='selectbox',
                                  on_change=newgraph)
with col2:
    newstock = st.text_input(label='Add a ticker to the list...',
                             placeholder="Type a ticker to add",
                             max_chars=4,
                             on_change=addstock,
                             key='textinput',
                             help='Please input a valid US ticker.')

# Load correctly formatted data in a pandas dataframe


def add_indicators(df):
    df['SMA'] = ta.sma(df.Close, length=25)
    df['EMA'] = ta.ema(df.Close, length=25)
    df['RSI'] = ta.rsi(df.Close, length=25)
    df['WILLR'] = ta.willr(df.High, df.Low, df.Close, length=25)
    # Bollinger Bands
    df.ta.bbands(length=20, append=True)
    df['upper_band'], df['middle_band'], df['lower_band'], x, y = ta.bbands(
        df['Adj Close'], timeperiod=20)
    # Adjs close will be the most accurate!!!
    df['Target'] = df['Adj Close']-df.Open
    df['Target'] = df['Target'].shift(-1)

    df['TargetClass'] = [1 if df.Target[i]
                         > 0 else 0 for i in range(len(df))]

    df['TargetNextClose'] = df['Adj Close'].shift(-1)
    # Returns
    # Specify the length. Default is 20
    df.ta.log_return(close=df['Adj Close'], cumulative=True, append=True)
    df.ta.percent_return(close=df['Adj Close'], cumulative=True, append=True)
    # print(help(ta.log_return))
    # print(help(ta.percent_return))
    # PLR
    # Trading signal (piece-wise linear regression)

    # Naive Forecast, aka the previous day's price

    df.dropna(inplace=True)


@st.cache_data(show_spinner=False)
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    add_indicators(data)
    return data


data = load_data(selected_stock)
st.session_state.currentdataframe = data
# DATA PREPROCESSING
# grab first and last observations from df.date and make a continuous date range from that
dt_all = pd.date_range(
    start=data['Date'].iloc[0], end=data['Date'].iloc[-1], freq='D')
# check which dates from your source that also accur in the continuous date range
dt_obs = [d.strftime("%Y-%m-%d") for d in data['Date']]
# isolate missing timestamps
dt_breaks = [d for d in dt_all.strftime(
    "%Y-%m-%d").tolist() if not d in dt_obs]


def predict(stockdataframe):

    # Put the data set in the correct form for training
    @st.cache_data
    def Prepare_Data(dataframe, days):

        df = dataframe.copy()
        # the last n rows in future will be NAN, where n = days
        df['future'] = df['scaled_close'].shift(-days)
        # gets the NAN rows from the future column
        # This is the prediction set
        last_sequence = np.array(df[['scaled_close']].tail(days))
        # then drops the rows from the whole df
        df.dropna(inplace=True)

        sequence_data = []
        sequences = deque(maxlen=lookback)

        for entry, target in zip(df[['scaled_close', 'Date']].values, df['future'].values):
            sequences.append(entry)
            if len(sequences) == lookback:
                sequence_data.append([np.array(sequences), target])

        last_sequence = list([s[:1] for s in sequences]) + list(last_sequence)
        last_sequence = np.array(last_sequence).astype(np.float32)
        print(last_sequence)
        # build X and Y training set
        X, Y = [], []
        for seq, target in sequence_data:
            X.append(seq)
            Y.append(target)

        # convert X and Y to numpy arrays for compatibility, then do
        # 80 10 10 split train test val
        X = np.array(X)
        Y = np.array(Y)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8,
                                                            shuffle=False)

        X_train = X_train[:, :, :1].astype(np.float32)
        X_test = X_test[:, :, :1].astype(np.float32)

        return last_sequence, X_train, y_train, X_test, y_test

    def Create_Model(lookback, batchsize, units, dropout, optimizer, loss,):
        # A linear stack of layers
        model = Sequential()

        model.add(LSTM(units, return_sequences=True,
                  input_shape=(lookback, 1)))
        model.add(Dropout(dropout))
        model.add(LSTM(units, return_sequences=False))
        model.add(Dropout(dropout))
        # model.add(LSTM(units=50, input_shape=(None, 120)))
        # model.add(Dropout(0.2))
        # Output layer
        model.add(Dense(1))
        opt = keras.optimizers.Adam(learning_rate=0.01)
        model.compile(loss='mse', optimizer=opt, metrics=[metrics.mean_squared_error,
                                                          metrics.mean_absolute_error,
                                                          tf.keras.metrics.Recall(),
                                                          ])
        return model

    def Create_Model_Tuning(hp):
        # Define a hyperparameter space
        # batch_size = hp.Int('batch_size', min_value=16,
        #                     max_value=64, step=16)
        # epochs = hp.Int('epochs', min_value=4, max_value=8, step=2)
        # units = hp.Int('units', min_value=60, max_value=120, step=30)
        units = 60
        model = Sequential()
        lookback = 45
        # this could be where the crash happens
        model.add(LSTM(units, return_sequences=True,
                  input_shape=(lookback, 1)))
        model.add(Dropout(rate=0.25))
        model.add(LSTM(units, return_sequences=False))
        model.add(Dropout(rate=0.25))
        # Makes sure that for each day, there is only one prediction
        model.add(Dense(1))
        # hp_learning_rate = hp.Choice('learning rate', [.001, .0001])
        optimizer = 'adam'
        model.compile(loss='mse', optimizer=optimizer,
                      metrics=[metrics.mean_squared_error,
                               metrics.mean_absolute_error,
                               tf.keras.metrics.Recall(),
                               ])
        return model

    def Train_Model(_model, x_train, y_train, x_val, y_val, batchsize, epochs):

        history = model.fit(x_train, y_train, epochs=epochs,
                            batch_size=batchsize, verbose=1, validation_data=(x_val, y_val))
        print(history.history.keys())
        return model, history.history

    def crossValidation(x_train, y_train, x_val, y_val):
        tuner = keras_tuner.GridSearch(
            hypermodel=Create_Model_Tuning, objective='val_loss',
            overwrite=True,
            directory="hyperparams",
            project_name="tune_hypermodel")
        tuner.search(x_train, y_train, epochs=3,
                     validation_data=(x_val, y_val))
        best_params = tuner.get_best_hyperparameters(3)

        return best_params[0]

    # TODO: make hyperparameter tuning stuff.
    # Hand-Picked hyperparameters if tuning is not selected
    # take the dataframe, chop it the amount back specified internally
    lookback = 45  # Number of days back that the model can look back on when making sets to train, validate, and test
    # Number of days that the model will predict. To predict the next three days, modify it as follows: [1,2,3]
    predictionsteps = list(range(1, dayslider+1))
    batchsize = 16  # Number of training samples that will be passed to the network in one epoch
    # Probability to exclude the input and recurrent connections to improve performance by regularization (25%)
    dropout = 0.25
    units = 120  # Number of neurons connected to the layer
    epochs = 3  # Number of times that the learning algorithm will work through the entire training set
    loss = 'mean_squared_error'  # Methodology to measure the inaccuracy
    optimizer = 'adam'  # Optimizer used to iterate to better states

    scaler = MinMaxScaler()
    # Fit the scaler so it understands what is going on
    # TODO: Scale test and train individually!
    stockdataframe['scaled_close'] = scaler.fit_transform(
        np.expand_dims(stockdataframe['Close'].values, axis=1))

    predictions = []

    for step in predictionsteps:
        # prediction data, training data, validation data

        last_sequence, x_train, y_train, x_test, y_test = Prepare_Data(
            stockdataframe, step)

        if st.session_state.hptuning:
            best_params = crossValidation(
                x_train, y_train, x_test, y_test)
            model = Create_Model_Tuning(best_params)
        else:
            model = Create_Model(lookback, batchsize, units,
                                 dropout, optimizer, loss,)

        model, modelhistory = Train_Model(model, x_train, y_train,
                                          x_test, y_test, batchsize, epochs)

        st.session_state.modelhistory = modelhistory

        stats = model.evaluate(
            x_test, y_test, verbose=1)
        st.session_state.metrics = stats

        # takes out the last day so shape matches lookback
        last_sequence = last_sequence[-lookback:]
        last_sequence = np.expand_dims(last_sequence, axis=0)
        # Returns the scaled prediction as a numpy array
        prediction = model.predict(x=last_sequence)
        # Converts the scaled prediction to actual $! Yay!
        predicted_price = scaler.inverse_transform(prediction)[0][0]

        predictions.append(round(float(predicted_price), 2))

    # Print Prediction
    if len(predictions) > 0:
        predictions_list = [str(d) for d in predictions]
        predictions_str = ', \$'.join(predictions_list)
        st.session_state.predictionary[f'{selected_stock}'] = '\$' + \
            predictions_str


# PREDICTION UI
col4, col5, col6 = st.columns([6, 3, 3])
with col4:
    dayslider = st.slider(label="Select how many days ahead you'd like to predict the closing price:",
                          min_value=1,
                          max_value=5)
with col5:
    st.write('')
    st.write('')
    adder = st.button(f'Predict **{dayslider}** day(s) ahead... :male_mage:')
with col6:
    st.write('')
    # prediction = st.write(st.session_state.predictiontext)
    if adder:
        doneindicator = st.write('')
        st.write('')
        with st.spinner('Predicting...'):
            predict(st.session_state.currentdataframe)


# Define the plot types and the default layouts

candlestick = go.Candlestick(x=data['Date'], open=data['Open'],
                             high=data['High'], low=data['Low'],
                             close=data['Close'],
                             increasing_line_color='#2ca02c',
                             decreasing_line_color='#ff4b4b',
                             hoverinfo=None, name='Candlestick',)

volume = go.Scatter(x=data['Date'], y=data['Volume'])

bbu = go.Scatter(name='Upper Band', x=data['Date'], y=data['BBU_20_2.0'],
                 marker_color='rgba(30, 149, 242, 0.8)', opacity=.1,)
bbm = go.Scatter(name='Middle Band', x=data['Date'], y=data['BBM_20_2.0'],
                 marker_color='rgba(255, 213, 0, 0.8)', opacity=.8,)
bbl = go.Scatter(name='Lower Band', x=data['Date'], y=data['BBL_20_2.0'],
                 marker_color='rgba(30, 149, 242, 0.8)', opacity=.1,
                 fill='tonexty', fillcolor='rgba(0, 187, 255, 0.15)')

stocklayout = dict(

    yaxis=dict(fixedrange=False,
               ),
    xaxis_rangeslider_visible=False,
    xaxis=dict(
        fixedrange=False,
        rangebreaks=[
            dict(bounds=["sat", "mon"]),  # hide weekends
            dict(values=dt_breaks)
        ],

    ),
)


# Computes a scaled view for both plots based on the view mode and data
# Returned as an x range and two y ranges for each plot type (candle and volume)
header, subinfo = st.columns([2, 4])
change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
with header:
    price = data['Close'].iloc[-1]

    percentage = (float(data['Close'].iloc[-1] -
                        data['Close'].iloc[-2])/abs(data['Close'].iloc[-2]))*100.00

    st.metric(label=selected_stock,
              value='${:0.2f}'.format(price),
              delta='{:0.2f}'.format(change) +
              ' ({:0.2f}'.format(percentage)+'%) today'
              )
    recentclose = data['Date'].iloc[-1].strftime('%Y-%m-%d')
    st.caption(f'Closed: {recentclose}')
    st.markdown("""
        <style>
        [data-testid=column]:nth-of-type(1) [data-testid=stVerticalBlock]{
            gap: 0rem;
        }
        </style>
        """, unsafe_allow_html=True)
with subinfo:
    if selected_stock in st.session_state.predictionary:
        message = f"{selected_stock}'s closing price prediction(s): :magic_wand: {st.session_state.predictionary[selected_stock]}"
        prediction = st.subheader(message)

# TODO: This method is fundamentally flawed. Fix it.


@st.cache_data
def defaultRanges(df, period):

    match period:
        case '1W':
            bf = 7
        case '1M':
            bf = 30
        case '6M':
            bf = 180
        case '1Y':
            bf = 365
        case '5Y':
            bf = 365*5
        case 'YTD':
            # TODO: the issue is that the first day of the year is new year's
            firstday = datetime.datetime(YEAR, 1, 1)
            # Try to get the first entry of the year
            # Then grab the data for closing and open
            # df.loc[start:end, 'Close']
            x = [firstday, df['Date'].iloc[-1]]
            ymax = df['High'].max()
            ymin = df['Low'].min()
            cbuffer = (ymax-ymin)*0.30
            ycandle = [ymin-cbuffer,
                       ymax+cbuffer]
            yvolume = [df['Volume'].min(), df['Volume'].max()]
            return x, ycandle, yvolume
        case 'Max':
            x = [df['Date'].iloc[0], df['Date'].iloc[-1]]
            ymax = df['High'].max()
            ymin = df['Low'].min()
            cbuffer = (ymax-ymin)*0.30
            ycandle = [ymin-cbuffer,
                       ymax+cbuffer]
            yvolume = [df['Volume'].min(), df['Volume'].max()]
            return x, ycandle, yvolume
        case _:
            bf = 30

    lower = df['Date'].iloc[-1]-np.timedelta64(bf, 'D')

    upper = df['Date'].iloc[-1]+np.timedelta64(1, 'D')

    x = [lower, upper]
    cbuffer = (df['High'].iloc[-bf: -1].max() -
               df['Low'].iloc[-bf:-1].min())*0.30
    ycandle = [df['Low'].iloc[-bf:-1].min()-cbuffer,
               df['High'].iloc[-bf:-1].max()+cbuffer]
    vbuffer = (df['Volume'].iloc[-bf:-1].max() -
               df['Volume'].iloc[-bf:-1].min())*0.30
    yvolume = [df['Volume'].iloc[-bf:-1].min()-vbuffer,
               df['Volume'].iloc[-bf:-1].max()+vbuffer]

    return x, ycandle, yvolume

# Calls defaultRanges to obtain proper scales and scales the plots passed in
# Assumes plot1 is a candlestick and plot2 is a volume/scatter plot


def scalePlots(df, period, plotly1, plotly2):
    xr, cr, vr = defaultRanges(df, period)
    plotly1.update_layout(xaxis_range=xr,
                          yaxis_range=cr,)
    plotly2.update_layout(xaxis_range=xr,
                          yaxis_range=vr,)


# Initialize candlestick and volume plots
fig = go.Figure(data=candlestick, layout=stocklayout)
fig2 = go.Figure(data=volume, layout=stocklayout)
if st.session_state.bbandcheck:
    fig.add_trace(bbm)
    fig.add_trace(bbu)
    fig.add_trace(bbl)
fig.update_layout(showlegend=False,
                  yaxis={'side': 'right'},
                  title='',
                  dragmode='pan',
                  yaxis_title='Share Price ($)',
                  modebar_remove=["autoScale2d", "autoscale", "lasso", "lasso2d",
                                  "resetview",
                                  "select2d",],
                  autosize=False,
                  width=700,
                  height=350,
                  margin=dict(
                      l=0,
                      r=0,
                      b=0,
                      t=0,
                      pad=0
                  ),)
# st.plotly_chart(fig, use_container_width=True)
fig2 = go.Figure(data=volume, layout=stocklayout)
fig2.update_layout(yaxis_title='Number of Shares',
                   autosize=False,
                   yaxis={'side': 'right'},
                   dragmode='pan',
                   modebar_remove=["autoScale2d", "autoscale", "lasso", "lasso2d",
                                   "resetview",
                                   "select2d",],
                   width=700,
                   height=400,
                   margin=dict(
                       l=0,
                       r=10,
                       b=0,
                       t=0,
                       pad=4
                   ),)


rangebutton = st.radio(
    label='Range Selector', options=('1W', '1M', '6M', 'YTD', '1Y', '5Y', 'Max'),
    horizontal=True, index=1, label_visibility='collapsed')

if st.session_state.newgraph:
    string = rangebutton
    scalePlots(st.session_state.currentdataframe, string, fig, fig2)
    st.session_state.newgraph = False


if rangebutton == '1W':
    scalePlots(st.session_state.currentdataframe, '1W', fig, fig2)
    st.plotly_chart(fig, use_container_width=True)
    if st.session_state.volumecheck:
        st.plotly_chart(fig2, use_container_width=True)
if rangebutton == '1M':
    scalePlots(st.session_state.currentdataframe, '1M', fig, fig2)
    st.plotly_chart(fig, use_container_width=True)
    if st.session_state.volumecheck:
        st.plotly_chart(fig2, use_container_width=True)
if rangebutton == '6M':
    scalePlots(st.session_state.currentdataframe, '6M', fig, fig2)
    st.plotly_chart(fig, use_container_width=True)
    if st.session_state.volumecheck:
        st.plotly_chart(fig2, use_container_width=True)
if rangebutton == 'YTD':
    scalePlots(st.session_state.currentdataframe, 'YTD', fig, fig2)
    st.plotly_chart(fig, use_container_width=True)
    if st.session_state.volumecheck:
        st.plotly_chart(fig2, use_container_width=True)
if rangebutton == '1Y':
    scalePlots(st.session_state.currentdataframe, '1Y', fig, fig2)
    st.plotly_chart(fig, use_container_width=True)
    if st.session_state.volumecheck:
        st.plotly_chart(fig2, use_container_width=True)
if rangebutton == '5Y':
    scalePlots(st.session_state.currentdataframe, '5Y', fig, fig2)
    st.plotly_chart(fig, use_container_width=True)
    if st.session_state.volumecheck:
        st.plotly_chart(fig2, use_container_width=True)
if rangebutton == 'Max':
    scalePlots(st.session_state.currentdataframe, 'Max', fig, fig2)
    st.plotly_chart(fig, use_container_width=True)
    if st.session_state.volumecheck:
        st.plotly_chart(fig2, use_container_width=True)

# Recent News Section!
# summarizer = pipeline("summarization")
newsheader, newsbutton = st.columns([1, 3])

# Label a company

# insurance_keywords = ['actuary', 'claims', 'coverage', 'deductible', 'policyholder', 'premium', 'underwriter', 'risk assessment', 'insurable interest', 'loss ratio', 'reinsurance', 'actuarial tables', 'property damage', 'liability', 'flood insurance', 'term life insurance', 'whole life insurance', 'health insurance', 'auto insurance', 'homeowners insurance', 'marine insurance', 'crop insurance', 'catastrophe insurance', 'umbrella insurance',
#                       'pet insurance', 'travel insurance', 'professional liability insurance', 'disability insurance', 'long-term care insurance', 'annuity', 'pension plan', 'group insurance', 'insurtech', 'insured', 'insurer', 'subrogation', 'adjuster', 'third-party administrator', 'excess and surplus lines', 'captives', 'workers compensation', 'insurance fraud', 'health savings account', 'health maintenance organization', 'preferred provider organization']

finance_keywords = ['asset', 'liability', 'equity', 'capital', 'portfolio', 'dividend', 'financial statement', 'balance sheet', 'income statement', 'cash flow statement', 'statement of retained earnings', 'financial ratio', 'valuation', 'bond', 'stock', 'mutual fund', 'exchange-traded fund', 'hedge fund', 'private equity', 'venture capital', 'mergers and acquisitions', 'initial public offering', 'secondary market',
                    'primary market', 'securities', 'derivative', 'option', 'futures', 'forward contract', 'swaps', 'commodities', 'credit rating', 'credit score', 'credit report', 'credit bureau', 'credit history', 'credit limit', 'credit utilization', 'credit counseling', 'credit card', 'debit card', 'ATM', 'bankruptcy', 'foreclosure', 'debt consolidation', 'taxes', 'tax return', 'tax deduction', 'tax credit', 'tax bracket', 'taxable income']

# banking_capital_markets_keywords = ['bank', 'credit union', 'savings and loan association', 'commercial bank', 'investment bank', 'retail bank', 'wholesale bank', 'online bank', 'mobile banking', 'checking account', 'savings account', 'money market account', 'certificate of deposit', 'loan', 'mortgage', 'home equity loan', 'line of credit', 'credit card', 'debit card', 'ATM', 'automated clearing house', 'wire transfer', 'ACH',
#                                     'SWIFT', 'international banking', 'foreign exchange', 'forex', 'currency exchange', 'central bank', 'Federal Reserve', 'interest rate', 'inflation', 'deflation', 'monetary policy', 'fiscal policy', 'quantitative easing', 'securities', 'stock', 'bond', 'mutual fund', 'exchange-traded fund', 'hedge fund', 'private equity', 'venture capital', 'investment management', 'portfolio management', 'wealth management', 'financial planning']

healthcare_life_sciences_keywords = ['medical', 'pharmaceutical', 'pharmaceuticals', 'biotechnology', 'clinical trial', 'FDA', 'healthcare provider', 'healthcare plan', 'healthcare insurance', 'patient', 'doctor', 'nurse', 'pharmacist', 'hospital', 'clinic',
                                     'healthcare system', 'healthcare policy', 'public health', 'healthcare IT', 'electronic health record', 'telemedicine', 'personalized medicine', 'genomics', 'proteomics', 'clinical research', 'drug development', 'drug discovery', 'medicine', 'health']

law_keywords = ['law', 'legal', 'attorney', 'lawyer', 'litigation', 'arbitration', 'dispute resolution', 'contract law', 'intellectual property',
                'corporate law', 'labor law', 'tax law', 'real estate law', 'environmental law', 'criminal law', 'family law', 'immigration law', 'bankruptcy law']

# sports_keywords = ['sports', 'football', 'basketball', 'baseball', 'hockey', 'soccer', 'golf', 'tennis', 'olympics', 'athletics',
#                    'coaching', 'sports management', 'sports medicine', 'sports psychology', 'sports broadcasting', 'sports journalism', 'esports', 'fitness']

media_keywords = ['media', 'entertainment', 'film', 'television', 'radio', 'music', 'news', 'journalism', 'publishing', 'public relations',
                  'advertising', 'marketing', 'social media', 'digital media', 'animation', 'graphic design', 'web design', 'video production']

manufacturing_keywords = ['manufacturing', 'production', 'assembly', 'logistics', 'supply chain', 'quality control', 'lean manufacturing', 'six sigma', 'industrial engineering',
                          'process improvement', 'machinery', 'automation', 'aerospace', 'automotive', 'chemicals', 'construction materials', 'consumer goods', 'electronics', 'semiconductors']

automotive_keywords = ['automotive', 'cars', 'trucks', 'SUVs', 'electric vehicles', 'hybrid vehicles', 'autonomous vehicles', 'car manufacturing',
                       'automotive design', 'car dealerships', 'auto parts', 'vehicle maintenance', 'car rental', 'fleet management', 'telematics']

telecom_keywords = ['telecom', 'telecommunications', 'wireless', 'networks', 'internet', 'broadband', 'fiber optics', '5G', 'telecom infrastructure',
                    'telecom equipment', 'VoIP', 'satellite communications', 'mobile devices', 'smartphones', 'telecom services', 'telecom regulation', 'telecom policy']
# other categories to add: agriculture, energy, construction

agriculture_keywords = ['tractors', 'agriculture',
                        'harvesters', 'machinery', 'nutrient', 'turf', 'forestry']

information_technology_keywords = [
    "Artificial intelligence", "Machine learning", "Data Science", "Big Data", "Cloud Computing",
    "Cybersecurity", "Information security", "Network security", "Blockchain", "Cryptocurrency",
    "Internet of things", "IoT", "Web development", "Mobile development", "Frontend development",
    "Backend development", "Software engineering", "Software development", "Programming",
    "Database", "Data analytics", "Business intelligence", "DevOps", "Agile", "Scrum",
    "Product management", "Project management", "IT consulting", "IT service management",
    "ERP", "CRM", "SaaS", "PaaS", "IaaS", "Virtualization", "Artificial reality", "AR", "Virtual reality",
    "VR", "Gaming", "E-commerce", "Digital marketing", "SEO", "SEM", "Content marketing",
    "Social media marketing", "User experience", "UX design", "UI design", "Cloud-native",
    "Microservices", "Serverless", "Containerization", "Wearables", "Smartphone", "Cloud", "Electric Vehicles"
]

industries = {
    # 'Insurance': insurance_keywords,
    'Finance': finance_keywords,
    # 'Banking': banking_capital_markets_keywords,
    'Healthcare': healthcare_life_sciences_keywords,
    'Legal': law_keywords,
    'Agriculture': agriculture_keywords,
    # 'Sports': sports_keywords,
    'Media': media_keywords,
    'Manufacturing': manufacturing_keywords,
    'Automotive': automotive_keywords,
    'Telecom': telecom_keywords,
    'Technology': information_technology_keywords
}


def labelCompany(text):
    # Count the number of occurrences of each keyword in the text for each industry
    counts = {}
    for industry, keywords in industries.items():
        count = sum([1 for keyword in keywords if re.search(
            r"\b{}\b".format(keyword), text, re.IGNORECASE)])
        counts[industry] = count

    # Get the top industries based on their counts
    top_industries = nlargest(2, counts, key=counts.get)

    # # If only one industry was found, return it
    # if len(top_industries) == 1:
    #     return top_industries[0]
    # # If two industries were found, return them both
    # else:
    return top_industries[0]


@st.cache_data(show_spinner=False)
def fetchInfo(ticker):
    info = yf.Ticker(ticker).info
    return info


info = fetchInfo(selected_stock)
if info:
    if 'longName' in info:
        name = info['longName']
else:
    name = selected_stock
if 'longBusinessSummary' in info:
    with st.expander(f"{name}'s summary"):
        label = labelCompany(info['longBusinessSummary'])
        data_df = pd.DataFrame(
            {
                "Tag": [
                    label,
                ],
            }
        )
        st.data_editor(
            data_df,
            column_config={
                "labels": st.column_config.ListColumn(
                    "Tags",
                    width="medium",
                ),
            },
            hide_index=True,)
        st.caption(info['longBusinessSummary'])
else:
    st.error(f"{name}'s summary not available")
st.divider()
st.subheader('Recent News:')


@st.cache_data(show_spinner=False)
def fetchNews(name):
    try:

        # Init
        # newsapi = NewsApiClient(api_key='100a9812d1544d1bb65ae12e83c14ce5')
        query_params = {
            'q': f'{name}',
            "sortBy": "relevancy",
            "apiKey": "4dbc17e007ab436fb66416009dfb59a8",
            "page": 1,
            # 'domains': TODO: get reputable sources
            "pageSize": 3,
            "language": "en"
        }
        main_url = "https://newsapi.org/v2/everything"

        # fetching data in json format
        res = requests.get(main_url, params=query_params)
        response = res.json()

        # getting all articles
        articles = response["articles"]

        return articles
    except:
        st.error("News search failed.")


if f'{selected_stock}' not in st.session_state.newsdictionary:
    st.session_state.newsdictionary[f'{selected_stock}'] = fetchNews(name)

# create dropdowns for each article
for ar in st.session_state.newsdictionary[f'{selected_stock}']:
    try:
        with st.expander(ar['title']):
            url = ar["url"]
            # fullarticle = Article(url)
            # fullarticle.download()
            # fullarticle.parse()
            # fullarticle.nlp()
            # data_df = fullarticle.keywords
            # print(type(data_df)) # list
            stripped = ar['publishedAt'].split("T", 1)[0]
            st.caption(f"{ar['description']}")
            # st.caption(f"{fullarticle.text}")
            sentbutton = st.button(label='Generate an analysis...',
                                   key=url)
            if sentbutton:
                sentiment_pipeline = pipeline("sentiment-analysis")
                sent = sentiment_pipeline(ar['title'])
                st.text(sent[0]['label'])
            st.caption(f'[Read at {ar["source"]["name"]}](%s)' % url)
            if ar["author"]:
                st.caption(f'Written by {ar["author"]}')
            st.caption(f'{stripped}')
    except:
        st.error("Failed to grab article.")

# Extras + Debug Menu
st.divider()
st.subheader('Extras:')
# hptuning = st.checkbox(
#     label='Perform cross-validation before prediction (2-3x delay, could improve predictions)',
#     key='hptuning')
with st.expander(f"{selected_stock}'s Dataframe"):
    st.dataframe(data=st.session_state.currentdataframe,
                 use_container_width=True)
bbandcheck = st.checkbox(label="Display Bollinger bands",
                         key='bbandcheck')
volumecheck = st.checkbox(label="Display volume plot",
                          key='volumecheck')
st.divider()

# Credits/Links
badge(type="github", name="aidanaalund/predticker")
url = "https://www.streamlit.io"
st.caption('Made with [Streamlit](%s)' % url)


# HTML to hide the 'Made with streamlit' text
hide_menu = """
<style>
footer{
    visibility:hidden;
}
<style>
"""

st.markdown(hide_menu, unsafe_allow_html=True)
