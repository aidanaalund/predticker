import streamlit as st
from datetime import date, timedelta
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

# BIG PROBLEM: we are going to get rate limited on yfinance.
# Not only does this pose issues for the app as a data displayer, but
# it means we will train the model with offline data.
import yfinance as yf
from plotly import graph_objs as go
import time
import random

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
YEAR = date.today().strftime("%Y")
startOfYear = "f'{YEAR}-01-01'"


st.title("Predticker - A stock prediction webapp")
if 'stocks' not in st.session_state:
    st.session_state.stocks = set(["AAPL", "GOOG", "MSFT", "GME"])

# User Input

selected_stock = st.selectbox("Select ticker for prediction", 
                              st.session_state.stocks,
                              key='search_1')
col1, col2 = st.columns([3,1])

def addstock(newstock):    
    st.session_state.stocks.add(newstock)
    st.session_state.search_1 = newstock
    
with col1:
    newstock = st.text_input(label='Add a ticker...', 
                         placeholder="Type a ticker and press enter to add to the list", 
                         max_chars=4,
                         value = 'AAPL')
with col2:
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
    return data

data_load_state = st.text("Loading data...")
data = load_data(selected_stock)
data_load_state.text("Finished!")

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

#TODO: machine learning goes here :)
col3, col4 = st.columns([3,1])
with col3:
    n_years = st.slider("Years of prediction:", 1, 4)
    
period = n_years * 365
with col4:
    st.write('')
    st.write('')
    adder = st.button('Predict... :male_mage:')


# Define the plot types and the default layout to them
candlestick = go.Candlestick(x=data['Date'], open=data['Open'], 
                             high=data['High'], low=data['Low'], 
                             close=data['Close'])

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
        rangeslider=dict(
            visible=False
            ),
        type="date",
        rangebreaks=[
        dict(bounds=["sat", "mon"]), #hide weekends
        dict(values=dt_breaks)
        ]
        ) 
    )
# Display candlestick and volume plots
fig = go.Figure(data=candlestick, layout=stocklayout)
fig.update_layout(title = 'Candlestick Plot')
fig2 = go.Figure(data=volume, layout=stocklayout)
fig2.update_layout(title = 'Volume Plot')

st.plotly_chart(fig)
st.plotly_chart(fig2)