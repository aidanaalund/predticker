import streamlit as st
from datetime import date
import pandas as pd

# BIG PROBLEM: we are going to get rate limited on yfinance.
# Not only does this pose issues for the app as a data displayer, but
# it means we will train the model with offline data.
import yfinance as yf
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

stocks = ("AAPL", "GOOG", "MSFT", "GME")
selected_stock = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

#we can use st.cache_resource for ML models later on!
@st.cache_resource
def load_data(ticker):
    data = yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    return data

data_load_state = st.text("Loading data...")
data = load_data(selected_stock)
data_load_state.text("Finished!")

st.subheader("Raw Data")
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=['Date'], y=data['Open'], name='stock_open'))
    # fig.add_trace(go.Scatter(x=['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible
                       =True)
    st.plotly_chart(fig)
    
plot_raw_data()

  