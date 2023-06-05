import streamlit as st
from datetime import date, timedelta
import pandas as pd

# BIG PROBLEM: we are going to get rate limited on yfinance.
# Not only does this pose issues for the app as a data displayer, but
# it means we will train the model with offline data.
import yfinance as yf
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
YEAR = date.today().strftime("%Y")
startOfYear = "f'{YEAR}-01-01'"

st.title("Stock Prediction App")

stocks = ("AAPL", "GOOG", "MSFT", "GME")
selected_stock = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

#we can use st.cache_resource for ML models later on!
#st.cache_resource
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

# TODO: add volume plot with this selector
# col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([3.5,1,1,1,1,1,1,1])
# buttonNames = ['1D', '5D', '1M', '6M', 'YTD', '1Y', '5Y']
# buttonDictionary = {
#     '1D' : date.today - timedelta(days = 1), 
#     '5D' : date.today - timedelta(days = 5), 
#     '1M' : date.today - timedelta(weeks = 4), 
#     '6M' : date.today - timedelta(weeks = 24), 
#     'YTD' : startOfYear,
#     '1Y' : date.today - timedelta(days = 365), 
#     '5Y' : date.today - timedelta(days = 365 * 5)
#     }
# buttons = []


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data['Date'], open=data['Open'], 
                                 high=data['High'], low=data['Low'], 
                                 close=data['Close']))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible
                       =True)
    
    fig.update_layout(
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
                visible=True
                ),
            type="date"
            )
    )
        
    st.plotly_chart(fig)
   


plot_raw_data()

# with col1:
#     graphtypes = ['Candlestick Plot', 'Volume']
#     graph_selector = st.selectbox("Select graph type", graphtypes)
# with col2:
#     buttons.append(st.button('1D'))
# with col3:
#     buttons.append(st.button('5D'))
# with col4:
#     buttons.append(st.button('1M'))
# with col5:
#     buttons.append(st.button('6M'))
# with col6:
#     buttons.append(st.button('YTD'))
# with col7:
#     buttons.append(st.button('1Y'))
# with col8:
#     buttons.append(st.button('5Y'))
    
# for button in enumerate(buttons):
#     if button:
#         plot_raw_data(buttonDictionary.get(button.label))
        





