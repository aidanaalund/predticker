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

# News/NLP imports
from transformers import pipeline
import json
from newspaper import Article
import newsapi
import nltk
import re
from heapq import nlargest

START = "2016-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
YEAR = int(date.today().strftime("%Y"))
STARTOFYEAR = "f'{YEAR}-01-01'"


st.set_page_config(page_title='ESGParrot', page_icon=':parrot:', layout="centered",
                   initial_sidebar_state="auto", menu_items={
                       'Get Help': 'https://github.com/aidanaalund/predticker',
                       'Report a bug': "https://github.com/aidanaalund/predticker",
                       'About': "A stock dashboard with a focus on ESG ratings and NLP analysis of recent news. Data fetched may not be accurate/up to date and this is not financial advice. Powered by Yahoo! Finance and NewsAPI."
                   })

# Initialize session state keys
if 'stocks' not in st.session_state:
    st.session_state.stocks = set(["AAPL", "CAT", "TSLA", "MSFT"])
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
col1, col2, col3 = st.columns([4, 3, 3])
# with bird:
#     st.title(':parrot:')
with col1:
    st.title(":parrot: ESGParrot")
    st.caption("An ESG-focused stock dashboard")


# Adds an inputted stock string to a list of stocks in the state
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
            st.toast(
                body=f'Error: "{st.session_state.textinput}" is an invalid ticker.',
                icon='🚩')
            st.session_state.textinput = ''

# Sets a streamlit state boolean to true, making the graph render a new stock's data set.


def newgraph():
    st.session_state.newgraph = True


with col2:
    st.text('')
    selected_stock = st.selectbox("Select a ticker from your list:",
                                  st.session_state.stocks,
                                  key='selectbox',
                                  on_change=newgraph)
with col3:
    st.text('')
    newstock = st.text_input(label='Add a ticker to the list...',
                             placeholder="Type a ticker to add",
                             max_chars=4,
                             on_change=addstock,
                             key='textinput',
                             help='Please input a valid US ticker.')


# Load correctly formatted data in a pandas dataframe.


def add_indicators(df):
    df['SMA'] = ta.sma(df.Close, length=25)
    df['EMA12'] = ta.ema(df.Close, length=10)
    df['EMA26'] = ta.ema(df.Close, length=30)
    df['RSI'] = ta.rsi(df.Close, length=14)
    # returns multiple values
    # df['ADX'], df['DMP'], df['DMN'] = ta.adx(
    #     df.High, df.Low, df.Close, length=14)
    df['WILLR'] = ta.willr(df.High, df.Low, df.Close, length=25)
    # MACD stuff (without TALib!)
    macd = df['EMA26']-df['EMA12']
    macds = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    macdh = macd - macds
    df['MACD'] = df.index.map(macd)
    df['MACDH'] = df.index.map(macdh)
    df['MACDS'] = df.index.map(macds)
    # Bollinger Bands
    df.ta.bbands(length=20, append=True)
    ta.bbands(df['Adj Close'], timeperiod=20)
    # Log return is defined as ln(d2/d1)
    # Starts at day 1
    df.ta.log_return(close=df['Adj Close'], cumulative=True, append=True)
    # Day over day log return
    df.ta.log_return(close=df['Adj Close'], cumulative=False, append=True)
    df['Target'] = np.where(df['LOGRET_1'] > 0, 1, 0)
    # df['return_'+benchmark] = 1
    # Percent return (now/day1)
    df.ta.percent_return(close=df['Adj Close'], cumulative=True, append=True)

    # Create signals
    df['EMA_12_EMA_26'] = np.where(df['EMA12'] > df['EMA26'], 1, -1)
    df['Close_EMA_12'] = np.where(df['Close'] > df['EMA12'], 1, -1)
    df['MACDS_MACD'] = np.where(df['MACDS'] > df['MACD'], 1, -1)


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

# TODO: This method returns slightly incorrect ranges for YTD


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
elif rangebutton == '1M':
    scalePlots(st.session_state.currentdataframe, '1M', fig, fig2)
    st.plotly_chart(fig, use_container_width=True)
    if st.session_state.volumecheck:
        st.plotly_chart(fig2, use_container_width=True)
elif rangebutton == '6M':
    scalePlots(st.session_state.currentdataframe, '6M', fig, fig2)
    st.plotly_chart(fig, use_container_width=True)
    if st.session_state.volumecheck:
        st.plotly_chart(fig2, use_container_width=True)
elif rangebutton == 'YTD':
    scalePlots(st.session_state.currentdataframe, 'YTD', fig, fig2)
    st.plotly_chart(fig, use_container_width=True)
    if st.session_state.volumecheck:
        st.plotly_chart(fig2, use_container_width=True)
elif rangebutton == '1Y':
    scalePlots(st.session_state.currentdataframe, '1Y', fig, fig2)
    st.plotly_chart(fig, use_container_width=True)
    if st.session_state.volumecheck:
        st.plotly_chart(fig2, use_container_width=True)
elif rangebutton == '5Y':
    scalePlots(st.session_state.currentdataframe, '5Y', fig, fig2)
    st.plotly_chart(fig, use_container_width=True)
    if st.session_state.volumecheck:
        st.plotly_chart(fig2, use_container_width=True)
elif rangebutton == 'Max':
    scalePlots(st.session_state.currentdataframe, 'Max', fig, fig2)
    st.plotly_chart(fig, use_container_width=True)
    if st.session_state.volumecheck:
        st.plotly_chart(fig2, use_container_width=True)


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
    ticker = yf.Ticker(ticker)
    info = ticker.get_info()
    return info


info = fetchInfo(selected_stock)
if info:
    if 'longName' in info:
        name = info['longName']
else:
    name = selected_stock
with st.expander(f"{name}'s summary"):
    if 'longBusinessSummary' in info:
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


@st.cache_data(show_spinner=False)
def fetchESG(ticker):
    url = f"https://yahoo-finance127.p.rapidapi.com/esg-score/{ticker}"

    headers = {
        "RapidAPI-Key": st.secrets["yahoofinancekey"],
        "RapidAPI-Host": "yahoo-finance127.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers)

    return response.json()


st.subheader(f"{name}'s ESG statistics")
json = fetchESG(selected_stock)
if 'message' not in json:
    delta = 0
    url = "https://www.sustainalytics.com/corporate-solutions/esg-solutions/esg-risk-ratings"
    o, e, s, g = st.columns([2, 1, 1, 1])
    with o:
        # st.text('')
        value = json['totalEsg']['fmt']
        st.metric(label='Overall ESG Risk', value=value, delta=None,
                  help='Overall risk is calculated by adding each individual risk score.')
        tier = float(value)
        tierstring = 'bug!'
        if tier > 0 and tier < 10:
            tierstring = 'Negligible'
        elif tier >= 10 and tier < 20:
            tierstring = 'Low'
        elif tier >= 20 and tier < 30:
            tierstring = 'Medium'
        elif tier >= 30 and tier < 40:
            tierstring = 'High'
        elif tier > 40:
            tierstring = 'Severe'
        st.caption(tierstring)
    with e:
        st.metric(label='Environment Risk',
                  value=json['environmentScore']['fmt'], delta=None)
    with s:
        st.metric(label='Social Risk',
                  value=json['socialScore']['fmt'], delta=None)
    with g:
        st.metric(label='Governance Risk',
                  value=json['governanceScore']['fmt'], delta=None)
    info, graphic = st.columns([2, 3])
    with info:
        st.caption(
            f'{json["percentile"]["fmt"]}th percentile in the {json["peerGroup"]} peer group as of {json["ratingMonth"]}/{json["ratingYear"]}')
    with graphic:
        st.image('https://www.sustainalytics.com/images/default-source/default-album/ratings.png?sfvrsn=74f12bcf_9', output_format='PNG')
    st.caption('[How does this work?](%s)' % url)
else:
    st.error(f'Sustainability data is currently not available for {name}')
st.divider()

# News Section:
st.subheader('Recent News:')
# TODO: make error message display properly. AKA, add an 'error' message to the dictionary entry for that company if it fails.


@st.cache_data(show_spinner=False)
def fetchNews(name):
    try:
        # TODO: query results in good articles, but further tuning may be needed

        query_params = {
            'q': f'{name}',
            "sortBy": "relevancy",
            "apiKey": st.secrets["newsapikey"],
            "page": 1,
            # "sources": "reuters,cbs-news,the-washington-post,the-wall-street-journal,financial-times",
            # "domains": 'cnbc.com/business,usatoday.com/money/,cnn.com/business,gizmodo.com/tech,apnews.com/business,forbes.com/business/,bloomberg.com,newsweek.com/business,finance.yahoo.com/news/,',
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
if st.session_state.newsdictionary[f'{selected_stock}']:
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
                    # TODO: improve pipeline, and get full article contents
                    sent = sentiment_pipeline(ar['title'])
                    st.text(sent[0]['label'])
                st.caption(f'[Read at {ar["source"]["name"]}](%s)' % url)
                if ar["author"]:
                    st.caption(f'Written by {ar["author"]}')
                st.caption(f'{stripped}')
        except:
            st.error("Failed to grab article.")
else:
    st.error('No articles found.')

# Extras + Debug Menu
st.divider()
st.subheader('Extras:')
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


# HTML to hide the default 'Made with streamlit' text
hide_menu = """
<style>
footer{
    visibility:hidden;
}
<style>
"""

st.markdown(hide_menu, unsafe_allow_html=True)
