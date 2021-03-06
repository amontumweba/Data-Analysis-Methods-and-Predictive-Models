# -*- coding: utf-8 -*-

import plotly

from google.colab import drive
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as offline

# offline.init_notebook_mode(connected=False)

drive.mount('/gdrive')

nifty_data = pd.read_csv('/gdrive/MyDrive/Files/NSEI.csv')

nifty_data['Date'] = pd.DatetimeIndex(nifty_data['Date'])

nifty_data.shape

nifty_data.drop(['Adj Close', 'Volume'], axis=1, inplace=True)

nifty_data = nifty_data.dropna(how='any', axis=0)

nifty_data.head()

nifty_data.isnull().sum()

# nifty_data = nifty_data.iloc[0:161]

nifty_data.head()

trace = go.Scatter(x = nifty_data['Date'], 
                   y = nifty_data['Close'], 
                   mode = 'markers', 
                   marker = dict(size= 7, color=nifty_data['Close'], 
                                 colorscale='Rainbow', 
                                 showscale=True, 
                                 opacity=0.5), 
                   text = nifty_data['Close'])

data = [trace]

layout = go.Layout(title= 'Stocks', 
                   hovermode='closest', 
                   xaxis= dict(title='Date'), 
                   yaxis= dict(title='Close'))

fig = go.Figure(data=data, layout=layout)

# offline.iplot(fig)
fig.show(renderer='colab')

nifty_data['Month'] = nifty_data['Date'].dt.month

trace = go.Box(x = nifty_data['Month'], 
               y = nifty_data['Close'])

data = [trace]

fig = go.Figure(data=data)

fig.show()

trace = go.Candlestick(x = nifty_data['Date'], 
                       open = nifty_data['Open'], 
                       high = nifty_data['High'], 
                       low = nifty_data['Low'], 
                       close = nifty_data['Close'])

data = [trace]


layout = go.Layout(title = 'Stocks', 
                   hovermode = 'closest', 
                   xaxis = dict(title = 'Date'), 
                   yaxis = dict(title = 'Close'))

fig = go.Figure(data=data, layout=layout)

fig.show(randerer='colab')

