# -*- coding: utf-8 -*-

!pip install geopandas

import plotly

from google.colab import drive
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as offline

# offline.init_notebook_mode(connected=True)

drive.mount('/gdrive')

trace = dict(type = 'scattergeo', 
             lon = [-87.6, 117.1, -120.74, 0], 
             lat = [41.8, 32.7, 47.75, 0], 
             
             marker = dict(size=10), 
             mode = 'markers')

data = [trace]

layout = dict(showlegend = False, 
              geo = dict(showland = True))

fig = go.Figure(data = data, 
           layout = layout)

fig.show()

trace = dict(type = 'scattergeo',
             lon = [6.6119, 7.185556, 9.467, 6.3667, 0], 
             lat = [2.4079, 1.988056, 1.617, 2.6167, 0], 
             marker = dict(size=7), 
             mode = 'markers')

data = [trace]

layout = dict(showlegend = False, 
              geo = dict(showland = True))

fig = go.Figure(data = data, 
           layout = layout)

fig.show()

parks_data = pd.read_csv('/gdrive/MyDrive/Files/parks.csv')

parks_data.head()

trace = dict(type = 'scattergeo',
             
             lat = parks_data['Latitude'], 
             lon = parks_data['Longitude'],

             text = parks_data[['Park Name', 'State']], 
             
             marker = dict(size = parks_data['Acres']/10000, 
                           sizemode = 'area', 
                           color = parks_data['Acres'], 
                           colorscale = 'Bluered', 
                           showscale = True), 
             
             mode = 'markers')


data = [trace]

layout = dict(title = 'National Parks', 
              showlegend = False, 
              
              geo = dict(showland = True, 
                         landcolor = 'skyblue'))


fig = go.Figure(data = data, 
                layout = layout)

fig.show(renderer='colab')

import plotly.express as px
import geopandas as gpd

# geo_df = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))

fig = px.scatter_geo(parks_data,
                    lat=parks_data['Latitude'],
                    lon=parks_data['Longitude'],
                    size = parks_data['Acres']/10000,
                    color = parks_data['Acres'], 
                    hover_name="State")
fig.show()

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_february_us_airport_traffic.csv')

df.head()

fig = px.scatter_geo(df,
                    lat=df['lat'],
                    lon=df['long'],
                    color = df['cnt'], 
                    size = df['cnt']/1000,
                    hover_name="state")
fig.show()

import os

path = 'https://raw.githubusercontent.com/plotly/datasets/master/'

def load_data(df = path):
  df = os.path.join(path, '2011_february_us_airport_traffic.csv')
  return pd.read_csv(df)
df_data = load_data()
df_data.head()

