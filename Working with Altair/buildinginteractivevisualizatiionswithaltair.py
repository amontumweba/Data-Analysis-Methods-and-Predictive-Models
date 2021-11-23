# -*- coding: utf-8 -*-

!pip install -U altair

!pip install -U altair vega_datasets notebook vega

import pandas as pd
import altair as alt
from vega_datasets import data

weather_data = data.seattle_weather()

weather_data.head()

weather_data['year'] = weather_data['date'].dt.year
weather_data['month'] = weather_data['date'].dt.month
weather_data['day'] = weather_data['date'].dt.day

weather_data.drop('date', axis=1, inplace=True)

weather_data.head()

alt.Chart(weather_data, height=500, width=700).mark_point().\
encode(x='temp_max:Q', 
       y='wind:Q', color='weather:N', 
       tooltip=['weather', 'temp_max']).interactive()

brush = alt.selection(type='interval', encodings=['x'])

bars = alt.Chart(height=400, width=600).mark_bar(color='limegreen').\
encode(x='month:O', y='mean(temp_max):Q', opacity=alt.condition(brush, 
                                                                alt.OpacityValue(1), 
                                                                alt.OpacityValue(0.5))).add_selection(brush)

line = alt.Chart().mark_rule(color='red').encode(y='mean(temp_max):Q', size=alt.SizeValue(5)).transform_filter(brush)

alt.layer(bars, line, data=weather_data)

chart = alt.Chart(weather_data, height=400, width=600).mark_point().encode(y='wind:Q', 
                                                                           color=alt.condition(brush, 
                                                                                               'weather:N', alt.value('lightgray'))).\
                                                                                               properties(width=250, height=250).\
                                                                                               add_selection(brush)

chart.encode(x='precipitation:Q') | chart.encode(x='temp_max:Q')

slider = alt.binding_range(min=2012, max=2015, step=1)

select_year = alt.selection_single(name='year', fields=['year'], 
                                   bind=slider, init={'year': 2012})

color = alt.Scale(domain=('drizzle', 'rain', 'sun', 'snow', 'fog'), 
                  range=['steelblue', 'yellow', 'red', 'green', 'violet'])

alt.Chart(weather_data, height=400, width=600).mark_bar().\
encode(x=alt.X('weather:N', title=None), y=alt.Y('temp_max:Q', scale=alt.Scale(domain=(0, 40))), 
       color=alt.Color('weather:N', scale=color), column='month:Q', tooltip=['precipitation'])\
       .properties(width=50)\
       .add_selection(select_year)\
       .transform_filter(select_year)\
       .configure_facet(spacing=8)

