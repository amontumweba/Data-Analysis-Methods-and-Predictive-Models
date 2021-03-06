# -*- coding: utf-8 -*-

!pip install -U altair

!pip install -U altair vega_datasets notebook vega

import pandas as pd
import altair as alt
from vega_datasets import data

# alt.renderers.enable('notebook')

movies_data = data.movies()

movies_data.dropna(inplace=True)

alt.Chart(movies_data, height=400, width=600).mark_bar().encode(alt.X('IMDB_Rating', bin=True), y='count()', color='count()')

alt.Chart(movies_data, height=400, width=600).\
mark_bar().encode(alt.X('Rotten_Tomatoes_Rating', bin=True), 
                  y='count()', color='count()')

alt.Chart(movies_data, height=400, width=600).\
mark_bar().encode(alt.X('Running_Time_min', bin=True), 
                  alt.Y('IMDB_Rating', bin=True), color='count()')

alt.Chart(movies_data, height=400, width=600).mark_circle().\
encode(alt.X('Running_Time_min', bin=True), alt.Y('IMDB_Rating', bin=True), 
       size='IMDB_Rating', color='average(Rotten_Tomatoes_Rating)')

alt.Chart(movies_data, height=400, width=600)\
.mark_rect()\
.encode(alt.X('Production_Budget', bin=True), 
        alt.Y('Worldwide_Gross', bin=True), 
        color='average(Rotten_Tomatoes_Rating)')

