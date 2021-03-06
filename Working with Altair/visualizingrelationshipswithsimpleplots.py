# -*- coding: utf-8 -*-

!pip install -U altair

!pip install -U altair vega_datasets notebook vega

import pandas as pd
import altair as alt
from vega_datasets import data

# alt.renderers.enable('notebook')

movies_data = data.movies()

movies_data.isnull().sum()

movies_data.dropna(inplace=True)

movies_data.shape

alt.Chart(movies_data, height=400, width=600).mark_boxplot(color='blue').encode(y='Worldwide_Gross:Q').properties(title='Movies')

alt.Chart(movies_data, height=-400, width=600).mark_point(color='darkcyan').encode(x='Production_Budget', y='Worldwide_Gross').properties(title='Production_Budget vs Worldwide_Gross')

alt.Chart(movies_data, height=400, width=600).\
mark_bar(size=20).\
encode(x='Major_Genre:O', 
       y='Worldwide_Gross:Q', 
       color='Major_Genre').\
       properties(title='Worldwide Gross for different Genres')

med_rating = movies_data['Rotten_Tomatoes_Rating'].median()

movies_data['above_average'] = (movies_data['Rotten_Tomatoes_Rating'] - med_rating) > 0

alt.Chart(movies_data, height=400, width=600).\
mark_point(color='darkcyan').encode(x='Production_Budget', y='Worldwide_Gross', 
                                    color='above_average').\
                                    properties(title='Production_Budget vs Worldwide_Gross')

alt.Chart(movies_data, height=400, width=150).\
mark_bar().encode(x='above_average:O', y='Worldwide_average:N', 
                  color='above_average:N', 
                  column='MPAA_Rating:N')

alt.Chart(movies_data, height=400, width=600).\
mark_bar().\
encode(x='US_Gross', y='MPAA_Rating', 
       color='MPAA_Rating', 
       order=alt.Order('MPAA_Rating', sort='ascending')).\
       properties(title='US Gross vs MPAA_Rating')

