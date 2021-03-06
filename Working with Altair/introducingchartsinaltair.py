# -*- coding: utf-8 -*-

!pip install -U altair

!pip install -U altair vega_datasets notebook vega

from google.colab import drive
import pandas as pd
import altair as alt



import vega
from vega_datasets import data

drive.mount('/gdrive')

# alt.renderers.enable('notebook')

print(data.list_datasets())

data.unemployment_across_industries.url

unemployment_data = data.unemployment_across_industries()

alt.Chart(unemployment_data, height=400, width=600).mark_point().encode(x='date', y='count').properties(title='US Unemployment')

alt.Chart(unemployment_data, height=400, width=600).mark_point().encode(x='date', y='count', color='series').properties(title='US Unemployment')

alt.Chart(unemployment_data, height=400, width=400).mark_boxplot(extent=500).encode(x='year:O', y='count:Q').properties(title='US Unemployment')

year_data = pd.DataFrame(unemployment_data.groupby('year', as_index=False)['count'].sum())

alt.Chart(year_data, height=400, width=600).mark_bar(color='red', size=20).encode(x='year:O', y='count:Q').properties(title=('US Unemployment'))

