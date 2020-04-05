#!/usr/bin/env python
# coding: utf-8

# # COVID-19 Tracking Europe Cases
# > Tracking coronavirus total cases, deaths and new cases by country.
# 
# - comments: true
# - author: Pratap Vardhan
# - categories: [overview, interactive, europe]
# - hide: true
# - permalink: /covid-overview-europe/

# In[1]:


#hide
print('''
Example of using jupyter notebook, pandas (data transformations), jinja2 (html, visual)
to create visual dashboards with fastpages
You see also the live version on https://gramener.com/enumter/covid19/europe.html
''')


# In[2]:


#hide
import numpy as np
import pandas as pd
from jinja2 import Template
from IPython.display import HTML


# In[3]:


#hide
from pathlib import Path
if not Path('covid_overview.py').exists():
    get_ipython().system(' wget https://raw.githubusercontent.com/pratapvardhan/notebooks/master/covid19/covid_overview.py')


# In[4]:


#hide
import covid_overview as covid


# In[5]:


#hide
COL_REGION = 'Country/Region'
europe_countries = covid.mapping['df'].pipe(lambda d: d[d['Continent'].eq('Europe')])['Name'].values
filter_europe = lambda d: d[d['Country/Region'].isin(europe_countries)]
kpis_info = [
    {'title': 'Italy', 'prefix': 'IT'},
    {'title': 'Spain', 'prefix': 'SP'},
    {'title': 'Germany', 'prefix': 'GE'}]
data = covid.gen_data(region=COL_REGION, filter_frame=filter_europe, kpis_info=kpis_info)


# In[6]:


#hide_input
template = Template(covid.get_template(covid.paths['overview']))
dt_cols, LAST_DATE_I = data['dt_cols'], data['dt_last']
html = template.render(
    D=data['summary'], table=data['table'],
    newcases=data['newcases'].loc[:, dt_cols[LAST_DATE_I - 40]:dt_cols[LAST_DATE_I]],
    COL_REGION=COL_REGION,
    KPI_CASE='Europe',
    KPIS_INFO=kpis_info,
    LEGEND_DOMAIN=[5, 50, 500, np.inf],
    np=np, pd=pd, enumerate=enumerate)
HTML(f'<div>{html}</div>')


# Visualizations by [Pratap Vardhan](https://twitter.com/PratapVardhan)[^1]
# 
# [^1]: Source: ["COVID-19 Data Repository by Johns Hopkins CSSE"](https://systems.jhu.edu/research/public-health/ncov/) [GitHub repository](https://github.com/CSSEGISandData/COVID-19). Link to [notebook](https://github.com/pratapvardhan/notebooks/blob/master/covid19/covid19-overview.ipynb), [orignal interactive](https://gramener.com/enumter/covid19/europe.html)
