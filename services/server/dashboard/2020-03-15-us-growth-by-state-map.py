#!/usr/bin/env python
# coding: utf-8

# # Interactive Map - Confirmed Cases in the US by State
# > Interactive Visualizations of The Count and Growth of COVID-19 in the US.
# 
# - comments: true
# - author: Asif Imran
# - categories: [growth, usa, altair, interactive]
# - image: images/us-growth-state-map.png
# - permalink: /growth-map-us-states/

# In[1]:


#hide
import numpy as np
import pandas as pd
import altair as alt

alt.data_transformers.disable_max_rows()


# In[2]:


#hide
url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv'
confirmed_df = pd.read_csv(url)

cols = confirmed_df.columns

confirmed_df = confirmed_df.astype({'Province/State': 'str','Country/Region':str})

#confirmed_df.head()


# In[3]:


#hide
dcols = cols[cols.str.match(pat='\d+\/\d+\/\d+', case=False)]
rcols = cols[~cols.isin(dcols)]
df = pd.melt(confirmed_df, 
             id_vars=rcols, value_vars=dcols,
             var_name='date', 
             value_name='confirmed_count')
df.columns = df.columns.str.lower().str.replace('/','_')
df['date'] = pd.to_datetime(df['date'])
df = df.astype({'province_state': 'str','country_region':str})

#clean
df.loc[df.province_state.isna(),'province_state'] = ''


# In[4]:


#hide
abbr2state = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}

state2abbr = {s:a for a,s in abbr2state.items()}


# In[5]:


#hide
us_df = df[(df.country_region == 'US')].copy()

def clean_state(col):
    if ',' in col:
        return col.split(', ', 1)[1]
    else:
        return state2abbr.get(col)

us_df['state'] = us_df.province_state.apply(clean_state)
us_df['state'] = us_df.apply(lambda x: 'DP' if x['province_state'] == 'Diamond Princess' else x['state'], axis=1)

us_df.confirmed_count.fillna(0, inplace=True)
us_df.rename(columns={'country_region':'country'}, 
             inplace=True)

us_daily_df = (us_df
         .groupby(['state','country','date'])
         .agg(
             confirmed_count=('confirmed_count','sum'), 
             lat=('lat','mean'),
             long=('long','mean'))
        )

us_daily_df['new_cases'] = us_daily_df.confirmed_count.diff()
us_daily_df.loc[us_daily_df.new_cases < 0, 'new_cases'] = 0

#us_daily_df['cs'] = us_daily_df.sort_values('date').groupby(['state','country']).confirmed_count.cumsum()

us_daily_df = us_daily_df.reset_index()

#us_daily_df.head()

#active = use the recent date
state_df = us_daily_df.sort_values('date').groupby(['state','country']).tail(1)
#state_df.head()


#https://github.com/altair-viz/altair/issues/1005#issuecomment-403237407
def to_altair_datetime(dt):
    return alt.DateTime(year=dt.year, month=dt.month, date=dt.day,
                        hours=dt.hour, minutes=dt.minute, seconds=dt.second,
                        milliseconds=0.001 * dt.microsecond)


# In[6]:


#hide
states_data = 'https://vega.github.io/vega-datasets/data/us-10m.json'
states = alt.topo_feature(states_data, feature='states')
selector = alt.selection_single(empty='none', fields=['state'], nearest=True, init={'state':'CA'})

curr_date = state_df.date.max().date().strftime('%Y-%m-%d')
dmax = (us_daily_df.date.max() + pd.DateOffset(days=3))
dmin = us_daily_df.date.min()

# US states background
background = alt.Chart(states).mark_geoshape(
    fill='lightgray',
    stroke='white'
).properties(
    width=500,
    height=400
).project('albersUsa')


points = alt.Chart(state_df).mark_circle().encode(
    longitude='long:Q',
    latitude='lat:Q',
    size=alt.Size('confirmed_count:Q', title= 'Number of Confirmed Cases'),
    color=alt.value('steelblue'),
    tooltip=['state:N','confirmed_count:Q']
).properties(
    title=f'Total Confirmed Cases by State as of {curr_date}'
).add_selection(selector)


timeseries = alt.Chart(us_daily_df).mark_bar().properties(
    width=500,
    height=350,
    title="New Cases by Day",
).encode(
    x=alt.X('date:T', title='Date', timeUnit='yearmonthdate',
            axis=alt.Axis(format='%y/%m/%d', labelAngle=-30), 
            scale=alt.Scale(domain=[to_altair_datetime(dmin), to_altair_datetime(dmax)])),
    y=alt.Y('new_cases:Q',
             axis=alt.Axis(title='# of New Cases',titleColor='steelblue'),
    ),
    color=alt.Color('state:O'),
    tooltip=['state:N','date:T','confirmed_count:Q', 'new_cases:Q']
).transform_filter(
    selector
).add_selection(alt.selection_single()
)

timeseries_cs = alt.Chart(us_daily_df).mark_line(color='red').properties(
    width=500,
    height=350,
).encode(
    x=alt.X('date:T', title='Date', timeUnit='yearmonthdate', 
            axis=alt.Axis(format='%y/%m/%d', labelAngle=-30),
            scale=alt.Scale(domain=[to_altair_datetime(dmin), to_altair_datetime(dmax)])),
    y=alt.Y('confirmed_count:Q',
             #scale=alt.Scale(type='log'),
             axis=alt.Axis(title='# of Confirmed Cases', titleColor='red'),
    ),
).transform_filter(
    selector
).add_selection(alt.selection_single(nearest=True)
)



final_chart = alt.vconcat(
    background + points, 
    alt.layer(timeseries, timeseries_cs).resolve_scale(y='independent'),
).resolve_scale(
    color='independent',
    shape='independent',
).configure(
    padding={'left':10, 'bottom':40}
).configure_axis(
    labelFontSize=10,
    labelPadding=10,
    titleFontSize=12,
).configure_view(
     stroke=None
)


# ### Click On State To Filter Chart Below

# In[7]:


#hide_input
final_chart


# NB: Cruise ship, the "Diamond Princess" is represented by state = DP
# 
# Prepared by [Asif Imran](https://twitter.com/5sigma)[^1]
# 
# [^1]: Source: ["2019 Novel Coronavirus COVID-19 (2019-nCoV) Data Repository by Johns Hopkins CSSE"](https://systems.jhu.edu/research/public-health/ncov/) [GitHub repository](https://github.com/CSSEGISandData/COVID-19).  This code is provided under the [BSD-3 License](https://github.com/twiecki/covid19/blob/master/LICENSE). 
