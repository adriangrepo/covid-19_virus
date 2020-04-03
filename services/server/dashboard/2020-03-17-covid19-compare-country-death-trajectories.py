#!/usr/bin/env python
# coding: utf-8

# # Compare Country Trajectories - Death Rate
# > Comparing how countries death rate trajectories are similar with Italy, South Korea and Japan
# 
# - comments: false
# - author: Pratap Vardhan
# - categories: [growth, compare, death, interactive]
# - image: images/covid-compare-country-death-trajectories.png
# - permalink: /compare-country-death-trajectories/

# In[1]:


#hide
import pandas as pd
import altair as alt
from IPython.display import HTML


# In[2]:


#hide
url = ('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/'
       'csse_covid_19_time_series/time_series_19-covid-Deaths.csv')
df = pd.read_csv(url)
# rename countries
df['Country/Region'] = df['Country/Region'].replace({'Korea, South': 'South Korea'})
dt_cols = df.columns[~df.columns.isin(['Province/State', 'Country/Region', 'Lat', 'Long'])]


# In[3]:


#hide
dff = (df.groupby('Country/Region')[dt_cols].sum()
       .stack().reset_index(name='Confirmed Cases')
       .rename(columns={'level_1': 'Date', 'Country/Region': 'Country'}))
dff['Date'] = pd.to_datetime(dff['Date'], format='%m/%d/%y')


# In[4]:


#hide
MIN_CASES = 10
LAST_DATE = dt_cols[-1]
# sometimes last column may be empty, then go backwards
for c in dt_cols[::-1]:
    if not df[c].fillna(0).eq(0).all():
        LAST_DATE = c
        break
countries = dff[dff['Date'].eq(LAST_DATE) & dff['Confirmed Cases'].ge(MIN_CASES) & 
        dff['Country'].ne('China')
       ].sort_values(by='Confirmed Cases', ascending=False)
countries = countries['Country'].values


# In[5]:


#hide
SINCE_CASES_NUM = 10
COL_X = f'Days since {SINCE_CASES_NUM}th death'
dff2 = dff[dff['Country'].isin(countries)].copy()
days_since = (dff2.assign(F=dff2['Confirmed Cases'].ge(SINCE_CASES_NUM))
              .set_index('Date')
              .groupby('Country')['F'].transform('idxmax'))
dff2[COL_X] = (dff2['Date'] - days_since.values).dt.days.values
dff2 = dff2[dff2[COL_X].ge(0)]


# In[6]:


#hide
def get_country_colors(x):
    mapping = {
        'Italy': 'black',
        'Iran': '#A1BA59',
        'South Korea': '#E45756',
        'Spain': '#F58518',
        'Germany': '#9D755D',
        'France': '#F58518',
        'US': '#2495D3',
        'Switzerland': '#9D755D',
        'Norway': '#C1B7AD',
        'United Kingdom': '#2495D3',
        'Netherlands': '#C1B7AD',
        'Sweden': '#C1B7AD',
        'Belgium': '#C1B7AD',
        'Denmark': '#C1B7AD',
        'Austria': '#C1B7AD',
        'Japan': '#9467bd'}
    return mapping.get(x, '#C1B7AD')


# In[7]:


#hide_input
baseline_countries = ['Italy', 'South Korea', 'Japan']
max_date = dff2['Date'].max()
color_domain = list(dff2['Country'].unique())
color_range = list(map(get_country_colors, color_domain))

def make_since_chart(highlight_countries=[], baseline_countries=baseline_countries):
    selection = alt.selection_multi(fields=['Country'], bind='legend', 
                                    init=[{'Country': x} for x in highlight_countries + baseline_countries])

    base = alt.Chart(dff2, width=550).encode(
        x=f'{COL_X}:Q',
        y=alt.Y('Confirmed Cases:Q', scale=alt.Scale(type='log'), axis=alt.Axis(title='Cumulative Deaths')),
        color=alt.Color('Country:N', scale=alt.Scale(domain=color_domain, range=color_range)),
        tooltip=list(dff2),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.05))
    )
    max_day = dff2[COL_X].max()
    ref = pd.DataFrame([[x, SINCE_CASES_NUM*1.33**x] for x in range(max_day+1)], columns=[COL_X, 'Confirmed Cases'])
    base_ref = alt.Chart(ref).encode(x=f'{COL_X}:Q', y='Confirmed Cases:Q')
    return (
        base_ref.mark_line(color='black', opacity=.5, strokeDash=[3,3]) +
        base_ref.transform_filter(
            alt.datum[COL_X] >= max_day
        ).mark_text(dy=-6, align='right', fontSize=10, text='33% Daily Growth') +
        base.mark_line(point=True).add_selection(selection) + 
        base.transform_filter(
            alt.datum['Date'] >= int(max_date.timestamp() * 1000)
        ).mark_text(dy=-8, align='right', fontWeight='bold').encode(text='Country:N')
    ).properties(
        title=f"Compare {', '.join(highlight_countries)} death trajectory with {', '.join(baseline_countries)}"
    )


# ## Learning from Italy, South Korea & Japan

# Italy, South Korea & Japan are three countries which show different growth rates and how it evolved over time. 
# 
# **South Korea** and **Japan** have lower growth rate since thier 10 deaths. **Italy** has been grow close to 33% till 3 weeks since it's 10 deaths.
# 
# Where does your Country stand today?
# 
# Click (Shift+ for multiple) on Countries legend to filter the visualization.

# In[8]:


#hide_input
HTML(f'<small class="float-right">Last Updated on {pd.to_datetime(LAST_DATE).strftime("%B, %d %Y")}</small>')


# In[9]:


#hide_input
chart = make_since_chart()
chart


# In[10]:


#hide_input
chart2 = make_since_chart(['Spain', 'Germany'])
chart2


# In[11]:


#hide_input
chart3 = make_since_chart(['US', 'France'])
chart3


# In[12]:


#hide_input
chart4 = make_since_chart(['Germany', 'United Kingdom'])
chart4


# Select a country from the drop down list below to toggle  the visualization.

# In[13]:


#hide_input
base = alt.Chart(dff2, width=600).encode(
    x=f'{COL_X}:Q',
    y=alt.Y('Confirmed Cases:Q', scale=alt.Scale(type='log'), axis=alt.Axis(title='Cumulative Deaths')),
    color=alt.Color('Country:N', scale=alt.Scale(domain=color_domain, range=color_range), legend=None),
    tooltip=['Country', 'Date', 'Confirmed Cases', COL_X]
)

country_selection = alt.selection_single(
    name='Select', fields=['Country'], 
    bind=alt.binding_select(options=list(sorted(set(countries) - set(baseline_countries)))),
    init={'Country': 'US'})

date_filter = alt.datum['Date'] >= int(max_date.timestamp() * 1000)
base2 = base.transform_filter(alt.FieldOneOfPredicate('Country', baseline_countries))
base3 = base.transform_filter(country_selection)
base4 = base3.transform_filter(date_filter)

max_day = dff2[COL_X].max()
ref = pd.DataFrame([[x, SINCE_CASES_NUM*1.33**x] for x in range(max_day+1)], columns=[COL_X, 'Confirmed Cases'])
base_ref = alt.Chart(ref).encode(x=f'{COL_X}:Q', y='Confirmed Cases:Q')
base_ref_f = base_ref.transform_filter(alt.datum[COL_X] >= max_day)

chart5 = (
 base_ref.mark_line(color='black', opacity=.5, strokeDash=[3,3]) + 
 base_ref_f.mark_text(dy=-6, align='right', fontSize=10, text='33% Daily Growth') + 
 base2.mark_line(point=True, tooltip=True) +
 base3.mark_line(point={'size':50}, tooltip=True) +
 base2.transform_filter(date_filter).mark_text(dy=-8, align='right').encode(text='Country:N') +
 base4.mark_text(dx=8, align='left', fontWeight='bold').encode(text='Country:N') +
 base4.mark_text(dx=8, dy=12, align='left', fontWeight='bold').encode(text='Confirmed Cases:Q')
).add_selection(country_selection).properties(
    title=f"Country's death trajectory compared to {', '.join(baseline_countries)}"
)
chart5


# Interactive by [Pratap Vardhan](https://twitter.com/PratapVardhan)[^1]
# 
# [^1]: Source: ["2019 Novel Coronavirus COVID-19 (2019-nCoV) Data Repository by Johns Hopkins CSSE"](https://systems.jhu.edu/research/public-health/ncov/) [GitHub repository](https://github.com/CSSEGISandData/COVID-19). Link to [original notebook](https://github.com/pratapvardhan/notebooks/blob/master/covid19/covid19-compare-country-death-trajectories.ipynb).
