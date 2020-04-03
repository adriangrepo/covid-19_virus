#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
sys.path.append('..')


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta, datetime, date
import os
from utils import data_paths, load_config
from pathlib import Path
from nltk.metrics import edit_distance #(Levenshtein)
import pycountry
import math


# # Estimating The Infected Population From Deaths
# > Estimating the number of infected people by country based on the number of deaths and case fatality rate. 
# 
# - comments: true
# - author: Joao B. Duarte
# - categories: [growth, compare, interactive, estimation]
# - hide: false
# - image: images/covid-estimate-infections.png
# - permalink: /covid-infected/
# - toc: true

# In[4]:


LOCAL_FILES=True
#jupyter or script
IS_SCRIPT = False


# In[5]:


os.getcwd()


# In[6]:


if IS_SCRIPT:
    RUN_PATH = Path(os.path.realpath(__file__))
    DATA_PARENT = RUN_PATH.parent.parent
else:
    #for jupyter
    cw = get_ipython().getoutput('pwd')
    RUN_PATH = Path(cw[0])
    DATA_PARENT = RUN_PATH.parent


# In[7]:


if IS_SCRIPT:
    csse_data = data_paths('tools/csse_data_paths.yml')
else:
    csse_data = data_paths('csse_data_paths.yml')


# In[8]:


if LOCAL_FILES:
    confirmed_url=csse_data.get("csse_ts_local", {}).get('confirmed', {})
    deaths_url=csse_data.get("csse_ts_local", {}).get('deaths', {})
    recovered_url=csse_data.get("csse_ts_local", {}).get('recovered', {})
    
    confirmed_url = str(DATA_PARENT/confirmed_url)
    deaths_url = str(DATA_PARENT/deaths_url)
    recovered_url = str(DATA_PARENT/recovered_url)
else:
    confirmed_url=csse_data.get("csse_ts_global", {}).get('confirmed', {})
    deaths_url=csse_data.get("csse_ts_global", {}).get('deaths', {})
    recovered_url=csse_data.get("csse_ts_global", {}).get('recovered', {})


# In[9]:


### UN stats


# In[10]:


df_un_pop_density_info=pd.read_csv(DATA_PARENT/'data/un/df_un_pop_density_info.csv')
df_un_urban_growth_info=pd.read_csv(DATA_PARENT/'data/un/urban_growth_info.csv')
df_un_health_info=pd.read_csv(DATA_PARENT/'data/un/df_un_health_info.csv')
df_un_tourism_info=pd.read_csv(DATA_PARENT/'data/un/df_un_tourism_info.csv')
df_un_gdp_info=pd.read_csv(DATA_PARENT/'data/un/df_un_gdp_info.csv')
df_un_edu_info=pd.read_csv(DATA_PARENT/'data/un/df_un_edu_info.csv')
df_un_pop_growth_info=pd.read_csv(DATA_PARENT/'data/un/df_un_pop_growth_info.csv')
df_un_gdrp_rnd_info=pd.read_csv(DATA_PARENT/'data/un/df_un_gdrp_rnd_info.csv')
df_un_education_info=pd.read_csv(DATA_PARENT/'data/un/df_un_education_info.csv')
df_un_sanitation_info=pd.read_csv(DATA_PARENT/'data/un/df_un_sanitation_info.csv')

df_un_health_expenditure_info=pd.read_csv(DATA_PARENT/'data/un/df_un_health_expenditure_info.csv')
df_un_immigration_info=pd.read_csv(DATA_PARENT/'data/un/df_un_immigration_info.csv')
df_un_trading_info=pd.read_csv(DATA_PARENT/'data/un/df_un_trading_info.csv')
df_un_land_info=pd.read_csv(DATA_PARENT/'data/un/df_un_land_info.csv')


# In[11]:


df_un_health_info.head()
#Health personnel: Pharmacists (per 1000 population)


# In[12]:


df_un_trading_info.tail(n=20)
#column Major trading partner 1 (% of exports)
#Major trading partner 1 (% of exports)
#Major trading partner 2 (% of exports)
#Major trading partner 3 (% of exports)


# In[13]:


df_population_density=df_un_pop_density_info.loc[df_un_pop_density_info['Series'] == 'Population density']


# In[14]:


df_population_density.tail(n=50)
#Population aged 60+ years old (percentage)
#Population density
#Population mid-year estimates (millions)


# In[15]:



df_population_density.loc[df_population_density.groupby('Country')['Year'].idxmax()]


# In[16]:


df_population_density


# In[17]:


### Freedom House stats


# In[18]:


#Freedon House stats
def country_freedom():
    global_freedom = str(DATA_PARENT/'data/freedom_house/Global_Freedom.csv')
    df_global_free = pd.read_csv(global_freedom)
    internet_freedom = str(DATA_PARENT/'data/freedom_house/Internet_Freedom.csv')
    df_internet_free = pd.read_csv(internet_freedom)
    return df_global_free, df_internet_free
df_global_freedom, df_internet_freedom = country_freedom()


# In[19]:


#csse countries
df_deaths = pd.read_csv(deaths_url, error_bad_lines=False)
df_confirmed = pd.read_csv(confirmed_url, error_bad_lines=False)
df_recovered = pd.read_csv(recovered_url, error_bad_lines=False)
csse_countries = []
for df in [df_deaths, df_confirmed, df_recovered]:
    c = set(df["Country/Region"].unique())
    csse_countries.append(c)
csse_countries = [item for sublist in csse_countries for item in sublist]
csse_countries = list(set(csse_countries))


# ## CSSE

# In[20]:


# Get data on deaths D_t
df_deaths = pd.read_csv(deaths_url, error_bad_lines=False)
df_deaths = df_deaths.drop(columns=["Lat", "Long"])
df_deaths = df_deaths.melt(id_vars= ["Province/State", "Country/Region"])
df_deaths = pd.DataFrame(df_deaths.groupby(['Country/Region', "variable"]).sum())
df_deaths.reset_index(inplace=True)  
df_deaths = df_deaths.rename(columns={"Country/Region": "location", "variable": "date", "value": "total_deaths"})
df_deaths['date'] =pd.to_datetime(df_deaths.date)
df_deaths = df_deaths.sort_values(by = "date")
df_deaths.loc[df_deaths.location == "US","location"] = "United States"
df_deaths.loc[df_deaths.location == "Korea, South","location"] = "South Korea"


# In[21]:


#confirmed


# In[22]:


df_confirmed = pd.read_csv(confirmed_url, error_bad_lines=False)
df_confirmed = df_confirmed.drop(columns=["Lat", "Long"])
df_confirmed = df_confirmed.melt(id_vars= ["Province/State", "Country/Region"])
df_confirmed = pd.DataFrame(df_confirmed.groupby(['Country/Region', "variable"]).sum())
df_confirmed.reset_index(inplace=True)  
df_confirmed = df_confirmed.rename(columns={"Country/Region": "location", "variable": "date", "value": "total_cases"})
df_confirmed['date'] =pd.to_datetime(df_confirmed.date)
df_confirmed = df_confirmed.sort_values(by = "date")
df_confirmed.loc[df_confirmed.location == "US","location"] = "United States"
df_confirmed.loc[df_confirmed.location == "Korea, South","location"] = "South Korea"


# In[23]:


df_confirmed.head()


# In[24]:


df_final = pd.merge(df_deaths,
                 df_confirmed)


# In[25]:


df_final.head()


# In[26]:


df_final["CFR"] = df_final["total_deaths"]/df_final["total_cases"]
df_final["total_infected"] = np.NaN
df_final = df_final.sort_values(by = ['location', 'date'])
df_final = df_final.reset_index(drop = True)


# In[27]:


df_un_pop_per_country=pd.read_csv(DATA_PARENT/'data/un/df_un_pop_per_country_info.csv')


# In[28]:


def get_country_list(pop_cutoff=5.0):
    pop_nmill=df_un_pop_per_country.loc[df_un_pop_per_country['Value'] >= pop_cutoff]
    countries_n_plus=pop_nmill.Country.tolist()
    return countries_n_plus


# In[29]:


csse_countries.sort()
csse_countries


# In[30]:


csse_countries=list(map(lambda x: x if x != 'Korea, South' else "South Kores", csse_countries))


# In[31]:


countries_n_plus = get_country_list(pop_cutoff=5.0)


# In[32]:


for j in countries_n_plus:
    if not j in csse_countries:
        print(j)


# In[33]:


for j in countries_n_plus:
    for i in df_final["date"].unique()[0:-8]:
        numer =  df_final.loc[(df_final.date == i + np.timedelta64(8, 'D')) & (df_final.location == j), "total_deaths"].iloc[0]
        denom = df_final.loc[(df_final.date == i + np.timedelta64(8, 'D')) & (df_final.location == j), "CFR"].iloc[0]
        df_final.loc[(df_final.date == i) & (df_final.location == j), "total_infected"] = numer/denom


# In[34]:


df_final.head()


# In[35]:


# Estimate growth rate of infected, g        
df_final['infected_g'] = np.log(df_final['total_infected'])
df_final['infected_g'] = df_final['infected_g'].diff() 


# In[36]:


# Estimate number of infected given g
today = df_final.date.iloc[-1]
for j in countries_n_plus:
    for i in range(7,-1,-1):
        df_final.loc[(df_final.location == j) & (df_final.date == today - timedelta(i)), "total_infected"] = df_final.loc[df_final.location == j, "total_infected"].iloc[-i-2]*(1+df_final.loc[df_final.location == j, "infected_g"].aggregate(func = "mean"))


# In[37]:


data_pc = df_final[['location', 'date', 'total_infected']].copy()


# In[38]:


data_countries = []
data_countries_pc = []


# In[39]:


for i in countries_n_plus:
    data_pc.loc[data_pc.location == i,"total_infected"] = data_pc.loc[data_pc.location == i,"total_infected"]


# In[40]:


# Get each country time series
filter1 = data_pc["total_infected"] > 1
for i in countries_n_plus:
    filter_country = data_pc["location"]== i
    data_countries_pc.append(data_pc[filter_country & filter1])      


# In[41]:


len(data_countries_pc)


# In[42]:


data_countries_pc[0]


# ## Estimated Infected Population By Country
# 
# by days since outbreak

# In[43]:


# Lastest Country Estimates  
label = 'Total_Infected'
temp = pd.concat([x.copy() for x in data_countries_pc]).loc[lambda x: x.date >= '3/1/2020']


# In[44]:


metric_name = f'{label}'
temp.columns = ['Country', 'Date', metric_name]
# temp.loc[:, 'month'] = temp.date.dt.strftime('%Y-%m')
temp.loc[:, "Total_Infected"] = temp.loc[:, "Total_Infected"].round(0)  
temp.groupby('Country').last()


# In[ ]:





# ## Infected vs. number of confirmed cases
# > Allows you to compare how countries have been tracking the true number of infected people. 
# The smaller deviation from the dashed line (45 degree line) the better job at tracking the true number of infected people.

# In[45]:


data_pc = df_final.copy()


# In[46]:


data_countries = []
data_countries_pc = []


# In[47]:


for i in countries_n_plus:
    data_pc.loc[data_pc.location == i,"total_infected"] = data_pc.loc[data_pc.location == i,"total_infected"]
    data_pc.loc[data_pc.location == i,"total_cases"] = data_pc.loc[data_pc.location == i,"total_cases"]
    # get each country time series
filter1 = data_pc["total_infected"] > 1
for i in countries_n_plus:
    filter_country = data_pc["location"]== i
    data_countries_pc.append(data_pc[filter_country & filter1])


# In[48]:


type(data_countries_pc[0])


# In[49]:


data_countries_pc[0]


# In[ ]:





# In[50]:


def get_df_country(country):
    for i, df in enumerate(data_countries_pc):
        if len(df.loc[df['location'] == country]):
            print(f'country: {country}, index: {i}')
        


# In[51]:


get_df_country('Italy')


# In[52]:


data_countries_pc[47]


# In[79]:


df_all_data_countries_pc=pd.concat(data_countries_pc)


# In[81]:


df_all_data_countries_pc.tail()


# In[ ]:


#### save all pred as one df


# In[82]:


df_all_data_countries_pc.to_csv(DATA_PARENT/'data/processed/csse/df_all_data_countries_pc.csv')


# In[ ]:





# In[ ]:


### Combine last day only pred with un and freedom house data


# In[53]:


df_country_un_stats = pd.read_csv(DATA_PARENT/'data/un/df_un_merged_stats.csv')


# In[60]:


df_country_un_stats.rename(columns={'Country': 'location'}, inplace=True)


# In[61]:


idx = data_countries_pc[0].groupby(['location'])['date'].transform(max) == data_countries_pc[0]['date']
sub_df=data_countries_pc[0][idx]
sub_df


# In[62]:


sub_df.iloc[0]['location']


# In[63]:


df_country_un_stats.head()


# In[ ]:





# In[ ]:


### freedom house


# In[72]:


df_freedomhouse_merged = pd.read_csv(DATA_PARENT/'data/freedom_house/df_freedomhouse_merged.csv')


# In[73]:


df_freedomhouse_merged.head()


# In[74]:


df_freedomhouse_merged.rename(columns={'Country': 'location'}, inplace=True)


# In[76]:


frames=[]
for df in data_countries_pc:
    idx = df.groupby(['location'])['date'].transform(max) == df['date']
    sub_df=df[idx]
    if len(sub_df)>0:
        #print(f'sub_df: {sub_df}')
        country=sub_df.iloc[0]['location']
        un_df=df_country_un_stats.loc[df_country_un_stats['location'] == country]
        #print(f'un_df: {un_df}')
        df_merged=pd.merge(sub_df, un_df)
        #freedom house data
        fh_df=df_freedomhouse_merged.loc[df_freedomhouse_merged['location'] == country]
        df_merged=pd.merge(df_merged, fh_df)
        frames.append(df_merged)
df_all_un_fh=pd.concat(frames)


# In[77]:


df_all_un_fh.head()


# In[78]:


df_all_un_fh.to_csv(DATA_PARENT/'data/processed/csse/df_data_countries_pc_latest.csv')


# In[ ]:





# ## Methodology

# We argue that the number of infected in the past can be infered using today's number of deaths and average fatality rate from confirmed cases in the following way:
# 
# {% raw %}
# $$ I_{t-j} = \frac{D_t}{{CFR}_t}$$
# {% endraw %}
# 
# where {% raw %}$I_t${% endraw %} = number of infected, {% raw %}$D_t${% endraw %} = number of deaths, and {% raw %}${CFR}_t ${% endraw %} = case fatality rate = {% raw %}$\frac{D}{C}${% endraw %}. The {% raw %}$j${% endraw %} depends on the average number of days that covid patients die after having the first symptoms.

# **Assumption 1**: The case fatality rate is a good proxy for the fatality rate of the infected population
# 

# Then, in order to estimate the current number of infected {% raw %}$I_t${% endraw %} we need to estimate its growth rate from {% raw %}$t-j${% endraw %} to {% raw %}$t${% endraw %}.
# 
# {% raw %}
# $$I_t = (1+\hat{g})^j I_{t-j}$$
# {% endraw %}

# **Assumption 2**: The growth rate of infected $\hat{g}$ is an unbiased estimate of $g$ .
# 
# For now we estimate $g$ using the average growth rate since having the first infected person.

# **Assumption 3**: It takes on average 8 days to day after having the first symptoms.

# This analysis was conducted by [Joao B. Duarte](https://www.jbduarte.com). Relevant sources are listed below: 
# 
# 
# 1. [2019 Novel Coronavirus COVID-19 (2019-nCoV) Data Repository by Johns Hopkins CSSE](https://systems.jhu.edu/research/public-health/ncov/) [GitHub repository](https://github.com/CSSEGISandData/COVID-19). 
# 
# 2. [Feenstra, Robert C., Robert Inklaar and Marcel P. Timmer (2015), "The Next Generation of the Penn World Table" American Economic Review, 105(10), 3150-3182](https://www.rug.nl/ggdc/productivity/pwt/related-research)
# 

# In[ ]:





# In[ ]:




