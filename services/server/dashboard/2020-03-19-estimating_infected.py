#!/usr/bin/env python
# coding: utf-8

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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from datetime import timedelta, datetime, date
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

chart_width = 550
chart_height= 400


def plot(data, type1, levels):
    data_countries_pc2 = data.copy()
    for i in range(0,len(countries)):
        data_countries_pc2[i] = data_countries_pc2[i].reset_index()
        data_countries_pc2[i]['n_days'] = data_countries_pc2[i].index
        if type1 == "scatter":
            data_countries_pc2[i]['cases'] = data_countries_pc2[i]["total_cases"]
        data_countries_pc2[i]['infected'] = data_countries_pc2[i]["total_infected"]
    data_plot = data_countries_pc2[0]
    for i in range(1, len(countries)):    
        data_plot = pd.concat([data_plot, data_countries_pc2[i]], axis=0)
    
    if type1 == "scatter":
        data_plot["45_line"] = data_plot["cases"]

    # Plot it using Altair
    source = data_plot
    
    if levels == True:
        ylabel = "Total"
    else :
        ylabel = "Per Million"

    scales = alt.selection_interval(bind='scales')
    selection = alt.selection_multi(fields=['location'], bind='legend')

    if type1 == "line": 
        base = alt.Chart(source, title =  "Estimated Infected Population By Country").encode(
            x = alt.X('n_days:Q', title = "Days since outbreak"),
            y = alt.Y("infected:Q",title = ylabel),
            color = alt.Color('location:N', legend=alt.Legend(title="Country", labelFontSize=15, titleFontSize=17),
                             scale=alt.Scale(scheme='tableau20')),
            opacity = alt.condition(selection, alt.value(1), alt.value(0.1))
        )
    
        lines = base.mark_line().add_selection(
            scales
        ).add_selection(
            selection
        ).properties(
            width=chart_width,
            height=chart_height
        )
        return(
        ( lines)
        .configure_title(fontSize=20)
        .configure_axis(labelFontSize=15,titleFontSize=18)
        )
    
    if levels == True:
        ylabel = "Infected"
        xlabel = "Cases"
    else :
        ylabel = "Per Million Infected"
        xlabel = "Per Million Cases"
        
    if type1 == "scatter":
        base = alt.Chart(source, title = "COVID-19 Cases VS Infected").encode(
            x = alt.X('cases:Q', title = xlabel),
            y = alt.Y("infected:Q",title = ylabel),
            color = alt.Color('location:N', legend=alt.Legend(title="Country", labelFontSize=15, titleFontSize=17),
                             scale=alt.Scale(scheme='tableau20')),
            opacity = alt.condition(selection, alt.value(1), alt.value(0.1))
        )


        
        scatter = base.mark_point().add_selection(
            scales
        ).add_selection(
            selection
        ).properties(
            width=chart_width,
            height=chart_height
        )

        line_45 = alt.Chart(source).encode(
            x = "cases:Q",
            y = alt.Y("45_line:Q",  scale=alt.Scale(domain=(0, max(data_plot["infected"])))),
        ).mark_line(color="grey", strokeDash=[3,3])
        
        return(
        (scatter + line_45)
        .configure_title(fontSize=20)
        .configure_axis(labelFontSize=15,titleFontSize=18)
        )

# Get data on deaths D_t

ddata = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv", error_bad_lines=False)
data = data.drop(columns=["Lat", "Long"])
data = data.melt(id_vars= ["Province/State", "Country/Region"])
data = pd.DataFrame(data.groupby(['Country/Region', "variable"]).sum())
data.reset_index(inplace=True)  
data = data.rename(columns={"Country/Region": "location", "variable": "date", "value": "total_deaths"})
data['date'] =pd.to_datetime(data.date)
data = data.sort_values(by = "date")
data.loc[data.location == "US","location"] = "United States"
data.loc[data.location == "Korea, South","location"] = "South Korea"


data_cases = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv", error_bad_lines=False)
data_cases = data_cases.drop(columns=["Lat", "Long"])
data_cases = data_cases.melt(id_vars= ["Province/State", "Country/Region"])
data_cases = pd.DataFrame(data_cases.groupby(['Country/Region', "variable"]).sum())
data_cases.reset_index(inplace=True)  
data_cases = data_cases.rename(columns={"Country/Region": "location", "variable": "date", "value": "total_cases"})
data_cases['date'] =pd.to_datetime(data_cases.date)
data_cases = data_cases.sort_values(by = "date")
data_cases.loc[data_cases.location == "US","location"] = "United States"
data_cases.loc[data_cases.location == "Korea, South","location"] = "South Korea"
countries = ["China", "Italy", "Spain", "France", "United Kingdom", "Germany", 
             "Portugal", "United States", "Singapore","South Korea", "Japan", 
             "Brazil","Iran"]

data_final = pd.merge(data,
                 data_cases
                 )
data_final["CFR"] = data_final["total_deaths"]/data_final["total_cases"]


data_final["total_infected"] = np.NaN
data_final = data_final.sort_values(by = ['location', 'date'])
data_final = data_final.reset_index(drop = True)


for j in countries:
    for i in data_final["date"].unique()[0:-8]:
        data_final.loc[(data_final.date == i) & (data_final.location == j), "total_infected"] = data_final.loc[(data_final.date == i + np.timedelta64(8, 'D')) & (data_final.location == j), "total_deaths"].iloc[0]/data_final.loc[(data_final.date == i + np.timedelta64(8, 'D')) & (data_final.location == j), "CFR"].iloc[0]
        

# Estimate growth rate of infected, g        
data_final['infected_g'] = np.log(data_final['total_infected'])
data_final['infected_g'] = data_final['infected_g'].diff() 

# Estimate number of infected given g
today = data_final.date.iloc[-1]
for j in countries:
    for i in range(7,-1,-1):
        data_final.loc[(data_final.location == j) & (data_final.date == today - timedelta(i)), "total_infected"] = data_final.loc[data_final.location == j, "total_infected"].iloc[-i-2]*(1+data_final.loc[data_final.location == j, "infected_g"].aggregate(func = "mean"))
        
data_pc = data_final[['location', 'date', 'total_infected']].copy()

countries = ["China", "Italy", "Spain", "France", "United Kingdom", "Germany", 
             "Portugal", "United States", "Singapore","South Korea", "Japan", 
             "Brazil","Iran"]
data_countries = []
data_countries_pc = []

for i in countries:
    data_pc.loc[data_pc.location == i,"total_infected"] = data_pc.loc[data_pc.location == i,"total_infected"]

# Get each country time series
filter1 = data_pc["total_infected"] > 1
for i in countries:
    filter_country = data_pc["location"]== i
    data_countries_pc.append(data_pc[filter_country & filter1])      


# ## Estimated Infected Population By Country
# 
# by days since outbreak
# 
# > Tip: Click (Shift+ for multiple) on countries in the legend to filter the visualization. 
# Plot estimated absolute number of infected
plot(data_countries_pc, "line", True)


# Lastest Country Estimates  
label = 'Infected - Total'
temp = pd.concat([x.copy() for x in data_countries_pc]).loc[lambda x: x.date >= '3/1/2020']

metric_name = f'{label}'
temp.columns = ['Country', 'Date', metric_name]
# temp.loc[:, 'month'] = temp.date.dt.strftime('%Y-%m')
temp.loc[:, "Infected - Total"] = temp.loc[:, "Infected - Total"].round(0)  
temp.groupby('Country').last()


# ## Infected vs. number of confirmed cases
# > Allows you to compare how countries have been tracking the true number of infected people. 
# The smaller deviation from the dashed line (45 degree line) the better job at tracking the true number of infected people.

# > Tip: Click (Shift+ for multiple) on countries in the legend to filter the visualization. 

# Plot it using Altair
data_pc = data_final.copy()

countries = ["Italy", "Spain", "France", "United Kingdom", "Germany", 
             "Portugal", "United States", "Singapore","South Korea", "Japan", 
             "Brazil","Iran"]
data_countries = []
data_countries_pc = []

for i in countries:
    data_pc.loc[data_pc.location == i,"total_infected"] = data_pc.loc[data_pc.location == i,"total_infected"]
    data_pc.loc[data_pc.location == i,"total_cases"] = data_pc.loc[data_pc.location == i,"total_cases"]
    # get each country time series
filter1 = data_pc["total_infected"] > 1
for i in countries:
    filter_country = data_pc["location"]== i
    data_countries_pc.append(data_pc[filter_country & filter1])


plot(data_countries_pc, "scatter", True)


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





