#!/bin/bash
curl https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv -o ../data/csse/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv
curl https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv -o ../data/csse/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv
curl https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv -o ../data/csse/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv

LATEST_DATA=`date +"%m-%d-%Y".csv`
echo ${LATEST_DATA}
curl https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/${LATEST_DATA} -o ../data/csse/csse_covid_19_data/csse_covid_19_daily_reports/${LATEST_DATA}
