#!/usr/bin/env python
# coding: utf-8

# modified after:
# https://github.com/github/covid19-dashboard
# # Estimating The Mortality Rate For COVID-19
# > Using Country-Level Covariates To Correct For Testing & Reporting Biases And Estimate a True Mortality Rate.
# - author: Joseph Richards
# - image: images/corvid-mortality.png
# - comments: true
# - categories: [MCMC, mortality]
# - permalink: /covid-19-mortality-estimation/
# - toc: true


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
from datetime import datetime
import os

import numpy as np
import pandas as pd
import arviz as az

CURR_DIR = os.path.dirname(os.path.abspath(__file__))

# constants
ignore_countries = [
    'Others',
    'Cruise Ship'
]

cpi_country_mapping = {
    'United States of America': 'US',
    'China': 'Mainland China'
}

wb_country_mapping = {
    'United States': 'US',
    'Egypt, Arab Rep.': 'Egypt',
    'Hong Kong SAR, China': 'Hong Kong',
    'Iran, Islamic Rep.': 'Iran',
    'China': 'Mainland China',
    'Russian Federation': 'Russia',
    'Slovak Republic': 'Slovakia',
    'Korea, Rep.': 'Korea, South'
}

wb_covariates = [
    ('SH.XPD.OOPC.CH.ZS',
        'healthcare_oop_expenditure'),
    ('SH.MED.BEDS.ZS',
        'hospital_beds'),
    ('HD.HCI.OVRL',
        'hci'),
    ('SP.POP.65UP.TO.ZS',
        'population_perc_over65'),
    ('SP.RUR.TOTL.ZS',
        'population_perc_rural')
]

# data loading and manipulation
def get_all_data():
    '''
    Main routine that grabs all COVID and covariate data and
    returns them as a single dataframe that contains:

    * count of cumulative cases and deaths by country (by today's date)
    * days since first case for each country
    * CPI gov't transparency index
    * World Bank data on population, healthcare, etc. by country
    '''

    all_covid_data = _get_latest_covid_timeseries()

    covid_cases_rollup = _rollup_by_country(all_covid_data['Confirmed'])
    covid_deaths_rollup = _rollup_by_country(all_covid_data['Deaths'])

    todays_date = covid_cases_rollup.columns.max()

    # Create DataFrame with today's cumulative case and death count, by country
    df_out = pd.DataFrame({'cases': covid_cases_rollup[todays_date],
                           'deaths': covid_deaths_rollup[todays_date]})

    _clean_country_list(df_out)
    _clean_country_list(covid_cases_rollup)

    # Add observed death rate:
    df_out['death_rate_observed'] = df_out.apply(
        lambda row: row['deaths'] / float(row['cases']),
        axis=1)

    # Add covariate for days since first case
    df_out['days_since_first_case'] = _compute_days_since_first_case(
        covid_cases_rollup)

    # Add CPI covariate:
    _add_cpi_data(df_out)

    # Add World Bank covariates:
    _add_wb_data(df_out)

    # Drop any country w/o covariate data:
    num_null = df_out.isnull().sum(axis=1)
    to_drop_idx = df_out.index[num_null > 1]
    print('Dropping %i/%i countries due to lack of data' %
          (len(to_drop_idx), len(df_out)))
    df_out.drop(to_drop_idx, axis=0, inplace=True)

    return df_out, todays_date


def _get_latest_covid_timeseries():
    ''' Pull latest time-series data from JHU CSSE database '''

    repo = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/'
    data_path = 'csse_covid_19_data/csse_covid_19_time_series/'

    all_data = {}
    for status in ['Confirmed', 'Deaths', 'Recovered']:
        file_name = 'time_series_19-covid-%s.csv' % status
        all_data[status] = pd.read_csv(
            '%s%s%s' % (repo, data_path, file_name))

    return all_data


def _rollup_by_country(df):
    '''
    Roll up each raw time-series by country, adding up the cases
    across the individual states/provinces within the country

    :param df: Pandas DataFrame of raw data from CSSE
    :return: DataFrame of country counts
    '''
    gb = df.groupby('Country/Region')
    df_rollup = gb.sum()
    df_rollup.drop(['Lat', 'Long'], axis=1, inplace=True, errors='ignore')
    
    # Drop dates with all 0 count data
    df_rollup.drop(df_rollup.columns[df_rollup.sum(axis=0) == 0],
                   axis=1,
                   inplace=True)

    # Convert column strings to dates:
    idx_as_dt = [datetime.strptime(x, '%m/%d/%y') for x in df_rollup.columns]
    df_rollup.columns = idx_as_dt
    return df_rollup


def _clean_country_list(df):
    ''' Clean up input country list in df '''
    # handle recent changes in country names:
    country_rename = {
        'Hong Kong SAR': 'Hong Kong',
        'Taiwan*': 'Taiwan',
        'Czechia': 'Czech Republic',
        'Brunei': 'Brunei Darussalam',
        'Iran (Islamic Republic of)': 'Iran',
        'Viet Nam': 'Vietnam',
        'Russian Federation': 'Russia',
        'Republic of Korea': 'South Korea',
        'Republic of Moldova': 'Moldova',
        'China': 'Mainland China'
    }
    df.rename(country_rename, axis=0, inplace=True)
    df.drop(ignore_countries, axis=0, inplace=True, errors='ignore')


def _compute_days_since_first_case(df_cases):
    ''' Compute the country-wise days since first confirmed case

    :param df_cases: country-wise time-series of confirmed case counts
    :return: Series of country-wise days since first case
    '''
    date_first_case = df_cases[df_cases > 0].idxmin(axis=1)
    days_since_first_case = date_first_case.apply(
        lambda x: (df_cases.columns.max() - x).days)
    # Add 1 month for China, since outbreak started late 2019:
    days_since_first_case.loc['Mainland China'] += 30

    return days_since_first_case


def _add_cpi_data(df_input):
    '''
    Add the Government transparency (CPI - corruption perceptions index)
    data (by country) as a column in the COVID cases dataframe.

    :param df_input: COVID-19 data rolled up country-wise
    :return: None, add CPI data to df_input in place
    '''
    cpi_data = pd.read_excel(
        'https://github.com/jwrichar/COVID19-mortality/blob/master/data/CPI2019.xlsx?raw=true',
        skiprows=2)
    cpi_data.set_index('Country', inplace=True, drop=True)
    cpi_data.rename(cpi_country_mapping, axis=0, inplace=True)

    # Add CPI score to input df:
    df_input['cpi_score_2019'] = cpi_data['CPI score 2019']


def _add_wb_data(df_input):
    '''
    Add the World Bank data covariates as columns in the COVID cases dataframe.

    :param df_input: COVID-19 data rolled up country-wise
    :return: None, add World Bank data to df_input in place
    '''
    wb_data = pd.read_csv(
        'https://raw.githubusercontent.com/jwrichar/COVID19-mortality/master/data/world_bank_data.csv',
        na_values='..')

    for (wb_name, var_name) in wb_covariates:
        wb_series = wb_data.loc[wb_data['Series Code'] == wb_name]
        wb_series.set_index('Country Name', inplace=True, drop=True)
        wb_series.rename(wb_country_mapping, axis=0, inplace=True)

        # Add WB data:
        df_input[var_name] = _get_most_recent_value(wb_series)


def _get_most_recent_value(wb_series):
    '''
    Get most recent non-null value for each country in the World Bank
    time-series data
    '''
    ts_data = wb_series[wb_series.columns[3::]]

    def _helper(row):
        row_nn = row[row.notnull()]
        if len(row_nn):
            return row_nn[-1]
        else:
            return np.nan

    return ts_data.apply(_helper, axis=1)


# # Observed mortality rates

def mortality(df, todays_date):
    print(f'Data as of {todays_date}')

    reported_mortality_rate = df['deaths'].sum() / df['cases'].sum()
    print(f'Overall reported mortality rate: {(100.0 * reported_mortality_rate)}')

    df_highest = df.sort_values('cases', ascending=False).head(15)
    mortality_rate = pd.Series(
        data=(df_highest['deaths']/df_highest['cases']).values,
        index=map(lambda x: '%s (%i cases)' % (x, df_highest.loc[x]['cases']),
                df_highest.index))
    ax = mortality_rate.plot.bar(
        figsize=(14,7), title='Reported Mortality Rate by Country (countries w/ highest case counts)')
    ax.axhline(reported_mortality_rate, color='k', ls='--')

    plt.savefig(CURR_DIR+'images/mortality_observed.png')
    return reported_mortality_rate, mortality_rate



# # Model

# Estimate COVID-19 mortality rate, controling for country factors.
def initialize_model(df):
    print('>>initialize_model')
    # Normalize input covariates in a way that is sensible:

    # (1) days since first case: upper
    # mu_0 to reflect asymptotic mortality rate months after outbreak
    _normalize_col(df, 'days_since_first_case', how='upper')
    # (2) CPI score: upper
    # mu_0 to reflect scenario in absence of corrupt govts
    _normalize_col(df, 'cpi_score_2019', how='upper')
    # (3) healthcare OOP spending: mean
    # not sure which way this will go
    _normalize_col(df, 'healthcare_oop_expenditure', how='mean')
    # (4) hospital beds: upper
    # more beds, more healthcare and tests
    _normalize_col(df, 'hospital_beds', how='mean')
    # (5) hci = human capital index: upper
    # HCI measures education/health; mu_0 should reflect best scenario
    _normalize_col(df, 'hci', how='mean')
    # (6) % over 65: mean
    # mu_0 to reflect average world demographic
    _normalize_col(df, 'population_perc_over65', how='mean')
    # (7) % rural: mean
    # mu_0 to reflect average world demographic
    _normalize_col(df, 'population_perc_rural', how='mean')

    n = len(df)
    print(f'--initialize_model, n: {n}')
    covid_mortality_model = pm.Model()

    with covid_mortality_model:
        print(f'--initialize_model, generating priors')
        # Priors:
        mu_0 = pm.Beta('mu_0', alpha=0.3, beta=10)
        sig_0 = pm.Uniform('sig_0', lower=0.0, upper=mu_0 * (1 - mu_0))
        beta = pm.Normal('beta', mu=0, sigma=5, shape=7)
        sigma = pm.HalfNormal('sigma', sigma=5)

        print(f'--initialize_model, modelling mu')
        # Model mu from country-wise covariates:
        # Apply logit transformation so logistic regression performed
        mu_0_logit = np.log(mu_0 / (1 - mu_0))
        mu_est = mu_0_logit +             beta[0] * df['days_since_first_case_normalized'].values +             beta[1] * df['cpi_score_2019_normalized'].values +             beta[2] * df['healthcare_oop_expenditure_normalized'].values +             beta[3] * df['hospital_beds_normalized'].values +             beta[4] * df['hci_normalized'].values +             beta[5] * df['population_perc_over65_normalized'].values +             beta[6] * df['population_perc_rural_normalized'].values
        mu_model_logit = pm.Normal('mu_model_logit',
                                   mu=mu_est,
                                   sigma=sigma,
                                   shape=n)
        # Transform back to probability space:
        mu_model = np.exp(mu_model_logit) / (np.exp(mu_model_logit) + 1)

        print(f'--initialize_model, modelling tau')
        # tau_i, mortality rate for each country
        # Parametrize with (mu, sigma)
        # instead of (alpha, beta) to ease interpretability.
        tau = pm.Beta('tau', mu=mu_model, sigma=sig_0, shape=n)
        # tau = pm.Beta('tau', mu=mu_0, sigma=sig_0, shape=n)

        print(f'--initialize_model, modelling binomial likelihood')
        # Binomial likelihood:
        d_obs = pm.Binomial('d_obs',
                            n=df['cases'].values,
                            p=tau,
                            observed=df['deaths'].values)

    return covid_mortality_model


def _normalize_col(df, colname, how='mean'):
    '''
    Normalize an input column in one of 3 ways:

    * how=mean: unit normal N(0,1)
    * how=upper: normalize to [-1, 0] with highest value set to 0
    * how=lower: normalize to [0, 1] with lowest value set to 0

    Returns df modified in place with extra column added.
    '''
    colname_new = '%s_normalized' % colname
    if how == 'mean':
        mu = df[colname].mean()
        sig = df[colname].std()
        df[colname_new] = (df[colname] - mu) / sig
    elif how == 'upper':
        maxval = df[colname].max()
        minval = df[colname].min()
        df[colname_new] = (df[colname] - maxval) / (maxval - minval)
    elif how == 'lower':
        maxval = df[colname].max()
        minval = df[colname].min()
        df[colname_new] = (df[colname] - minval) / (maxval - minval)

def get_cleaned_data():
    # Load the data (see source/data.py):
    df, todays_date = get_all_data()
    # Impute NA's column-wise:
    df = df.apply(lambda x: x.fillna(x.mean()),axis=0)
    return df, todays_date

def fig_test():
    print('>>fig_test')
    rng = np.arange(50)
    rnd = np.random.randint(0, 10, size=(3, rng.size))
    yrs = 1950 + rng

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.stackplot(yrs, rng + rnd, labels=['A', 'B', 'C'])
    ax.set_xlim(xmin=yrs[0], xmax=yrs[-1])
    img_pth = CURR_DIR+'/images/testfig.png'
    plt.savefig(img_pth)
    return img_pth

def run_mcmc_model():
    df, todays_date = get_cleaned_data()
    #get observed mortality
    reported_mortality_rate, mortality_rate=mortality(df, todays_date)

    # Initialize the model:
    mod = initialize_model(df)

    # Run MCMC sampler1
    with mod:
        trace = pm.sample(300, tune=100,
                        chains=3, cores=2)


    n_samp = len(trace['mu_0'])
    mu0_summary = pm.summary(trace).loc['mu_0']
    print("COVID-19 Global Mortality Rate Estimation:")
    print("Posterior mean: %0.2f%%" % (100*trace['mu_0'].mean()))
    print("Posterior median: %0.2f%%" % (100*np.median(trace['mu_0'])))
    lower = np.sort(trace['mu_0'])[int(n_samp*0.025)]
    upper = np.sort(trace['mu_0'])[int(n_samp*0.975)]
    print("95%% posterior interval: (%0.2f%%, %0.2f%%)" % (100*lower, 100*upper))
    prob_lt_reported = sum(trace['mu_0'] < reported_mortality_rate) / len(trace['mu_0'])
    print("Probability true rate less than reported rate (%.2f%%) = %.2f%%" %
        (100*reported_mortality_rate, 100*prob_lt_reported))
    print("")

    # Posterior plot for mu0
    print('Posterior probability density for COVID-19 mortality rate, controlling for country factors:')
    ax = pm.plot_posterior(trace, var_names=['mu_0'], figsize=(18, 8), textsize=18,
                        credible_interval=0.95, bw=3.0, lw=3, kind='kde',
                        ref_val=round(reported_mortality_rate, 3))
    plt.savefig(CURR_DIR+'/images/mortality_posterior_pd.png')

    # ## Magnitude and Significance of Factors 
    # 
    # For bias in reported COVID-19 mortality rate

    # Posterior summary for the beta parameters:
    beta_summary = pm.summary(trace).head(7)
    beta_summary.index = ['days_since_first_case', 'cpi', 'healthcare_oop', 'hospital_beds', 'hci', 'percent_over65', 'percent_rural']
    beta_summary.reset_index(drop=False, inplace=True)

    err_vals = ((beta_summary['hpd_3%'] - beta_summary['mean']).values,
                (beta_summary['hpd_97%'] - beta_summary['mean']).values)
    ax = beta_summary.plot(x='index', y='mean', kind='bar', figsize=(14, 7),
                    title='Posterior Distribution of Beta Parameters',
                    yerr=err_vals, color='lightgrey',
                    legend=False, grid=True,
                    capsize=5)
    beta_summary.plot(x='index', y='mean', color='k', marker='o', linestyle='None',
                    ax=ax, grid=True, legend=False, xlim=plt.gca().get_xlim())

    plt.savefig(CURR_DIR+'/images/mortality_beta_summary.png')

    # # Appendix: Model Diagnostics
    # The following trace plots help to assess the convergence of the MCMC sampler.
    az.plot_trace(trace, compact=True)
    plt.savefig(CURR_DIR+'/images/mortality_MCMC-trace.png')
    return reported_mortality_rate, mortality_rate


# # About This Analysis
# 
# This analysis was done by [Joseph Richards](https://twitter.com/joeyrichar)
# 
# In this project[^3], we attempt to estimate the true mortality rate[^1] for COVID-19 while controlling for country-level covariates[^2][^4] such as:
# * age of outbreak in the country
# * transparency of the country's government
# * access to healthcare
# * demographics such as age of population and rural vs. urban
# 
# Estimating a mortality rate lower than the overall reported rate likely implies that there has been **significant under-testing and under-reporting of cases globally**.
# 
# ## Interpretation of Country-Level Parameters 
# 
# 1. days_since_first_case - positive (very statistically significant).  As time since outbreak increases, expected mortality rate **increases**, as expected.
# 2. cpi - negative (statistically significant).  As government transparency increases, expected mortality rate **decreases**.  This may mean that less transparent governments under-report cases, hence inflating the mortality rate.
# 3. healthcare avg. out-of-pocket spending - no significant trend.
# 4. hospital beds per capita - no significant trend.
# 5. Human Capital Index - no significant trend (slightly negative = mortality rates decrease with increased mobilization of the country)
# 6. percent over 65 - positive (statistically significant).  As population age increases, the mortality rate also **increases**, as expected.
# 7. percent rural - no significant trend.
# 
# 
# [^1]: As of March 10, the **overall reported mortality rate is 3.5%**.  However, this figure does not account for **systematic biases in case reporting and testing**.  The observed mortality of COVID-19 has varied widely from country to country (as of early March 2020).  For instance, as of March 10, mortality rates have ranged from < 0.1% in places like Germany (1100+ cases) to upwards of 5% in Italy (9000+ cases) and 3.9% in China (80k+ cases).
# 
# [^2]: The point of our modelling work here is to **try to understand and correct for the country-to-country differences that may cause the observed discrepancies in COVID-19 country-wide mortality rates**.  That way we can "undo" those biases and try to **pin down an overall *real* mortality rate**.
# 
# [^3]: Full details about the model are available at:  https://github.com/jwrichar/COVID19-mortality
# 
# [^4]: The affects of these parameters are subject to change as more data are collected.
#             







