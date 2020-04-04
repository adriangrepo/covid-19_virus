
import io
from bson.objectid import ObjectId
from datetime import datetime
import pymongo
from pymongo import MongoClient
import yaml
import requests
import pandas as pd
import json
import logging
from services.server.utils import setup_logging
from pandas.core.common import maybe_box_datetimelike
from data_urls import csse_ts_local, csse_ts_global

logger = logging.getLogger(__name__)

INITIAL_LOAD=True
USE_LOCAL_DATA=True

setup_logging('data_loader')

client = MongoClient(PYMONGO_SETTINGS.get('host'), PYMONGO_SETTINGS.get('port'))
db = client[MONGODB_SETTINGS.get('db')]

def download_data(url):
    s=requests.get(url).content
    csv=io.StringIO(s.decode('utf-8'))
    return csv

def load_data(csv_data, collection_name, drop_prev=False):
    logger.debug(f'>>load_data: {csv_data}, collection: {collection_name}')
    date_now = datetime.utcnow().isoformat()
    collection = db[collection_name]
    if drop_prev:
        collection.drop() 
    #skip offending lines
    df = pd.read_csv(csv_data, error_bad_lines=False)
    data_json = json.loads(df.to_json(orient='records'))
    data_list = [dict((k, maybe_box_datetimelike(v)) for k, v in zip(df.columns, row) if v != None and v == v) for row in df.values]
    for l in data_list:
        l.update({'insert_date': date_now})
    try:
        result=collection.insert_many(data_list)
    except Exception as e:
        logger.debug(e)

def load_ts_total_infected_estimate_data(csv_data, collection_name, drop_prev=False):
    date_now = datetime.utcnow().isoformat()
    collection = db[collection_name]
    if drop_prev:
        collection.drop() 
    #skip offending lines
    df = pd.read_csv(csv_data, error_bad_lines=False)
    data_json = json.loads(df.to_json(orient='records'))
    data_list = [dict((k, maybe_box_datetimelike(v)) for k, v in zip(df.columns, row) if v != None and v == v) for row in df.values]
    for l in data_list:
        l.update({'insert_date': date_now})
    try:
        result=collection.insert_many(data_list)
    except Exception as e:
        logger.debug(e)

csse_data = data_paths('tools/csse_data_paths.yml')


for csse_case_type in ['confirmed','deaths','recovered']:
    if USE_LOCAL_DATA:
        url=csse_ts_local.get(csse_case_type, {})
    else:
        url=csse_ts_global.get(csse_case_type, {})
    load_data(url, 'csse_'+csse_case_type, drop_prev=True)

#estimated infected over time per country
data_csv='data/processed/csse/df_all_data_countries_pc.csv'
load_data(data_csv, 'csse_estimated_infected_ts', drop_prev=True)

#estimated infected latest per country plus un and fh stats
data_csv='data/processed/csse/df_data_countries_pc_latest.csv'
load_data(data_csv, 'csse_estimated_infected_latest_stats', drop_prev=True)



