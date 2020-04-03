
import io
from bson.objectid import ObjectId
from datetime import datetime
import pymongo
from pymongo import MongoClient
from pymongo import MongoClient, errors
import time
import yaml
import logging
from services.server.utils import setup_logging

logger = logging.getLogger(__name__)

def load_config() -> dict:
    conf = {}
    try:
        with open('services/server/config/config.yml') as yaml_file:
            conf = yaml.load(yaml_file.read(), Loader=yaml.SafeLoader)
    except FileNotFoundError:
        with open('services/server/config/config.ci.yml') as yaml_file:
            conf = yaml.load(yaml_file.read(), Loader=yaml.SafeLoader)
    return conf

CONF = load_config()

setup_logging('data_loader', log_level=CONF['log_level'])
logger.debug(CONF)
PORT= int(CONF.get("databases", {}).get("test", {}).get("PORT"))

start = time.time()
try:
    client = MongoClient('localhost', PORT)
    client.server_info() 
except Exception as e:
    print (f"connection error: {e}")

print (f'connection time: {time.time() - start}')

db_name=f'{CONF.get("databases", {}).get("test", {}).get("NAME")}'
print(f'db_name: {db_name}')
db = client[db_name]
col = db["connection_test"]

try:
    mydict = { "name": "Tester", "address": "Planet Earth", "time": str(time.time()) }
    result= col.insert_one(mydict)
    print(result)
    one_doc= col.find_one()
    print ("find_one():", one_doc)

except errors.ServerSelectionTimeoutError as err:
    print ("find_one() ERROR:", err)