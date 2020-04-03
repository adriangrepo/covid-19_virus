# backend/covid/config/config.yml

from motor.motor_asyncio import AsyncIOMotorClient
import yaml
import logging

logger = logging.getLogger(__name__)

def load_config() -> dict:
    conf = {}
    try:
        with open('config/config.yml') as yaml_file:
            conf = yaml.load(yaml_file.read(), Loader=yaml.SafeLoader)
    except FileNotFoundError as fe:
        logger.error(fe)
        raise RuntimeError()
    return conf


CONF = load_config()
API_VERSION = CONF.get("api", {}).get("versions", {}).get("current", 'v1')

def _get_client_config():
    conf = CONF
    if conf.get("databases", {}).get("default", {}).get("USER", False) and\
            conf.get("databases", {}).get("default", {}).get(
                "PASSWORD", False):
        client_conf = {
            "host": conf.get(
                "databases", {}).get("default", {}).get("HOST"),
            "port": int(conf.get(
                "databases", {}).get("default", {}).get("PORT")),
            "username": conf.get(
                "databases", {}).get("default", {}).get("USER"),
            "password": conf.get(
                "databases", {}).get("default", {}).get("PASSWORD"),
        }
    else:
        client_conf = {
            "host": conf.get(
                "databases", {}).get("default", {}).get("HOST"),
            "port": int(conf.get(
                "databases", {}).get("default", {}).get("PORT")),
        }
    return client_conf


DB_CLIENT = AsyncIOMotorClient(**_get_client_config())

DB = DB_CLIENT[
    CONF.get("databases", {}).get("default", {}).get("NAME")
]


def close_db_client():
    DB_CLIENT.close()
