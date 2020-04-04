from motor.motor_asyncio import AsyncIOMotorClient
import logging
from config.server_config import PYMONGO_SETTINGS, MONGODB_SETTINGS

logger = logging.getLogger(__name__)


DB_CLIENT = AsyncIOMotorClient(**PYMONGO_SETTINGS)

DB = DB_CLIENT[
    MONGODB_SETTINGS.get("db")
]
logger.info(f'Using db: {MONGODB_SETTINGS.get("db")}')

def close_db_client():
    DB_CLIENT.close()

