from pathlib import Path
import os
import logging

def get(key, default):
    """
    Used for getting default configs for dev but forces environment only configs in production.
    """
    value = os.environ.get(key)
    if not value and RUN_MODE == PRODUCTION:
        raise Exception('env config "{0}" is missing'.format(key))
    return value or default

LOCAL = 'local'
PRODUCTION = 'production'
REMOTE_DEV = 'development'
REMOTE_TEST = 'test'
#default to production, when running locally set to LOCAL and run . local.env
RUN_MODE = os.environ.get('RUN_MODE', PRODUCTION)
print('Run mode: {0}'.format(RUN_MODE))

# default to be overriden
LOG_LEVEL = logging.ERROR
DEBUG_MODE = False

if RUN_MODE == LOCAL or RUN_MODE == REMOTE_DEV:
    LOG_LEVEL = logging.DEBUG
    DEBUG_MODE = True

elif RUN_MODE == REMOTE_TEST:
    LOG_LEVEL = logging.DEBUG
    DEBUG_MODE = True

elif RUN_MODE == PRODUCTION:
    LOG_LEVEL = logging.INFO
    DEBUG_MODE = False

#Paths
CURR_DIR = os.path.abspath(os.path.dirname(__file__))
BASE_DIR = Path(CURR_DIR).parent.parent.parent
print(f'BASE_DIR: {BASE_DIR}')

DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    
# raw data upload folder
RAW_UPLOAD_FOLDER = DATA_DIR + '/raw/'
PROCESSED_FOLDER = DATA_DIR + '/processed/'

# Database
DB_ENGINE='mongodb'
DB_DRIVER='motor'

# default to be overriden
MONGODB_SETTINGS=None
PYMONGO_SETTINGS=None

if RUN_MODE == PRODUCTION:
    # db local to server
    PYMONGO_SETTINGS = {
        'host': get('DB_PROD_HOST', '127.0.0.1'),
        'port': int(get('DB_PROD_PORT', '27017')),
        'username': get('DB_PROD_USERNAME', 'prod_user'),
        'password': get('DB_PROD_PASSWORD', 'prod_password'),
    }
    MONGODB_SETTINGS = {'db': get('DB_PROD_NAME', 'prod_db')}
elif RUN_MODE == REMOTE_DEV:
    PYMONGO_SETTINGS = {
        'host': get('DB_REMOTE_DEV_HOST', '127.0.0.1'),
        'port': int(get('DB_REMOTE_DEV_PORT', '27017')),
        'username': get('DB_REMOTE_DEV_USERNAME', 'remote_dev_user'),
        'password': get('DB_REMOTE_DEV_PASSWORD', 'remote_dev_password'),
    }
    MONGODB_SETTINGS = {'db': get('DB_REMOTE_DEV_NAME', 'remote_dev_db')}
elif RUN_MODE == REMOTE_TEST:
    PYMONGO_SETTINGS = {
        'host': get('DB_REMOTE_TEST_HOST', '127.0.0.1'),
        'port': int(get('DB_REMOTE_TEST_PORT', '27017')),
        'username': get('DB_REMOTE_TEST_USERNAME', 'remote_test_user'),
        'password': get('DB_REMOTE_TEST_PASSWORD', 'remote_test_password'),
    }
    MONGODB_SETTINGS = {'db': get('DB_REMOTE_TEST_NAME', 'remote_test_db')}
elif RUN_MODE == LOCAL:
    # local dev db
    PYMONGO_SETTINGS = {
        'host': get('DB_LOCAL_HOST', '127.0.0.1'),
        'port': int(get('DB_LOCAL_PORT', '27017')),
        'username': get('DB_LOCAL_USERNAME', 'local_user'),
        'password': get('DB_LOCAL_PASSWORD', 'local_password'),
    }
    MONGODB_SETTINGS = {'db': get('DB_LOCAL_NAME', 'local_db')}
else:
    raise ValueError(f'RUN_MODE: {RUN_MODE} not valid')

assert PYMONGO_SETTINGS is not None
assert MONGODB_SETTINGS is not None


API_VERSION = 'v1'
API_CORS_VALID = get('API_CORS_VALID', 'http://localhost:8080')

ASGI_HOST=get('ASGI_HOST', 'localhost')
ASGI_PORT=get('ASGI_PORT', 8000)

# logging
LOGGING_FILE = os.path.join(BASE_DIR, 'logs/app.log')
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s %(levelname)s %(name)s.%(funcName)s(): %(message)s '
                '(%(filename)s:%(lineno)d) PID:%(process)d ',
            'datefmt': "%Y-%m-%d %H:%M:%S",
        }
    },
    'handlers': {
        'console': {
            'level': LOG_LEVEL,
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'rotate_file': {
            'level': LOG_LEVEL,
            'formatter': 'standard',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': LOGGING_FILE,
            'encoding': 'utf8',
            #10MB
            'maxBytes': 10485760,
            'backupCount': 1,
        }
    },
    'loggers': {
        '': {
            'handlers': ['console', 'rotate_file'],
            'level': LOG_LEVEL,
        }
    }
}

