
import logging
from config.server_config import LOG_LEVEL, LOGGING_CONFIG

def setup_logging(log_file=LOGGING_FILE, log_level=LOG_LEVEL):
    config = LOGGING_CONFIG
    config['handlers']['rotate_file']['filename'] = log_file
    config['loggers']['level'] = log_level
    logging.config.dictConfig(config)