
import logging
from config.server_config import LOGGING_CONFIG, LOG_LEVEL, LOGGING_FILE

def setup_rotating_logging(log_file=LOGGING_FILE, log_level=LOG_LEVEL):
    confg = LOGGING_CONFIG
    confg['handlers']['rotate_file']['filename'] = log_file
    confg['loggers']['level'] = log_level
    print(f'<< log_level: {log_level}, confg: {confg}, type: {type(confg)}')
    logging.config.dictConfig(confg)

def setup_logging(log_file=LOGGING_FILE,log_level=LOG_LEVEL):
    format='%(asctime)s %(levelname)s %(name)s.%(funcName)s(): %(message)s (%(filename)s:%(lineno)d) PID:%(process)d '
    handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
    logging.basicConfig(level = log_level, format = format, handlers = handlers)