
import logging

def setup_logging(log_file_name, log_level):
    if log_level == 'DEBUG':
        level=logging.DEBUG
    elif log_level == 'INFO':
        level=logging.INFO
    elif log_level == 'WARNING':
        level=logging.WARNING
    elif log_level == 'ERROR':
        level=logging.ERROR
    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.FileHandler(f'../../logs/{log_file_name}.log'), logging.StreamHandler()]
    logging.basicConfig(level = level, format = format, handlers = handlers)