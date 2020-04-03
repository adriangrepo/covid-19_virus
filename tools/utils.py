import yaml
import logging

logger = logging.getLogger(__name__)

def data_paths(yaml_file) -> dict:
    logger.debug(f'data_paths: {yaml_file}')
    yaml_data = {}
    try:
        with open(yaml_file) as yaml_file:
            yaml_data = yaml.load(yaml_file.read(), Loader=yaml.SafeLoader)
    except Exception as e:
        print(e)
    return yaml_data

def load_config() -> dict:
    conf = {}
    try:
        with open('services/server/config/config.yml') as yaml_file:
            conf = yaml.load(yaml_file.read(), Loader=yaml.SafeLoader)
    except FileNotFoundError:
        with open('services/server/config/config.ci.yml') as yaml_file:
            conf = yaml.load(yaml_file.read(), Loader=yaml.SafeLoader)
    return conf