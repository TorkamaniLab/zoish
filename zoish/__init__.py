__version__ = "1.63.0"


from dotenv import load_dotenv
import logging
import logging.config
import os
import yaml
path = "zoish/config.yaml"
DEFAULT_LEVEL = logging.INFO


def log_setup(log_cfg_path=path):
    with open(path, "r") as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    # set up logging configuration


log_setup()
# create logger

load_dotenv()
env = os.getenv("env")
logger = logging.getLogger(env)
