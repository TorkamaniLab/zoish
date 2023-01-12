__version__ = "2.1.0"


import logging
import logging.config
import os

import yaml
from dotenv import load_dotenv

path = "zoish/config.yaml"
DEFAULT_LEVEL = logging.INFO


def log_setup(log_cfg_path=path):
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)
            # set up logging configuration
            return True
    except Exception as e:
        print(
            f"In this module, the default logging will be applied. The error is {e} which will be skipped!"
        )
        return False


if log_setup():
    # create logger

    load_dotenv()
    env = os.getenv("env")
    logger = logging.getLogger(env)
else:
    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger()
