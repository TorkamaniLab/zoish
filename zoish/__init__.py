__version__ = "4.1.0"

import logging
import logging.config
import os

import yaml
from dotenv import load_dotenv

from .project_conf import ROOT_PROJECT

PATH_TO_LOG_CONF = ROOT_PROJECT / "zoish" / "config.yaml"
# DEFAULT_LEVEL in production env
DEFAULT_LEVEL = logging.ERROR


def log_setup(log_cfg_path=PATH_TO_LOG_CONF):
    try:
        with open(PATH_TO_LOG_CONF, "r") as f:
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
