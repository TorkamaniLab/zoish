from zoish.abstracs.feature_selector_abstracts import FeatureSelector, PlotFeatures
from zoish.base_classes.best_estimator_getters import (
    BestEstimatorFindByGridSearch,
    BestEstimatorFindByOptuna,
    BestEstimatorFindByRandomSearch,
)
import fasttreeshap
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from feature_engine.selection import SelectBySingleFeaturePerformance
# from dotenv import load_dotenv
# import logging
# import logging.config
# import os
# import yaml
# path = "zoish/config.yaml"
# DEFAULT_LEVEL = logging.INFO


# def log_setup(log_cfg_path=path):
#     with open(path, "r") as f:
#         config = yaml.safe_load(f.read())
#         logging.config.dictConfig(config)
#     # set up logging configuration


# log_setup()
# # create logger

# load_dotenv()
# env = os.getenv("env")
# print(env)
# logger = logging.getLogger(env)
# logger.info("Zosih feature selector has started !")

