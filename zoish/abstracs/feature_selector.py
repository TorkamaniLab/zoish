from sklearn.base import BaseEstimator, TransformerMixin
from abc import ABCMeta, abstractmethod


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Base class for creating feature selector. """

    def __init__(self, *args, **kwargs):

        """
        Class initalizer
        """
        pass

    def prepare_data(self):
        pass

    def fit(self, *args, **kwargs):
        """
        Fit estimator using params
        """
        pass
    def calc_best_estimator(self, *args, **kwargs):
        """
        Calculate best estimator using params
        """
        pass

    def transform(self, *args, **kwargs):
        """
        Return a transform
        """
        pass
    def predict(self, *args, **kwargs):
        """
        Predict 
        """
        pass
    

class BestEstimatorSetter(metaclass=ABCMeta):
    """Base class for creating feature selector. """

    def __init__(self, *args, **kwargs):

        """
        Class initalizer
        """
        pass

    def get_best_estomator(self, *args, **kwargs):
        """
        Get best estomator 
        """
        pass

    def get_feature_selector(self, *args, **kwargs):
        """
        Get best estomator 
        """
        pass

    def set_best_estomator_to_feature_selector(self, *args, **kwargs):
        """
        Asign best estomator to feature selector 
        """
        pass
