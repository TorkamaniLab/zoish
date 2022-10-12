from abc import ABCMeta, abstractmethod


class BestEstimatorFactory(metaclass=ABCMeta):
    @abstractmethod
    def get_best_estimator(*args, **kargs):
        """
        Return the best estimator using an optimization
        engine, e.g., Optuna, Grid, or Random search.
        ...
        Attributes
        ----------
        *args: list
            A list of possible argumnets
        **kwargs: dict
            A dict of possible argumnets
        """
        pass

