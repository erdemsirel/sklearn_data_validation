import pickle
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from feature_matrix import FeatureMatrix
import utils
import metrics
import copy
import logging

logger = logging.getLogger(__name__)


class DataValidator(BaseEstimator, TransformerMixin):
    def __init__(self, log_successfull_metrics=False,
                 log_metrics_summary=True,
                 log_individual_metrics=True,
                 **kwargs):
        self.feature_matrix = None
        self.log_successfull_metrics = log_successfull_metrics
        self.log_metrics_summary = log_metrics_summary
        self.log_individual_metrics = log_individual_metrics
        self.kwargs = kwargs

    def fit(self, X, y=None):
        self.feature_matrix = FeatureMatrix(X=X, **self.kwargs)
        self.is_fitted = True
        return self

    def transform(self, X, y=None):
        metrics = self.feature_matrix.calculate_metrics(X)

        if self.log_metrics_summary:
            utils.log_metrics_summary(metrics=metrics,
                                      log_successfull_metrics=self.log_successfull_metrics)

        if self.log_individual_metrics:
            utils.log_individual_metrics(metrics=metrics,
                                         log_successfull_metrics=self.log_successfull_metrics)

        return X

    def to_pickle(self, path):
        if self.feature_matrix is None:
            raise Exception("Only fitted objects could be pickled.")
        with open(path, "wb") as f:
            pickle.dump(copy.copy(self), f)
            logger.debug(
                f"The DataValidator instance has been saved as pickle into {path} path.")

    @staticmethod
    def read_pickle(path):
        with open(path, "rb") as f:
            loaded_instance = pickle.load(f)
            logger.debug(
                f"The DataValidator instance has been loaded from {path} path.")
            return loaded_instance
