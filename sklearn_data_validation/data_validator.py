import pickle
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from feature_matrix import FeatureMatrix
from utils import to_markdown
import metrics
import copy
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DataValidator(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.feature_matrix = None

    def fit(self, X, y=None):
        self.feature_matrix = FeatureMatrix(X=X, **self.kwargs)
        self.is_fitted = True
        return self

    def transform(self, X, y=None):
        score_metrics = self.feature_matrix.calculate_metrics(X)
        if self.kwargs.get("only_unsuccessfull", False):
            score_metrics = score_metrics[~score_metrics["success"]]
        
        if self.kwargs.get("log_metrics", True):
            logger.info("Feature Matrix metrics\n" + to_markdown(score_metrics))
        return score_metrics

    def to_pickle(self, path):
        if self.feature_matrix is None:
            raise Exception("Only fitted objects could be pickled.")
        with open(path, "wb") as f:
            pickle.dump(copy.copy(self), f)
            logger.info(f"The DataValidator instance has been saved as pickle into {path} path.")

    @staticmethod
    def read_pickle(path):
        with open(path, "rb") as f:
            loaded_instance = pickle.load(f)
            logger.info(f"The DataValidator instance has been loaded from {path} path.")
            return loaded_instance
