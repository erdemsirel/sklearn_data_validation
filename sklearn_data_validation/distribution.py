import pandas as pd
import numpy as np
import metrics
from scipy.stats import kstest, normaltest, chisquare
from sklearn.preprocessing import StandardScaler
import logging
import time

logger = logging.getLogger(__name__)


class Distribution:
    def __init__(self, var: pd.Series, bins: int = 100, p_threshold=0.05):
        self.bins = bins
        self.pdf_ = var.value_counts(bins=self.bins, normalize=True).sort_index()
        self.cdf_ = self.pdf_.cumsum()

        self.dist_q, self.dist_q_bins = pd.qcut(var, self.bins, retbins=True)
        self.dist_q = self.dist_q.value_counts(normalize=True).sort_index()

        self.dist, self.dist_bins = pd.cut(var, self.bins, retbins=True)
        self.dist = self.dist.value_counts(normalize=True).sort_index()

        self.stats = var.describe(percentiles=np.arange(0, 1, 1 / self.bins)[1:], include="all").to_dict()
        for stat_key, stat_value in self.stats.items():
            setattr(self, stat_key, stat_value)

        self.is_var_normal = metrics.is_var_normal(var, p_threshold=p_threshold)

    @staticmethod
    def get_distribution_function_value(x, df_):
        # Get corresponding value from cumulative or probability distributions.
        if x > df_.index.max().right:
            x = df_.index.max().right - 0.00001
        elif x < df_.index.min().left:
            x = df_.index.min().left + 0.00001
        return df_.iloc[df_.index.get_loc(x)]

    def cdf(self, x):
        if hasattr(x, "__iter__"):
            x = [i for i in x]
            return list(map(lambda x_: self.get_distribution_function_value(x_, self.cdf_), x))
        else:
            return self.get_distribution_function_value(x, self.cdf_)

    def pdf(self, x):
        if hasattr(x, "__iter__"):
            x = [i for i in x]
            return list(map(lambda x_: self.get_distribution_function_value(x_, self.pdf_), x))
        else:
            return self.get_distribution_function_value(x, self.cdf_)

    def __repr__(self):
        return str(vars(self).keys())

    def calculate_metric(self, var, metric):
        start = time.time()
        result = metric(var=var, ref_dist=self)
        logger.info(f"{metric.__name__} calculated in {round(time.time() - start)} seconds.")
        return result