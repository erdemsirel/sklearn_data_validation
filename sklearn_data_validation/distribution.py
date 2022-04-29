import pandas as pd
import numpy as np
import metrics
from scipy.stats import kstest, normaltest, chisquare
from sklearn.preprocessing import StandardScaler

class Distribution:
    def __init__(self, var: pd.Series, bins: int=100):
        self.bins = bins
        self.pdf_ = var.value_counts(bins=self.bins, normalize=True).sort_index()
        self.cdf_ = self.pdf_.cumsum()
        
        self.dist_q, self.dist_q_bins = pd.qcut(var, self.bins, retbins=True)
        self.dist_q = self.dist_q.value_counts(normalize=True)
        
        self.stats =  var.describe(percentiles=np.arange(0 ,1, 1/self.bins)[1:], include='all').to_dict()
        for stat_key, stat_value in self.stats.items():
            setattr(self, stat_key, stat_value)
    
    @staticmethod
    def get_df_value(x, df_):
        # Get corresponding value from cumulative or probability distributions.
        if x > df_.index.max().right:
            x = df_.index.max().right - 0.00001
        elif x < df_.index.min().left:
            x = df_.index.min().left + 0.00001
        return df_.iloc[df_.index.get_loc(x)]
    
    def cdf(self, x):
        if hasattr(x, '__iter__'):
            x = [i for i in x]
            return list(map(lambda x_: self.get_df_value(x_, self.cdf_), x))
        else:
            return self.get_df_value(x, self.cdf_)
        
    def pdf(self, x):
        if hasattr(x, '__iter__'):
            x = [i for i in x]
            return list(map(lambda x_: self.get_df_value(x_, self.pdf_), x))
        else:
            return self.get_df_value(x, self.cdf_)
        
    def __repr__(self):
        return str(vars(self).keys())

    def kstest(self, var):
        return metrics.is_ks_successfull(var, dist_ref=self)