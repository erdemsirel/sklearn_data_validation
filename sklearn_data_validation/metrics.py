import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance as scipy_wasserstein_distance
from sklearn.preprocessing import StandardScaler
from scipy.stats import kstest, normaltest, chisquare

def abs_difference_of_means(ref_var, var):
    return np.abs(np.mean(ref_var) - np.mean(var))

def relative_abs_difference_of_means(ref_var, var):
    return np.abs(np.mean(ref_var) - np.mean(var)) / np.mean(ref_var)


def relative_wasserstein_varance(ref_var, var):
    return scipy_wasserstein_distance(ref_var, var) / ref_var.mean()

def is_var_within_z_threshold(ref_var, var, z_threshold=3):
    return abs_difference_of_means(ref_var, var) < (ref_var.std() * z_threshold)

def is_ks_successfull(ref_var, var, p_threshold=0.05):
    return kstest(ref_var, var).pvalue > p_threshold

def is_vars_normal(ref_var, var, p_threshold=0.05):
    ref_var = StandardScaler().fit_transform(np.array(ref_var).reshape(-1, 1))
    var = StandardScaler().fit_transform(np.array(var).reshape(-1, 1))
    
    if normaltest(ref_var).pvalue < p_threshold:
        return None
    elif normaltest(ref_var).pvalue > p_threshold and normaltest(var).pvalue > p_threshold:
        return True
    else:
        return False

def psi(ref_var, var, bins=10):
    ref_dist, ref_dist_bins = pd.cut(ref_var, 10, retbins=True)
    ref_dist = ref_dist.value_counts(normalize=True)
    
    var_dist = pd.cut(var, bins=ref_dist_bins).value_counts(normalize=True)
    PSI=((var_dist - ref_dist) * np.log(var_dist / ref_dist)).sum()
    return PSI

def is_chisquare_successfull(ref_var, var, p_threshold=0.05):
    ref_dist, ref_dist_bins = pd.qcut(ref_var, 4, retbins=True)
    ref_dist = ref_dist.value_counts(normalize=True)

    var_dist = pd.cut(var, bins=ref_dist_bins).value_counts(normalize=True)
    
    return chisquare(var_dist, f_exp=ref_dist).pvalue > p_threshold

def cdf(x, var, bins=10):
    cdf_ = var.value_counts(bins=bins, normalize=True).sort_index().cumsum()
    def get_cdf(x):
        if x > cdf_.index.max().right:
            x = cdf_.index.max().right - 0.00001
        elif x < cdf_.index.min().left:
            x = cdf_.index.min().left + 0.00001
        return cdf_.iloc[cdf_.index.get_loc(x)]
    if hasattr(x, '__iter__'):
        x = [i for i in x]
        return list(map(lambda x_: get_cdf(x_), x))
    else:
        return get_cdf(x)

def pdf(x, var, bins=10):
    pdf_ = var.value_counts(bins=bins, normalize=True).sort_index()
    return pdf_.iloc[pdf_.index.get_loc(x)]