import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance as scipy_wasserstein_distance
from sklearn.preprocessing import StandardScaler
from scipy.stats import kstest, normaltest, chisquare
from distribution import Distribution

def abs_difference_of_means(dist_ref: Distribution, dist: Distribution):
    return np.abs(dist_ref.mean - dist.mean)

def relative_abs_difference_of_means(dist_ref: Distribution, dist: Distribution):
    return np.abs(dist_ref.mean - dist.mean) / dist_ref.mean

def is_ks_successfull(var, dist_ref: Distribution, p_threshold=0.05):
    return kstest(var, dist_ref.cdf).pvalue > p_threshold

def psi(var, dist_ref: Distribution, bins=10):
    ref_dist, ref_dist_bins = pd.cut(ref_dist, 10, retbins=True)
    ref_dist = ref_dist.value_counts(normalize=True)
    
    var_dist = pd.cut(dist, bins=ref_dist_bins).value_counts(normalize=True)
    PSI=((var_dist - ref_dist) * np.log(var_dist / ref_dist)).sum()
    return PSI

def is_chisquare_successfull(ref_dist, dist, p_threshold=0.05):
    ref_dist, ref_dist_bins = pd.qcut(ref_dist, 4, retbins=True)
    ref_dist = ref_dist.value_counts(normalize=True)

    var_dist = pd.cut(dist, bins=ref_dist_bins).value_counts(normalize=True)
    
    return chisquare(var_dist, f_exp=ref_dist).pvalue > p_threshold

def cdf(x, dist, bins=10):
    cdf_ = dist.value_counts(bins=bins, normalize=True).sort_index().cumsum()
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

def pdf(x, dist, bins=10):
    pdf_ = dist.value_counts(bins=bins, normalize=True).sort_index()
    return pdf_.iloc[pdf_.index.get_loc(x)]


# def is_dist_normal(dist: Distribution, p_threshold=0.05):
#     dist = StandardScaler().fit_transform(np.array(dist).reshape(-1, 1))
#     return normaltest(dist).pvalue > p_threshold

# def relative_wasserstein_varance(dist_ref: Distribution, dist: Distribution):
#     return scipy_wasserstein_distance(ref_dist, dist) / ref_dist.mean()

# def is_var_within_z_threshold(ref_dist, dist, z_threshold=3):
#     return abs_difference_of_means(ref_dist, dist) < (ref_dist.std() * z_threshold)
