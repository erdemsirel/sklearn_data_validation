import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance as scipy_wasserstein_distance
from sklearn.preprocessing import StandardScaler
from scipy.stats import kstest, normaltest, chisquare


# def abs_difference_of_means(dist_ref, dist):
#     return np.abs(dist_ref.mean - dist.mean)


# def relative_abs_difference_of_means(dist_ref, dist):
#     return np.abs(dist_ref.mean - dist.mean) / dist_ref.mean


def kstest_pvalue(var, ref_dist_cdf):
    # Implemented
    return kstest(var, ref_dist_cdf).pvalue


def psi(var, ref_dist, ref_dist_bins):
    # Implemented
    var_dist = pd.cut(var, bins=ref_dist_bins).value_counts(normalize=True)
    PSI = ((var_dist - ref_dist) * np.log(var_dist / ref_dist)).sum()
    return PSI


def chisquare_pvalue(var, ref_dist, ref_dist_bins):
    # Implemented
    var_dist = pd.cut(var, bins=ref_dist_bins).value_counts(normalize=True)

    return chisquare(var_dist, f_exp=ref_dist).pvalue


def is_var_normal(var, p_threshold=0.05):
    var_scaled = StandardScaler().fit_transform(np.array(var).reshape(-1, 1))
    return normaltest(var_scaled).pvalue > p_threshold

# def relative_wasserstein_varance(dist_ref, dist):
#     return scipy_wasserstein_distance(ref_dist, dist) / ref_dist.mean()

# def is_var_within_z_threshold(ref_dist, dist, z_threshold=3):
#     return abs_difference_of_means(ref_dist, dist) < (ref_dist.std() * z_threshold)
