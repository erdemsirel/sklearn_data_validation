from typing import Dict
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance as scipy_wasserstein_distance
from sklearn.preprocessing import StandardScaler
from scipy.stats import kstest, normaltest, chisquare


# def abs_difference_of_means(dist_ref, dist):
#     return np.abs(dist_ref.mean - dist.mean)




def relative_difference_of_means(dist_mean, dist_ref_mean):
    return (dist_ref_mean - dist_mean) / dist_ref_mean


def is_var_normal(var, p_threshold):
    var_scaled = StandardScaler().fit_transform(np.array(var).reshape(-1, 1))
    return int(normaltest(var_scaled).pvalue[0] > p_threshold)


def _kstest_pvalue(var, ref_dist_cdf):
    # Implemented
    return kstest(var, ref_dist_cdf).pvalue


def dist_kstest_pvalue(var, ref_dist):
    # Implemented
    return _kstest_pvalue(var, ref_dist.cdf)


def _chisquare_pvalue(var, ref_dist_dist_q, ref_dist_dist_q_bins):
    # Implemented
    var_dist = pd.cut(var, bins=ref_dist_dist_q_bins).value_counts(normalize=True)
    return chisquare(var_dist, f_exp=ref_dist_dist_q).pvalue


def dist_chisquare_pvalue(var, ref_dist):
    # Implemented
    return _chisquare_pvalue(var, ref_dist.dist_q, ref_dist.dist_q_bins)


def _psi(var, ref_dist_dist_q, ref_dist_dist_q_bins):
    # Implemented
    var_dist = pd.cut(var, bins=ref_dist_dist_q_bins).value_counts(normalize=True)
    PSI = ((var_dist - ref_dist_dist_q) * np.log(var_dist / ref_dist_dist_q)).sum()
    if np.isinf(PSI):
        print(var_dist)
        print(ref_dist_dist_q)
    return PSI


def dist_psi(var, ref_dist):
    return _psi(var, ref_dist.dist_q, ref_dist.dist_q_bins)


def _missing_value_ratio_relative_diff(var_missing_value_ratio, ref_dist_missing_value_ratio):
    return np.abs(np.round((var_missing_value_ratio / (ref_dist_missing_value_ratio+0.0000001)), 3))
    
    
def dist_missing_val_rat_rel_diff(var, ref_dist):
    return _missing_value_ratio_relative_diff(var_missing_value_ratio=var.isna().mean(), 
                                              ref_dist_missing_value_ratio=ref_dist.missing_value_ratio)

def relative_abs_difference_of_means(dist_mean, dist_ref_mean):
    return np.abs(dist_ref_mean - dist_mean) / dist_ref_mean

def dist_rel_abs_diff_of_means(var, ref_dist):
    return relative_abs_difference_of_means(dist_mean=np.mean(var), dist_ref_mean=ref_dist.mean)


def z_score(dist_mean, dist_ref_mean, dist_ref_std):
    return (dist_ref_mean - dist_mean) / dist_ref_std


def z_score_abs(dist_mean, dist_ref_mean, dist_ref_std):
    return np.abs(dist_ref_mean - dist_mean) / dist_ref_std

def dist_z_score_abs(var, ref_dist):
    return z_score_abs(dist_mean=np.mean(var), dist_ref_mean=ref_dist.mean, dist_ref_std=ref_dist.std)

def dist_normal(var, ref_dist):
    if ref_dist.is_normal:
        return is_var_normal(var)
    else:
        return np.nan

DISTRIBUTION_METRICS = [
    dist_chisquare_pvalue,
    dist_kstest_pvalue,
    dist_psi,
    dist_rel_abs_diff_of_means,
    dist_z_score_abs,
    dist_missing_val_rat_rel_diff,
    dist_normal,
]

DEFAULT_METRIC_THRESHOLDS = {
    "dist_chisquare_pvalue": {"threshold": 0.05, "type_expected": "g"},
    "dist_kstest_pvalue": {"threshold": 0.05, "type_expected": "l"},
    "dist_psi": {"threshold": 0.2, "type_expected": "l"},
    "dist_rel_abs_diff_of_means": {"threshold": 1.1, "type_expected": "l"},
    "dist_z_score_abs": {"threshold": 1, "type_expected": "l"},
    "dist_missing_val_rat_rel_diff": {"threshold": 0.2, "type_expected": "l"},
    "dist_normal": {"threshold": [1, np.nan], "type_expected": "in"},
}


def determine_metric_results(metrics: pd.DataFrame, 
                             thresholds: Dict = DEFAULT_METRIC_THRESHOLDS):
    def comparison_(metric_col: pd.Series, threshold, type_expected):
        if type_expected == "g":
            result = metric_col > threshold
        elif type_expected == "ge":
            result = metric_col >= threshold
        elif type_expected == "l":
            result = metric_col < threshold
        elif type_expected == "le":
            result = metric_col <= threshold
        elif type_expected == "e":
            result = metric_col == threshold
        elif type_expected == "in":
            result = metric_col.isin(threshold)
        else:
            raise ValueError("type_expected should be one of the following options [g, ge, l, le, e]")

        return result

    metrics_results = metrics.loc[:, [col for col in metrics.columns if col.startswith("dist_")]].apply(
        lambda col: comparison_(
            metric_col=col,
            threshold=thresholds[col.name]["threshold"],
            type_expected=thresholds[col.name]["type_expected"],
        ),
        axis=0,
    )

    return metrics.join(metrics_results, rsuffix="_result")


# def relative_wasserstein_varance(dist_ref, dist):
#     return scipy_wasserstein_distance(ref_dist, dist) / ref_dist.mean()

# def is_var_within_z_threshold(ref_dist, dist, z_threshold=3):
#     return abs_difference_of_means(ref_dist, dist) < (ref_dist.std() * z_threshold)
