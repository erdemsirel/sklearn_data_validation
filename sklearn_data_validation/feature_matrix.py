from os import rename
from typing import Dict
import pandas as pd
import numpy as np
import metrics
from distribution import Distribution
import matplotlib.pyplot as plt
import logging
import math

logger = logging.getLogger(__name__)

class FeatureMatrix:
    def __init__(self, X: pd.DataFrame, bins: int = 100, p_threshold=0.05, **kwargs):

        # Determine numerical & non-numerical colums
        self.numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        self.numerical_binary_columns = [col for col in self.numerical_columns if X[col].nunique() == 2]

        for numerical_binary_column in self.numerical_binary_columns:
            self.numerical_columns.remove(numerical_binary_column)

        self.categorical_columns = X.select_dtypes(include=["category", "bool_", "object_", "object"]).columns.tolist()
        self.categorical_columns.extend(self.numerical_binary_columns)

        assert len(X.columns) == len(self.categorical_columns) + len(self.numerical_columns)

        self.distributions = {}
        for col in self.numerical_columns:
            try:
                self.distributions[col] = Distribution(var=X[col], bins=bins, p_threshold=p_threshold)
            except Exception as e:
                logger.warn(f"Cannot create Distribution object for {col}.")

    def calculate_metrics(self, X_):
        feat_metrics = {}
        for col in X_.columns:
            col_dist = self.distributions.get(col)
            if col_dist is None:
                continue
            logger.debug(f"Calculating metrics for {col}")
            feat_metrics[col] = {
                metric.__name__: col_dist.calculate_metric(X_[col], metric) for metric in metrics.DISTRIBUTION_METRICS
            }

        result = pd.DataFrame(feat_metrics).T
        result = metrics.determine_metric_results(result)

        result_cols = [col for col in result.columns if col.endswith("_result")]
        result["success"] = result[result_cols].all(axis=1)
        result["failed_test_count"] = (~result[result_cols]).sum(axis=1)
        result = result.sort_values("failed_test_count", ascending=False)
        result = result.reindex(sorted(result.columns), axis=1)
        result.columns = result.columns.str.replace("dist_", "")
        return result

    def plot_attr(self, attr, ncols=2):
        print("TEST", math.ceil(len(self.distributions)))
        nrows = int(math.ceil(len(self.distributions)) / ncols)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, nrows * 4))
        axes = iter(axs.flat)
        for feature_name, distribution in self.distributions.items():
            getattr(distribution, attr).plot(ax=next(axes), title=feature_name)
        fig.tight_layout()
        plt.show()

    def __repr__(self) -> str:
        return pd.DataFrame(
            {
                feature_name: {"count": distribution.count, "mean": distribution.mean}
                for feature_name, distribution in self.distributions.items()
            },
        ).T.to_markdown()
