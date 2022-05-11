import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def to_markdown(metrics: pd.DataFrame):
    # metrics: Output of FeatureMatrix.calculate_metrics method
    metrics_ = metrics.copy()
    if len(metrics)==0:
        return ""
    
    metric_columns = [col for col in metrics.columns
                   if not col.endswith("_result") and col+"_result" in metrics.columns]
    for col in metric_columns:
        metrics_[col] = metrics_[
            col+"_result"].replace({True: "(OK)", False: "(X)"}) + " " + metrics_[col].round(3).astype(str)
    metrics_["success"] = metrics_["success"].replace({True: "(OK)", False: "(X)"})
    return metrics_[metric_columns + ["success"]].to_markdown()


def log_individual_metrics(metrics: pd.DataFrame, log_successfull_metrics=False):
    # metrics: Output of FeatureMatrix.calculate_metrics method
    metric_cols = [col for col in metrics.columns
                   if not col.endswith("_result") and col+"_result" in metrics.columns]

    for feature_name, feature_metrics in metrics[~metrics['success']][metric_cols].iterrows():
        for feature_metric_name, feature_metric_value in feature_metrics.iteritems():
            if not metrics.loc[feature_name, f"{feature_metric_name}_result"]:
                logger.warning(f"For feature '{feature_name}', the metric '{feature_metric_name}': {np.round(feature_metric_value,3)} is beyond limits.",
                               extra={"feature_name": feature_name,
                                      "feature_metric_name": feature_metric_name,
                                      "feature_metric_value": feature_metric_value})
            elif log_successfull_metrics:
                logger.info(f"For feature '{feature_name}', the metric '{feature_metric_name}': {np.round(feature_metric_value,3)} is within limits.",
                            extra={"feature_name": feature_name,
                                   "feature_metric_name": feature_metric_name,
                                   "feature_metric_value": feature_metric_value})


def log_metrics_summary(metrics: pd.DataFrame, log_successfull_metrics=False):
    # metrics: Output of FeatureMatrix.calculate_metrics method
    if log_successfull_metrics:
        metrics_ = metrics
    else:
        metrics_ = metrics[~metrics["success"]]

    number_of_unsuccessful_feature = (~metrics[[col for col in metrics.columns if col.endswith("_result")]]).sum(axis=1).astype(bool).sum()
    if number_of_unsuccessful_feature>0:
        msg= f"The number of unsuccessful feature: {number_of_unsuccessful_feature}."
        logger.warning(f"{msg}\n{to_markdown(metrics_)}")
    elif log_successfull_metrics:
        msg= f"All features are as expected."
        logger.info(f"{msg}\n{to_markdown(metrics_)}")
