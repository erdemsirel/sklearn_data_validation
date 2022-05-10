import pandas as pd


def to_markdown(metrics: pd.DataFrame):
    metrics_ = metrics.copy()
    metric_columns = [col for col in metrics_.columns if not col.endswith("_result")]
    for col in metric_columns:
        if col+"_result" not in metrics_.columns: continue
        metrics_[col] = metrics_[col+"_result"].replace({True: "✓", False:"X"}) + " " +  metrics_[col].round(3).astype(str)
    metrics_["success"] = metrics_["success"].replace({True: "✓", False:"X"})
    return metrics_[metric_columns].to_markdown()