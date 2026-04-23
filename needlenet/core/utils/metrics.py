from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall
)

def get_classification_metrics(cfg, num_classes: int, prefix: str) -> MetricCollection:

    metrics = cfg['metrics']['list']

    produces_metrics = {}

    for metric in metrics:
        metric_name = str(metric.get("name")).lower()
        metric_average = metric.get("average", None)
        metric_average = str(metric_average).lower() if metric_average is not None else None
        if metric_average is not None:
            if metric_average == "none":
                metric_average = None

        if metric_name == "accuracy":
            produces_metrics["accuracy"] = MulticlassAccuracy(num_classes=num_classes)
        elif metric_name == "f1":
            if "average" not in metric:
                raise ValueError("F1 metric requires 'average' parameter (e.g. 'macro' or 'weighted' or 'none')")
            produces_metrics[f'f1_{metric_average if metric_average is not None else ""}'] = MulticlassF1Score(num_classes=num_classes, average=metric_average)
        elif metric_name == "precision":
            if "average" not in metric:
                raise ValueError("Precision metric requires 'average' parameter (e.g. 'macro' or 'weighted' or 'none')")
            produces_metrics[f'precision_{metric_average if metric_average is not None else ""}'] = MulticlassPrecision(num_classes=num_classes, average=metric_average)
        elif metric_name == "recall":
            if "average" not in metric:
                raise ValueError("Recall metric requires 'average' parameter (e.g. 'macro' or 'weighted' or 'none')")
            produces_metrics[f'recall_{metric_average if metric_average is not None else ""}'] = MulticlassRecall(num_classes=num_classes, average=metric_average)
        else:
            raise ValueError(f"Unsupported metric: {metric_name}")
        
    return MetricCollection(produces_metrics, prefix=prefix)