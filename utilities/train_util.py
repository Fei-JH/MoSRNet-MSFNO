"""
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-13 19:15:04
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-08-14 16:17:03
"""

import torch


def compute_dataset_stats(model, dataset, batch_size, device, sample_fn, stats_flags=None):
    """
    Compute statistical metrics for a dataset using a provided sampling function.

    Args:
        model (torch.nn.Module): Model used to generate predictions.
        dataset (torch.utils.data.Dataset): Dataset that yields (x, y) or (x, f, y).
        batch_size (int): Batch size for evaluation.
        device (torch.device): Compute device.
        sample_fn (callable): Function that maps (pred, target) -> 1D tensor.
        stats_flags (list[bool] or None): Flags for
            ["std", "cv", "skewness", "kurtosis", "min", "max", "median", "variance"].

    Returns:
        dict or None: Metric dictionary, or None if no stats are requested.
    """
    if stats_flags is None:
        stats_flags = [False] * 8
    elif not any(stats_flags):
        return None

    (
        compute_std,
        compute_cv,
        compute_skewness,
        compute_kurtosis,
        compute_min,
        compute_max,
        compute_median,
        compute_variance,
    ) = stats_flags

    model.eval()
    metric_batches = []
    with torch.no_grad():
        num_samples = len(dataset)
        for start_idx in range(0, num_samples, batch_size):
            batch_items = [
                dataset[j] for j in range(start_idx, min(start_idx + batch_size, num_samples))
            ]
            x_batch = torch.stack([item[0] for item in batch_items]).to(device)
            y_batch = torch.stack([item[1] for item in batch_items]).to(device)

            outputs = model(x_batch).view(x_batch.size(0), -1)
            y_batch = y_batch.view(y_batch.size(0), -1)

            metric_values = sample_fn(outputs, y_batch).unsqueeze(0)
            metric_batches.append(metric_values)

    all_metrics = torch.cat(metric_batches, dim=0)
    stats = {}

    if compute_std:
        stats["std"] = torch.std(all_metrics, unbiased=False).item()
    if compute_variance:
        stats["variance"] = torch.var(all_metrics, unbiased=False).item()
    if compute_median:
        stats["median"] = torch.median(all_metrics).item()
    if compute_cv:
        mean_val = torch.mean(all_metrics).item()
        std_val = stats["std"] if "std" in stats else torch.std(all_metrics, unbiased=False).item()
        stats["cv"] = (std_val / mean_val) if mean_val != 0 else float("nan")
    if compute_skewness:
        mean_tensor = torch.mean(all_metrics)
        std_tensor = torch.std(all_metrics, unbiased=False)
        stats["skewness"] = (
            torch.mean(((all_metrics - mean_tensor) / std_tensor) ** 3).item()
            if std_tensor != 0
            else float("nan")
        )
    if compute_kurtosis:
        mean_tensor = torch.mean(all_metrics)
        std_tensor = torch.std(all_metrics, unbiased=False)
        stats["kurtosis"] = (
            torch.mean(((all_metrics - mean_tensor) / std_tensor) ** 4).item() - 3
            if std_tensor != 0
            else float("nan")
        )
    if compute_min:
        stats["min"] = torch.min(all_metrics).item()
    if compute_max:
        stats["max"] = torch.max(all_metrics).item()

    return stats
