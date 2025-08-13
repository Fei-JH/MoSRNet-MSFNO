'''
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-13 19:15:04
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-08-13 19:15:48
'''


import torch


def compute_dataset_stats(model, dataset, batch_size, device, sample_fn, stats_flags=None):
    """
    Compute statistical metrics for a given dataset using the provided sample function.
    
    Args:
        model (torch.nn.Module): The model to generate predictions.
        dataset (torch.utils.data.Dataset): The dataset from which to compute statistics.
        batch_size (int): Batch size used for processing the dataset.
        device (torch.device): The device to perform computation.
        sample_fn (callable): Function that takes (predictions, ground_truth) as input and returns
                              a 1D torch.Tensor of metric values.
        stats_flags (list of bool or None): A list of 8 boolean values indicating whether to compute
                                            ["std", "cv", "skewness", "kurtosis", "min", 
                                            "max", "median", "variance"]. If None, all are set to False.
        
    Returns:
        dict or None: A dictionary with keys as statistical metric names and values as computed results,
                      or None if all flags are False.
    """
    # 默认不计算所有统计量（全为 False）
    if stats_flags is None:
        stats_flags = [False] * 8  # 全部不计算
    elif not any(stats_flags):
        return None  # 如果所有布尔值均为 False，则直接返回 None，跳过计算

    # 解包布尔值
    (compute_std, compute_cv, compute_skewness, compute_kurtosis,
     compute_min, compute_max, compute_median, compute_variance) = stats_flags

    model.eval()
    metric_batches = []
    with torch.no_grad():
        num_samples = len(dataset)
        for i in range(0, num_samples, batch_size):
            batch_items = [dataset[j] for j in range(i, min(i + batch_size, num_samples))]
            # Assumption: Each item in dataset is a tuple (x, f, y)
            x_batch = torch.stack([item[0] for item in batch_items]).to(device)
            y_batch = torch.stack([item[1] for item in batch_items]).to(device)

            # Forward pass
            outputs = model(x_batch)
            outputs = outputs.view(outputs.size(0), -1)
            y_batch = y_batch.view(y_batch.size(0), -1)

            # Compute metric using the provided function
            metric_values = sample_fn(outputs, y_batch)
            metric_values = metric_values.unsqueeze(0)
            metric_batches.append(metric_values)

    # 合并所有 batch 的数据
    all_metrics = torch.cat(metric_batches, dim=0)

    # 统计结果字典
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
        stats["cv"] = (std_val / mean_val) if mean_val != 0 else float('nan')
    if compute_skewness:
        mean_tensor = torch.mean(all_metrics)
        std_tensor = torch.std(all_metrics, unbiased=False)
        stats["skewness"] = torch.mean(((all_metrics - mean_tensor) / std_tensor) ** 3).item() if std_tensor != 0 else float('nan')
    if compute_kurtosis:
        mean_tensor = torch.mean(all_metrics)
        std_tensor = torch.std(all_metrics, unbiased=False)
        stats["kurtosis"] = (torch.mean(((all_metrics - mean_tensor) / std_tensor) ** 4).item() - 3) if std_tensor != 0 else float('nan')
    if compute_min:
        stats["min"] = torch.min(all_metrics).item()
    if compute_max:
        stats["max"] = torch.max(all_metrics).item()

    return stats


def compute_dataset_stats_mosrnet(model, dataset, batch_size, device, sample_fn, stats_flags=None):
    """
    Compute statistical metrics for a given dataset using the provided sample function.
    
    Args:
        model (torch.nn.Module): The model to generate predictions.
        dataset (torch.utils.data.Dataset): The dataset from which to compute statistics.
        batch_size (int): Batch size used for processing the dataset.
        device (torch.device): The device to perform computation.
        sample_fn (callable): Function that takes (predictions, ground_truth) as input and returns
                              a 1D torch.Tensor of metric values.
        stats_flags (list of bool or None): A list of 8 boolean values indicating whether to compute
                                            ["std", "cv", "skewness", "kurtosis", "min", 
                                            "max", "median", "variance"]. If None, all are set to False.
        
    Returns:
        dict or None: A dictionary with keys as statistical metric names and values as computed results,
                      or None if all flags are False.
    """
    # 默认不计算所有统计量（全为 False）
    if stats_flags is None:
        stats_flags = [False] * 8  # 全部不计算
    elif not any(stats_flags):
        return None  # 如果所有布尔值均为 False，则直接返回 None，跳过计算

    # 解包布尔值
    (compute_std, compute_cv, compute_skewness, compute_kurtosis,
     compute_min, compute_max, compute_median, compute_variance) = stats_flags

    model.eval()
    metric_batches = []
    with torch.no_grad():
        num_samples = len(dataset)
        for i in range(0, num_samples, batch_size):
            batch_items = [dataset[j] for j in range(i, min(i + batch_size, num_samples))]
            # Assumption: Each item in dataset is a tuple (x, f, y)
            mode_down_batch  = torch.stack([item[0] for item in batch_items]).to(device)
            mode_gt_batch    = torch.stack([item[1] for item in batch_items]).to(device)
            dmg_batch        = torch.stack([item[2] for item in batch_items]).to(device)
            Mcond_batch      = torch.stack([item[3] for item in batch_items]).to(device)
            Kcond_batch      = torch.stack([item[4] for item in batch_items]).to(device)

            # Forward pass
            outputs = model(mode_down_batch)
            outputs = outputs.view(outputs.size(0), -1)
            mode_gt_batch = mode_gt_batch.view(mode_gt_batch.size(0), -1)

            # Compute metric using the provided function
            loss_name = sample_fn.__class__.__name__
            if loss_name in ["TVLoss"]:
                # TVLoss: 2D tensor, shape (batch, -1)
                metric_values = sample_fn(outputs)
            elif loss_name in ["MassOrthogonalityLoss"]:
                # MassOrthogonalityLoss: (mode, M)
                metric_values = sample_fn(outputs, Mcond_batch)
            elif loss_name in ["StiffnessOrthogonalityLoss"]:
                # StiffnessOrthogonalityLoss: (mode, K)
                metric_values = sample_fn(outputs, Kcond_batch)
            elif loss_name in ["SCLoss"]:
                # StiffnessConsistencyLoss: (mode_pred, k_true)
                metric_values = sample_fn(outputs, dmg_batch)
            else:
                # Default case for other losses
                metric_values = sample_fn(outputs.view(outputs.shape[0], -1), mode_gt_batch.view(mode_gt_batch.shape[0], -1))
            metric_values = metric_values.unsqueeze(0)
            metric_batches.append(metric_values)

    # 合并所有 batch 的数据
    all_metrics = torch.cat(metric_batches, dim=0)

    # 统计结果字典
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
        stats["cv"] = (std_val / mean_val) if mean_val != 0 else float('nan')
    if compute_skewness:
        mean_tensor = torch.mean(all_metrics)
        std_tensor = torch.std(all_metrics, unbiased=False)
        stats["skewness"] = torch.mean(((all_metrics - mean_tensor) / std_tensor) ** 3).item() if std_tensor != 0 else float('nan')
    if compute_kurtosis:
        mean_tensor = torch.mean(all_metrics)
        std_tensor = torch.std(all_metrics, unbiased=False)
        stats["kurtosis"] = (torch.mean(((all_metrics - mean_tensor) / std_tensor) ** 4).item() - 3) if std_tensor != 0 else float('nan')
    if compute_min:
        stats["min"] = torch.min(all_metrics).item()
    if compute_max:
        stats["max"] = torch.max(all_metrics).item()

    return stats