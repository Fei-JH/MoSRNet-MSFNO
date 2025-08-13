import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import os

# 全局美化设置
plt.rcParams['font.family'] = 'Helvetica'
sns.set_theme(style="whitegrid", font_scale=1.4)

# 参数区
n_elem = 540
bin_range = range(8, 41)  # 8~40
output_dir = "./postprocess"
os.makedirs(output_dir, exist_ok=True)

# 读取原始数据
df = pd.read_csv("./MSC_test/msc_maxpos_rawidx.csv")

case_names = ["msfno", "resnet", "msfno_int", "resnet_int"]
idx_keys = ["msfno_idx", "resnet_idx", "msfno_int_idx", "resnet_int_idx"]

# 结果字典
results = {case: {"n_bins": [], "acc": [], "prec": [], "rec": [], "f1": []} for case in case_names}

for n_bins in bin_range:
    bin_edges = np.linspace(0, n_elem, n_bins + 1, dtype=int)
    # 分箱函数
    def digitize_pos(pos_list):
        pos_bin = np.digitize(pos_list, bin_edges) - 1
        pos_bin = np.clip(pos_bin, 0, n_bins - 1)
        return pos_bin

    true_bin = digitize_pos(df["true_idx"].values)
    for case, idx_key in zip(case_names, idx_keys):
        pred_bin = digitize_pos(df[idx_key].values)
        acc = accuracy_score(true_bin, pred_bin)
        prec = precision_score(true_bin, pred_bin, average='macro', zero_division=0)
        rec = recall_score(true_bin, pred_bin, average='macro', zero_division=0)
        f1 = f1_score(true_bin, pred_bin, average='macro', zero_division=0)
        results[case]["n_bins"].append(n_bins)
        results[case]["acc"].append(acc)
        results[case]["prec"].append(prec)
        results[case]["rec"].append(rec)
        results[case]["f1"].append(f1)

# 曲线绘制
metric_names = ["acc", "prec", "rec", "f1"]
metric_labels = {
    "acc": "Accuracy",
    "prec": "Macro Precision",
    "rec": "Macro Recall",
    "f1": "Macro F1-score"
}

# 配色
palette = sns.color_palette("tab10", n_colors=len(case_names))

fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
axes = axes.flatten()

for idx, metric in enumerate(metric_names):
    ax = axes[idx]
    for ci, case in enumerate(case_names):
        ax.plot(
            results[case]["n_bins"], 
            np.array(results[case][metric]) * 100,  # 百分比
            label=case.upper(), 
            linewidth=2.5, 
            marker='o', 
            markersize=6,
            color=palette[ci]
        )
    ax.set_ylabel(f"{metric_labels[metric]} (%)", fontsize=15, weight='bold')
    ax.set_xlabel("Number of Bins", fontsize=15, weight='bold')
    ax.set_title(metric_labels[metric], fontsize=16, weight='bold', pad=8)
    ax.set_xticks(list(range(8, 41, 4)))
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, linestyle="--", alpha=0.6)

axes[0].legend(loc='lower left', fontsize=13, ncol=2, frameon=True, fancybox=True, shadow=True)
fig.suptitle("Classification Metrics vs. Number of Bins", fontsize=20, weight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.96])

save_fig_path = os.path.join(output_dir, "metrics_vs_bins.png")
plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
plt.close()

# 结果保存为csv
for metric in metric_names:
    df_metric = pd.DataFrame({case: results[case][metric] for case in case_names}, index=list(bin_range))
    df_metric.index.name = "n_bins"
    df_metric.to_csv(os.path.join(output_dir, f"metrics_{metric}_vs_bins.csv"))

print(f"主指标随bin变化的曲线图和csv已保存到 {output_dir}")
