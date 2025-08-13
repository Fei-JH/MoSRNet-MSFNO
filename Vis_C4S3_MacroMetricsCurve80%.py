import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import os

plt.rcParams['font.family'] = 'Helvetica'
sns.set_theme(style="whitegrid", font_scale=1.4)

# 参数设置
n_elem = 540
left_skip = 54
right_skip = 54
valid_start = left_skip
valid_end = n_elem - right_skip
bin_range = [9, 10, 12, 15, 18, 20, 27, 30, 36]
output_dir = "./postprocess"
os.makedirs(output_dir, exist_ok=True)

# === 读取并过滤数据，仅保留目标区间 ===
df = pd.read_csv("./MSC_test/msc_maxpos_rawidx.csv")
df = df[(df["true_idx"] >= valid_start) & (df["true_idx"] < valid_end)].reset_index(drop=True)

case_names = ["msfno", "resnet", "msfno_int", "resnet_int"]
idx_keys = ["msfno_idx", "resnet_idx", "msfno_int_idx", "resnet_int_idx"]

results = {case: {"n_bins": [], "acc": [], "prec": [], "rec": [], "f1": []} for case in case_names}

for n_bins in bin_range:
    bin_edges = np.linspace(valid_start, valid_end, n_bins + 1, dtype=int)
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

# ===== 绘图和保存 =====
metric_names = ["acc", "prec", "rec", "f1"]
metric_labels = {
    "acc": "Accuracy",
    "prec": "Macro Precision",
    "rec": "Macro Recall",
    "f1": "Macro F1-score"
}
palette = sns.color_palette("tab10", n_colors=len(case_names))

fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
axes = axes.flatten()

for idx, metric in enumerate(metric_names):
    ax = axes[idx]
    for ci, case in enumerate(case_names):
        ax.plot(
            results[case]["n_bins"],
            np.array(results[case][metric]) * 100,
            label=case.upper(),
            linewidth=2.5,
            marker='o',
            markersize=6,
            color=palette[ci]
        )
    ax.set_ylabel(f"{metric_labels[metric]} (%)", fontsize=15, weight='bold')
    ax.set_xlabel("Number of Bins", fontsize=15, weight='bold')
    ax.set_title(metric_labels[metric], fontsize=16, weight='bold', pad=8)
    ax.set_xlim(7, 38)
    ax.set_xticks(bin_range)
    ax.set_ylim(15, 105)
    ax.set_yticks(np.arange(20, 101, 20))
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, linestyle="--", alpha=0.6)

axes[0].legend(loc='lower left', fontsize=13, ncol=2, frameon=True, fancybox=True, shadow=True)
fig.suptitle("Classification Metrics vs. Number of Bins (Only Central 432 Units)", fontsize=20, weight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.96])

save_fig_path = os.path.join(output_dir, "metrics_vs_bins_central432_filter_data.png")
plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
plt.close()

# 保存csv
for metric in metric_names:
    df_metric = pd.DataFrame({case: results[case][metric] for case in case_names}, index=list(bin_range))
    df_metric.index.name = "n_bins"
    df_metric.to_csv(os.path.join(output_dir, f"metrics_{metric}_vs_bins_central432_filter_data.csv"))

print(f"主指标曲线和csv已保存到 {output_dir}，仅中央432单元的样本被统计。")
