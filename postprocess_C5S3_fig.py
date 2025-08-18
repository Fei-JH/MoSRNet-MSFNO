import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, precision_recall_fscore_support
import os
import pandas as pd

# ---------- Matplotlib global styles ----------
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 28
matplotlib.rcParams['axes.labelsize'] = 24
matplotlib.rcParams['legend.fontsize'] = 24
matplotlib.rcParams['axes.titlesize'] = 28
matplotlib.rcParams['xtick.labelsize'] = 24 
matplotlib.rcParams['ytick.labelsize'] = 24


def C5S3_plot_cm(
    y_true, y_pred, save_path=None, font_size=14, title="", show=True
):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    n_bins = len(labels)

    # 类别标签从1开始
    class_labels = [str(i+1) for i in range(n_bins)]
    xticklabels = class_labels + ['R']
    yticklabels = class_labels + ['P']

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
    cm_percent = np.nan_to_num(cm_percent)

    # 计算precision, recall
    precisions, recalls, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # 每个格子的注释（频数加粗由fontweight控制，百分比不加粗）
    annot = np.empty(cm.shape, dtype=object)
    for i in range(n_bins):
        for j in range(n_bins):
            count = cm[i, j]
            percent = cm_percent[i, j]
            # 只放字符串，不加latex
            annot[i, j] = f"{count}\n{percent:.1f}%"

    # 扩展混淆矩阵与注释到 (n+1, n+1)
    cm_full = np.zeros((n_bins + 1, n_bins + 1))
    cm_full[:-1, :-1] = cm_percent
    annot_full = np.empty((n_bins + 1, n_bins + 1), dtype=object)
    annot_full[:-1, :-1] = annot

    # Precision: 每列的总和+百分比
    for j in range(n_bins):
        col_sum = cm[:, j].sum()
        precision_percent = precisions[j] * 100
        annot_full[-1, j] = f"{col_sum}\n{precision_percent:.1f}%"
        cm_full[-1, j] = precision_percent  # 用于着色

    # Recall: 每行的总和+百分比
    for i in range(n_bins):
        row_sum = cm[i, :].sum()
        recall_percent = recalls[i] * 100
        annot_full[i, -1] = f"{row_sum}\n{recall_percent:.1f}%"
        cm_full[i, -1] = recall_percent

    # 右下角
    annot_full[-1, -1] = f"Acc:\n{acc*100:.1f}%"
    cm_full[-1, -1] = acc * 100

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 10))
    # 加粗控制在 annot_kws 里
    if n_bins > 15:
        annot_heatmap = False
        annot_kws = None
    else:
        annot_heatmap = annot_full
        annot_kws = {"size": font_size, "va": "center", "fontweight": "bold", "fontfamily": "Times New Roman"}

    sns.heatmap(
        cm_full, annot=annot_heatmap, fmt='', cmap="GnBu", cbar=False,
        xticklabels=xticklabels, yticklabels=yticklabels,
        linewidths=0.25, linecolor='black', square=True,
        annot_kws=annot_kws if annot_kws is not None else {}
    )

    # 主对角线绿色框
    for i in range(n_bins):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='green', lw=2))

    # 轴标题与标签
    axis_fontsize = font_size + 3
    label_fontsize = font_size + 3
    ax.set_xlabel("Predicted Damage Location (Element Index)", fontsize=axis_fontsize+2, weight='bold', labelpad=5)
    ax.set_ylabel("Actual Damage Location (Element Index)", fontsize=axis_fontsize+2, weight='bold', labelpad=5)
    ax.set_xticklabels(xticklabels, fontsize=label_fontsize, rotation=0, ha='center', weight='normal')
    ax.set_yticklabels(yticklabels, fontsize=label_fontsize, rotation=0, va='center', weight='normal')

    # 添加同高cbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.3)
    im = ax.collections[0]
    cb = plt.colorbar(im, cax=cax, orientation='vertical')
    cb.ax.tick_params(labelsize=font_size + 3, width=1.7, length=6, labelcolor='black')
    cb.ax.set_ylim(0, 100)
    im.set_cmap('OrRd')
    im.set_clim(0, 100)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close()

    # 返回主指标
    return acc, macro_prec, macro_recall, macro_f1

#%% 下面是后处理与主流程

# 后处理可自定义bin数
n_elem = 540
for n in [10, 15, 18, 24]:
    n_bins = n  # 可任意修改
    bin_edges = np.linspace(0, n_elem, n_bins + 1, dtype=int)

    def digitize_pos(pos_list):
        pos_bin = np.digitize(pos_list, bin_edges) - 1
        pos_bin = np.clip(pos_bin, 0, n_bins - 1)
        return pos_bin

    df = pd.read_csv("./results/mcs_test/msc_maxpos_rawidx.csv")
    # 动态分箱
    true_bin = digitize_pos(df["true_idx"].values)
    msfno_bin = digitize_pos(df["msfno_idx"].values)
    resnet_bin = digitize_pos(df["resnet_idx"].values)
    msfno_int_bin = digitize_pos(df["msfno_int_idx"].values)
    resnet_int_bin = digitize_pos(df["resnet_int_idx"].values)

    cases = {
        "msfno": msfno_bin,
        "resnet": resnet_bin,
        "msfno_int": msfno_int_bin,
        "resnet_int": resnet_int_bin
    }
    true_labels = true_bin

    metrics_dict = {
        "Acc": {},
        "Prec": {},
        "Rec": {},
        "F1": {}
    }

    output_dir = "./postprocessed"
    os.makedirs(output_dir, exist_ok=True)

    for case_name, pred_bin in cases.items():
        fig_name = f"C4S3_confmat_{case_name}_bin={n_bins}.png"
        fig_path = os.path.join(output_dir, "c5s3", "fig")
        os.makedirs(fig_path, exist_ok=True)
        acc, prec, rec, f1 = C5S3_plot_cm(
            true_labels, pred_bin, 
            save_path=os.path.join(fig_path, fig_name), 
            font_size=14,
            title=f"{case_name.upper()}, bin={n_bins}", show=False
        )
        metrics_dict["Acc"][case_name] = acc
        metrics_dict["Prec"][case_name] = prec
        metrics_dict["Rec"][case_name] = rec
        metrics_dict["F1"][case_name] = f1

    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df = metrics_df.T  # 行为指标
    csv_path = os.path.join(output_dir, "c5s3", "csv")
    os.makedirs(csv_path, exist_ok=True)
    metrics_df.to_csv(os.path.join(csv_path, f"C4S3_main_metrics_bin={n_bins}.csv") )

    print(f"所有混淆矩阵图片和主指标已保存 bin={n_bins}")
