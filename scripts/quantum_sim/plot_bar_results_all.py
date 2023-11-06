from collections import namedtuple
import glob
import json
import os

from matplotlib.colors import hsv_to_rgb
from matplotlib.legend_handler import HandlerTuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from utils.bootstrap_confidence import bootstrap_confidence
from utils.mplsetting import get_custom_rcparams

customrc = {
    # 'axes.labelsize': 8.8,
    # 'axes.titlesize': 9.6,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'font.size': 20,

    # font
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'stix',  # Computer Modern
    'ytick.direction': 'in',
}
plt.rcParams.update(customrc)

Result = namedtuple('Result', ['mean_eval_success', 'std_eval_success',
                               'mean_eval_failure', 'std_eval_failure', 'converged_rate', 'mean_eval_to_global', 'eval_total', 'converged_to_global'])

def confidence_str(mean, std):
    if std is not None:
        return '{:.4g}'.format(mean) + '\\pm '+ '{:.4g}'.format(std)
    else:
        return '---'

def get_mean_eval_to_global(evals, is_converged):
    evals = np.array(evals)
    is_converged = np.array(is_converged)

    p = np.mean(is_converged)
    suc = np.mean(evals[is_converged])
    fail = np.mean(evals[np.logical_not(is_converged)])
    if p == 0:
        return np.inf
    if p != 1:
        return suc + fail * (1-p) / p
    else:
        return suc


def wrapper_bootstrap(samples):
    return get_mean_eval_to_global(samples[:, 0], samples[:, 1])

if __name__ == '__main__':

    api = wandb.Api()
    runs = api.runs(f"preview-control/mitou-quads-quantum2")

    results = {}

    for run in runs:
        summary = run.summary
        if "mean_eval_success" not in summary:
            continue
        results[run.name] = Result(
            summary["mean_eval_success"],
            summary["std_eval_success"],
            summary["mean_eval_failure"],
            summary["std_eval_failure"],
            summary["converged_rate"],
            get_mean_eval_to_global(summary["eval_total"], summary["converged_to_global"]),
            summary["eval_total"],
            summary["converged_to_global"]
        )

    table = []

    methods = ["grover", "cmaes", "quads"]
    methods_label = ["GAS", "CMA-ES", "QuADS"]
    funs = [ "rastrigin", "schwefel","styblinski_tang",  "ackley","rosenbrock",  "alpine01", "alpine02", "deflectedCorrugatedSpring", "griewank",
            # "mishra",
           "squared", "wavy"]
    synonims = { "rastrigin": "rastrigin", "schwefel": "schwefel" ,"styblinski_tang": "styblinski tang",  "ackley": "ackley","rosenbrock": "rosenbrock",  "alpine01": "alpine01", "alpine02":"apline02", "deflectedCorrugatedSpring": "deflected corrugated spring", "griewank": "griewank", "squared": "squared", "wavy": "wavy"}
    for fun in funs:
        table.append([])

        for method in methods:
            if fun == "easom":
                result = results[method+"_"+fun+"_2"]
            else:
                result = results[method+"_"+fun+"_3"]

            base = result.mean_eval_to_global
            conf_interval = bootstrap_confidence(np.stack([result.eval_total, result.converged_to_global], axis=-1), wrapper_bootstrap, n_bootstrap=5000, alpha=0.05, )
            table[-1].append(conf_interval[0])
            table[-1].append(base)
            table[-1].append(conf_interval[1])

    print(pd.DataFrame(table, index=funs).to_latex())

    df = pd.DataFrame(table, columns=pd.MultiIndex.from_product([methods, ['lower', 'base', 'upper']]), index=funs)

    # 棒グラフの描画
    fig, ax = plt.subplots(figsize=(15, 8))  # 横長の図のサイズを設定

    width = 0.25  # 棒グラフの幅
    colors = [hsv_to_rgb((260.0/360.0, 0.5, 0.85)), hsv_to_rgb((120.0 / 360.0, 0.5, 0.7)), hsv_to_rgb((30.0/360.0, 0.5, 0.95))]  # 手法ごとに異なる色を設定

        # データをプロット
    for i, method in enumerate(methods):
        positions = np.arange(len(funs)) + i * width
        means = df[method, 'base']
        lower = df[method, 'base'] - df[method, 'lower']
        upper = df[method, 'upper'] - df[method, 'base']
        # replace inf
        big_value = 1e10
        upper = np.where(np.isinf(upper), big_value, upper)
        print(means, lower, upper, type(upper))

        ax.bar(positions, means, width=width, label=methods_label[i], color=colors[i], yerr=[lower, upper], capsize=5, log=True)

    # 軸とタイトルの設定
    ax.set_ylabel('Expected oracle call count (log scale)')
    # ax.set_title('Comparative Evaluation Counts per Method')
    ax.set_ylim(1, 5e5)
    ax.set_xticks(np.arange(len(funs)) + width)
    ax.set_xticklabels([synonims[f] for f in funs])
    plt.setp(ax.get_xticklabels(), rotation=20, horizontalalignment='right')

    # 凡例を上部に表示
    ax.legend(loc='upper left', ncol=3)

    # プロットの表示
    plt.tight_layout()  # レイアウトの自動調整
    plt.savefig("outputs/bar_all.pdf")
