from collections import namedtuple
import glob
import json
import os
from typing import List

from matplotlib.colors import hsv_to_rgb
from matplotlib.legend_handler import HandlerTuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from utils.bootstrap_confidence import bootstrap_confidence

from utils.mplsetting import get_costom_rcparams

plt.rcParams.update(get_costom_rcparams())

Result = namedtuple('Result', ['mean_eval_success', 'std_eval_success',
                               'mean_eval_failure', 'std_eval_failure', 'converged_rate', 'mean_eval_to_global', 'eval_total', 'converged_to_global'])

color = [hsv_to_rgb((260.0/360.0, 0.5, 0.85)), hsv_to_rgb((120.0 / 360.0, 0.5, 0.7)), hsv_to_rgb((37.0/360.0, 0.93, 0.95))]


def get_mean_eval_to_global(evals, is_converged):
    evals = np.array(evals)
    is_converged = np.array(is_converged)

    p = np.mean(is_converged)
    suc = np.mean(evals[is_converged])
    fail = np.mean(evals[np.logical_not(is_converged)])
    if p == 0:
        return np.inf
    if p != 1:
        return min(suc, fail) + fail * (1-p) / p
    else:
        return suc


def wrapper_bootstrap(samples):
    return get_mean_eval_to_global(samples[:, 0], samples[:, 1])


def plot_expected_eval(results, funs):
    for fun in funs:
        fig, ax = plt.subplots()
        method_name = ["GAS", "CMA-ES", "QuADS", ]
        for method_i, method in enumerate(["grover", "cmaes", "quads"]):
            line_x = []
            line_y = []
            errors = []
            for dim in range(2, 13):
                suggest_name = method+"_"+fun+"_"+str(dim)
                if suggest_name in results:
                    result = results[suggest_name]
                    conf_interval = bootstrap_confidence(np.stack([result.eval_total, result.converged_to_global], axis=-1), wrapper_bootstrap, n_bootstrap=5000, alpha=0.05, )

                    errors.append(conf_interval)
                    line_x.append(dim)
                    line_y.append(result.mean_eval_to_global)


            for errors_i, (min_interval, max_interval) in enumerate(errors):
                if max_interval == np.inf:
                    x_quiver = errors_i + 2 + 0.05 * method_i
                    ax.annotate('', xytext=(x_quiver, min_interval), xy=(errors_i+2+0.05*method_i, min_interval*10), arrowprops=dict(arrowstyle='->', connectionstyle='arc3', facecolor=color[method_i], edgecolor=color[method_i], lw=1.5, shrinkA=0))
            errorbar = np.abs(np.array(errors).T-np.array(line_y))
            ax.errorbar(np.array(line_x)+0.05*method_i, line_y, yerr=errorbar, label=method_name[method_i], capsize=5, ecolor=color[method_i], color=color[method_i])

        ax.set_ylim(1, 1000000)
        ax.set_yscale("log")
        ax.legend(loc='lower right')
        ax.set_xlabel("dim")
        ax.set_ylabel("Oracle call counts")
        fig.tight_layout()
        fig.savefig(f"expected_eval_{fun}.pdf")


def plot_evals(results: List[Result], funs):
    for fun in funs:
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.5))
        for i, method in enumerate(["quads", "grover", "cmaes"]):
            eval_sums_list = []
            rate_list = []
            dims = []
            for dim in range(2, 13):
                suggest_name = method+"_"+fun+"_"+str(dim)
                if suggest_name in results:
                    dims.append(dim)
                    result = results[suggest_name]
                    eval_sums_list.append(result.eval_total)
                    rate_list.append(result.converged_rate)

            ax[i].boxplot(eval_sums_list, positions=dims, whis=[0., 100.])
            ax[i].set_yscale('log')
            ax[i].set_xlim(1, 11)
            ax[i].set_ylim(1, 1000000)
            ax_twin = ax[i].twinx()
            ax_twin.plot(dims, rate_list)
            ax_twin.set_ylim(0, 1.01)

        # ax_twin.set_ylabel("convergence rate")
        # ax[0].set_title("proposed")
        # ax[0].set_xlabel("dimension")
        # ax[0].set_ylabel("oracle eval count")
        # ax[1].set_title("grover adaptive")
        # ax[1].set_xlabel("dimension")
        # ax[2].set_title("cmaes")
        # ax[2].set_xlabel("dimension")
        ax_twin.set_ylabel("収束割合")
        ax[0].set_title("QuADS")
        ax[0].set_xlabel("次元")
        ax[0].set_ylabel("関数評価回数")
        ax[1].set_title("グローバー適応探索")
        ax[1].set_xlabel("次元")
        ax[2].set_title("CMA-ES")
        ax[2].set_xlabel("次元")
        fig.tight_layout()
        fig.savefig(f"evals_{fun}.pdf")
        

if __name__ == '__main__':

    api = wandb.Api()
    runs = api.runs(f"preview-control/mitou-quads-classical2")

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

    funs = ["schwefel", "styblinski_tang", "rastrigin", ]

    plot_expected_eval(results, funs)
    # plot_evals(results, funs)
