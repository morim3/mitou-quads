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
from scipy.stats import linregress
from utils.bootstrap_confidence import bootstrap_confidence

from utils.mplsetting import get_custom_rcparams

customrc = {
    # 'axes.labelsize': 8.8,
    # 'axes.titlesize': 9.6,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'font.size': 20,

    'font.family': 'Times New Roman',
    'mathtext.fontset': 'stix',  # Computer Modern
    'xtick.direction': 'in',
    'ytick.direction': 'in',
}

plt.rcParams.update(customrc)

Result = namedtuple('Result', ['mean_eval_success', 'std_eval_success',
                               'mean_eval_failure', 'std_eval_failure',
                               'converged_rate', 'mean_eval_to_global',
                               'eval_total', 'converged_to_global'])

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
        return suc + fail * (1-p) / p
    else:
        return suc


def wrapper_bootstrap(samples):
    return get_mean_eval_to_global(samples[:, 0], samples[:, 1])


def plot_expected_eval(classical_results, quantum_results, funs):
    for fun in funs:
        fig, ax = plt.subplots(figsize=(8, 6))
        method_name = ["GAS", "CMA-ES", "QuADS"]

        # classical results
        
        for method_i, method in enumerate(["grover", "cmaes", "quads"]):
            line_x = []
            line_y = []
            # errors = []
            for dim in range(1, 13):
                suggest_name = method+"_"+fun+"_"+str(dim)
                if suggest_name in classical_results:
                    result = classical_results[suggest_name]
                    # conf_interval = bootstrap_confidence(np.stack([result.eval_total, result.converged_to_global], axis=-1), wrapper_bootstrap, n_bootstrap=5000, alpha=0.05)
                    #
                    # errors.append(conf_interval)
                    line_x.append(dim)
                    if method == "cmaes":
                        line_y.append(result.mean_eval_to_global)
                    else:
                        line_y.append(result.mean_eval_to_global * 2)

            # for errors_i, (min_interval, max_interval) in enumerate(errors):
            #     if max_interval == np.inf:
            #         x_quiver = errors_i + 2 + 0.05 * method_i
            #         ax.annotate('', xytext=(x_quiver, min_interval), xy=(errors_i+2+0.05*method_i, min_interval*10), arrowprops=dict(arrowstyle='->', connectionstyle='arc3', facecolor=color[method_i], edgecolor=color[method_i], lw=1.5, shrinkA=0))
            # errorbar = np.abs(np.array(errors).T-np.array(line_y))
            # ax.errorbar(np.array(line_x)+0.05*method_i, line_y,
            #             yerr=errorbar, label=method_name[method_i], capsize=5,
            #             ecolor=color[method_i], color=color[method_i],
            #             marker='o', linestyle=''
            # )
            ax.plot(np.array(line_x)+0.05*method_i, line_y, label=method_name[method_i], color=color[method_i], marker='o', linestyle='')

            finite_inds = np.isfinite(line_y)
            finite_line_x = np.array(line_x)[finite_inds]
            finite_line_y = np.array(line_y)[finite_inds]

            # calculate regression line
            slope, intercept, r_value, p_value, std_err = linregress(finite_line_x, np.log10(finite_line_y))
            r_squared = r_value**2
            
            regression_line_x = np.array(finite_line_x)
            regression_line_y = 10**(intercept + slope * regression_line_x)

            print(method, finite_line_x, finite_line_y)
            
            # plot regression line
            ax.plot(regression_line_x + 0.05 * method_i, regression_line_y,
                    # linestyle='--',
                    color=color[method_i])
            
            if method == "cmaes":
                ax.text(regression_line_x[-3] - 2.25, regression_line_y[-3] * 1.5 ,
                        '$o_{\\rm total}\\approx' + f'{10**intercept:.2f} \\times 10^{{{slope:.2f}d}}$\n' + f'$r^2 = {r_squared:.3f}$',
                        color=color[method_i],
                        fontsize=15)

            else:
                ax.text(regression_line_x[-3] - 2.25, regression_line_y[-3] * 1.5 ,
                        '$\\tilde o_{\\rm total}\\approx' + f'{10**intercept:.2f} \\times 10^{{{slope:.2f}d}}$\n' + f'$r^2 = {r_squared:.3f}$',
                        color=color[method_i],
                        fontsize=15)
        
        # quantum results
        for method_i, method in enumerate(["grover", "cmaes", "quads"]):
            if method == "cmaes":
                continue
            line_x = []
            line_y = []
            for dim in range(1, 13):
                suggest_name = method+"_"+fun+"_"+str(dim)
                if suggest_name in quantum_results:
                    result = quantum_results[suggest_name]
                    line_x.append(dim)
                    line_y.append(result.mean_eval_to_global)

            ax.plot(line_x, line_y,
                    linestyle='--',
                    color=color[method_i] * 0.75, marker='o')

        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xticks(range(1, 11))
        ax.set_ylim(1, 1000000)
        ax.set_yscale("log")
        ax.legend(loc='lower right')
        ax.set_xlabel("dimension")
        ax.set_ylabel("expected oracle call counts\n to global optimum")
        fig.tight_layout()
        fig.savefig(f"outputs/expected_eval_{fun}.pdf")
        fig.savefig(f"outputs/expected_eval_{fun}.svg")


# def plot_evals(results: List[Result], funs):
#     for fun in funs:
#         fig, ax = plt.subplots(1, 3, figsize=(11, 3.5))
#         for i, method in enumerate(["quads", "grover", "cmaes"]):
#             eval_sums_list = []
#             rate_list = []
#             dims = []
#             for dim in range(2, 13):
#                 suggest_name = method+"_"+fun+"_"+str(dim)
#                 if suggest_name in results:
#                     dims.append(dim)
#                     result = results[suggest_name]
#                     eval_sums_list.append(result.eval_total)
#                     rate_list.append(result.converged_rate)
#
#             ax[i].boxplot(eval_sums_list, positions=dims, whis=[0., 100.])
#             ax[i].set_yscale('log')
#             ax[i].set_xlim(1, 11)
#             ax[i].set_ylim(1, 1000000)
#             ax_twin = ax[i].twinx()
#             ax_twin.plot(dims, rate_list)
#             ax_twin.set_ylim(0, 1.01)
#
#         ax_twin.set_ylabel("convergence rate")
#         ax[0].set_title("proposed")
#         ax[0].set_xlabel("dimension")
#         ax[0].set_ylabel("oracle eval count")
#         ax[1].set_title("grover adaptive")
#         ax[1].set_xlabel("dimension")
#         ax[2].set_title("cmaes")
#         ax[2].set_xlabel("dimension")
#         # ax_twin.set_ylabel("収束割合")
#         # ax[0].set_title("QuADS")
#         # ax[0].set_xlabel("次元")
#         # ax[0].set_ylabel("関数評価回数")
#         # ax[1].set_title("グローバー適応探索")
#         # ax[1].set_xlabel("次元")
#         # ax[2].set_title("CMA-ES")
#         # ax[2].set_xlabel("次元")
#         fig.tight_layout()
#         fig.savefig(f"outputs/evals_{fun}.pdf")
#         

def optimal_to_upper_bound(optimal_bound):
    # optimal_num = pi/2-arcsin(sqrt(p))
    p = np.sin(np.pi / 2 / (2*optimal_bound+1)) ** 2
    # upper bound is 1/(2*sqrt(p*(1-p))) * 9/2 when lambda is 6/5
    return 1 / (2*np.sqrt(p*(1-p))) * 9 / 2

if __name__ == '__main__':

    api = wandb.Api()
    classical_results = {}
    quantum_results = {}
    runs = api.runs(f"preview-control/mitou-quads-classical2")
    for run in runs:
        if run.state != "finished":
            continue
        summary = run.summary
        if "mean_eval_success" not in summary:
            continue

        classical_results[run.name] = Result(
            summary["mean_eval_success"],
            summary["std_eval_success"],
            summary["mean_eval_failure"],
            summary["std_eval_failure"],
            summary["converged_rate"],
            get_mean_eval_to_global(summary["eval_total"], summary["converged_to_global"]),
            summary["eval_total"],
            summary["converged_to_global"]
        )
    
    runs = api.runs(f"preview-control/mitou-quads-quantum2")
    for run in runs:
        summary = run.summary
        if "mean_eval_success" not in summary:
            continue

        quantum_results[run.name] = Result(
            summary["mean_eval_success"],
            summary["std_eval_success"],
            summary["mean_eval_failure"],
            summary["std_eval_failure"],
            summary["converged_rate"],
            get_mean_eval_to_global(summary["eval_total"], summary["converged_to_global"]),
            summary["eval_total"],
            summary["converged_to_global"]
        )
        # # we just log lower bound, so recalculate upper bound from lower bound.
        # if run.config["method"] == "grover" or run.config["method"] == "quads":
        #     artifact = api.artifact("preview-control/mitou-quads-classical2/" + run.logged_artifacts()[0].name )
        #     table = artifact.get("optimization process")
        #     table = np.array(table.data)
        #     lower_eval_total = []
        #     upper_eval_total = []
        #     for i in range(99):
        #         iteri = table[table[:, 0] == i]
        #         eval_lower_bound = np.concatenate([[iteri[0, 1]], iteri[1:, 1] - iteri[:-1, 1]])
        #         if run.config["method"] == "grover":
        #             upper_bound = optimal_to_upper_bound(eval_lower_bound-1) + 1
        #         elif run.config["method"] == "quads":
        #             upper_bound = (optimal_to_upper_bound(eval_lower_bound / run.config["n_samples"] - 1) + 1) * run.config["n_samples"]
        #
        #         upper_eval_total.append(np.sum(upper_bound))
        #
        #     upper_eval_total = np.array(upper_eval_total)
        #     print(upper_eval_total)
        #     print(summary["eval_total"])
        #
        # else:
        #     upper_eval_total = summary["eval_total"]
        #
        # results[run.name] = Result(

    table = []

    funs = ["schwefel", "styblinski_tang", "rastrigin", "ackley", "alpine01", "griewank"]

    plot_expected_eval(classical_results, quantum_results, funs)
    # plot_evals(results, funs)
