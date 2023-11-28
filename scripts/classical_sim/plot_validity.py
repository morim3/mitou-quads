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

rcparams = get_custom_rcparams()
del rcparams["lines.markeredgewidth"]

plt.rcParams.update(rcparams)

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


def plot_estimation(classical_results, quantum_results, funs):
    fig, ax = plt.subplots(figsize=(8, 6))
    method_name = ["GAS", "CMA-ES", "QuADS"]
    for method_i, method in enumerate(["grover", "cmaes", "quads"]):
        if method == "cmaes":
            continue
        q_line_x = []
        q_line_y = []
        q_errors = []
        line_x = []
        line_y = []
        errors = []

        for fun in funs:
            for dim in range(1, 13):
                suggest_name = method+"_"+fun+"_"+str(dim)
                if suggest_name in classical_results and suggest_name in quantum_results:
                    result = classical_results[suggest_name]
                    line_x.append(dim)
                    line_y.append(result.mean_eval_to_global)

                    result = quantum_results[suggest_name]
                    q_line_x.append(dim)
                    q_line_y.append(result.mean_eval_to_global)

        ax.scatter(line_y, q_line_y,
                color=color[method_i] * 0.75, label=method_name[method_i])
        
    estimation_line_x = np.array([1, 10, 100, 1e3, 1e4, 1e5])
    
    # plot 
    ax.plot(estimation_line_x, estimation_line_x * 2,
            # linestyle='--',
            color="black",
            label=r"$o^{q}_{\rm total} = 2 o^c_{\rm lower}$")

    # ax.plot(estimation_line_x, estimation_line_x * 6,
    #         linestyle='--',
    #         color="grey")
    ax.plot(estimation_line_x, estimation_line_x,
            linestyle='--',
            color="grey",
            label=r"$o^{q}_{\rm total} = o^{c}_{\rm lower}$")

    ax.set_xlim(1, 1e5)
    ax.set_ylim(1, 1e5)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=50))
    ax.legend(loc="lower right" )
    ax.set_xlabel(r"$o^{c}_{\rm lower}$")
    ax.set_ylabel(r"$o^{q}_{\rm total}$")
    fig.tight_layout()
    fig.savefig(f"outputs/estimation_validity.svg")
    fig.savefig(f"outputs/estimation_validity.pdf")

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

    table = []

    funs = ["schwefel", "styblinski_tang", "rastrigin", "ackley", "alpine01", "griewank"]

    plot_estimation(classical_results, quantum_results, funs)
