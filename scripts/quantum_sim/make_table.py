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

    funs = ["schwefel", "rastrigin", "styblinski_tang", "ackley", "squared"]
    for fun in funs:
        table.append([])

        for method in ["quads", "grover", "cmaes"]:
            if fun == "easom":
                result = results[method+"_"+fun+"_2"]
            else:
                result = results[method+"_"+fun+"_3"]

            base = result.mean_eval_to_global
            conf_interval = bootstrap_confidence(np.stack([result.eval_total, result.converged_to_global], axis=-1), wrapper_bootstrap, n_bootstrap=5000, alpha=0.05, )
            table[-1].append('{:.4g}'.format(conf_interval[0]))
            table[-1].append('{:.4g}'.format(base))
            table[-1].append('{:.4g}'.format(conf_interval[1]))

    print(pd.DataFrame(table, index=funs).to_latex())

