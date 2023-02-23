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

def confidence_str(mean, std):

    if std is not None:
        return '{:.4g}'.format(mean) + '\\pm '+ '{:.4g}'.format(std)
    else:
        return '---'


def get_mean_eval_to_global(suc, fail, p):
    if p != 1:
        return suc + fail * (1-p) / p
    else:
        return suc


if __name__ == '__main__':
    Result = namedtuple('Result', ['mean_eval_success', 'std_eval_success',
                        'mean_eval_failure', 'std_eval_failure', 'converged_rate', 'mean_eval_to_global'])

    api = wandb.Api()
    runs = api.runs(f"morim3/mitou-quads")

    results = {}

    for run in runs:
        summary = run.summary
        results[run.name] = Result(summary["mean_eval_success"], summary["std_eval_success"], summary["mean_eval_failure"], summary["std_eval_failure"], summary["converged_rate"], summary["mean_eval_to_global"])

    table = []

    funs = ["schwefel", "rastrigin", "styblinski_tang", "ackley", "squared", "easom"]
    for fun in funs:
        table.append([])
        # table[-1].append(confidence_str(results["quads_"+fun].mean_eval_success, results["quads_"+fun].std_eval_success))
        # table[-1].append(confidence_str(results["grover_"+fun].mean_eval_success, results["grover_"+fun].std_eval_success))
        # table[-1].append(confidence_str(results["cmaes_"+fun].mean_eval_success, results["cmaes_"+fun].std_eval_success))
        # table[-1].append(confidence_str(results["quads_"+fun].mean_eval_failure, results["quads_"+fun].std_eval_failure))
        # table[-1].append(confidence_str(results["grover_"+fun].mean_eval_failure, results["grover_"+fun].std_eval_failure))
        # table[-1].append(confidence_str(results["cmaes_"+fun].mean_eval_failure, results["cmaes_"+fun].std_eval_failure))
        # table[-1].append('{:.4g}'.format(results["quads_"+fun].converged_rate))
        # table[-1].append('{:.4g}'.format(results["grover_"+fun].converged_rate))
        # table[-1].append('{:.4g}'.format(results["cmaes_"+fun].converged_rate))
        table[-1].append('{:.4g}'.format(get_mean_eval_to_global(results["quads_"+fun].mean_eval_success, results["quads_"+fun].mean_eval_failure, results["quads_"+fun].converged_rate)))
        table[-1].append('{:.4g}'.format(get_mean_eval_to_global(results["grover_"+fun].mean_eval_success, results["grover_"+fun].mean_eval_failure, results["grover_"+fun].converged_rate)))
        table[-1].append('{:.4g}'.format(get_mean_eval_to_global(results["cmaes_"+fun].mean_eval_success, results["cmaes_"+fun].mean_eval_failure, results["cmaes_"+fun].converged_rate)))
            

    print(pd.DataFrame(table, index=funs).to_latex())

