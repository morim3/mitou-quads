from typing import Callable, List

import numpy as np
from numpy.typing import NDArray

from models.amp_sim.sampler import sampling_grover_oracle

def optimal_amplify_num(p):
    return np.arccos(np.sqrt(p)) / 2 / np.arcsin(np.sqrt(p))

def uniform_sampling_classical(func, dim, threshold, oracle_eval_limit):
    n_eval = 0

    while True:
        sample = np.random.uniform(0.0, 1.0, dim)

        func_val = func(sample)
        n_eval += 1

        if func_val < threshold:
            threshold = func_val
            break

        if n_eval > oracle_eval_limit:
            raise TimeoutError

    p = 1 / n_eval
    n_eval_estimated = optimal_amplify_num(p) + 1

    return sample, func_val, n_eval_estimated


def run_grover_minimization(
    func: Callable[[NDArray], NDArray],
    config,
    verbose=False
):

    threshold: float = config["init_threshold"]
    eval_num_hist = []
    threshold_hist = []
    dist_target_hist = []

    dim = config["n_dim"]
    target = config["target"]

    for i in range(config["max_iter"]):

        try:

            if config["sampler_type"] == "quantum":
                x, y, eval_num = sampling_grover_oracle(
                    func, None, None, config["n_digits"], dim, threshold, uniform=True, optimal_amplify_num=config["optimal_amplify_num"], 
                    oracle_eval_limit=config["eval_limit_one_sample"]
                )
            elif config["sampler_type"] == "classical":
                x, y, eval_num = uniform_sampling_classical(func, dim, threshold, config["eval_limit_one_sample"])
            else:
                raise ValueError("Invalid sampler type")

        except TimeoutError:
            break

        eval_num_hist.append(eval_num)
        threshold = y
        threshold_hist.append(threshold)
        dist_target = np.linalg.norm(x-target)
        dist_target_hist.append(dist_target)

        if verbose:
            print("-------")
            print("accepted", x)
            print("accepted_val", y)
            print("iter: ", i)
            print("threshold: ", threshold)
            print("eval_num: ", eval_num_hist[-1])
            print("--------")

        if dist_target < config["terminate_eps"]:
            break

    if verbose:
        print("total_eval_num: ", sum(eval_num_hist))
    return x, (np.array(threshold_hist), np.array(eval_num_hist), np.array(dist_target_hist))


if __name__ == "__main__":
    init_threshold = 1.
    init_mean, init_cov = np.array([0.5, 0.5]), np.array([[1, 0], [0, 1]])

    target = [0.2, 0.8]
    config = {
        "sampler_type": "quantum",
        "target": target,
        "dim": 2,
        "init_threshold": 1.,
        "n_digits": 8,
        "max_iter": 100,
        "terminate_eps": 0.01,
        "eval_limit_one_sample": 10000,
        "optimal_amplify_num": True,
    }


    def func(x): return (20 + (10*(x[..., 0]-target[0])) ** 2 + (10*(x[..., 1]-target[1])) ** 2 - 10 * np.cos(
        2*np.pi*(10*(x[..., 0]-target[0]))) - 10 * np.cos(2*np.pi*(10*(x[..., 1]-target[1])))) / 40

    result_param, (param_hist, eval_num_hist) = run_grover_minimization(
        func, config, verbose=True)
