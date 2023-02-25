from typing import Callable, List

import numpy as np
from numpy.typing import NDArray

from models.amp_sim.sampler import optimal_amplify_num, GroverSampler


def uniform_sampling_classical(func, dim, threshold, oracle_eval_limit):
    n_eval = 0

    while True:

        n_parallel = 100000
        sample = np.random.uniform(0.0, 1.0, (n_parallel, dim))

        func_val = func(sample)
        is_accepted = func_val < threshold
        if np.any(is_accepted):
            accepted_index = np.where(is_accepted)[0][0]
            n_eval += accepted_index + 1
            sample = sample[accepted_index]
            func_val = func_val[accepted_index]
            break
        else:
            n_eval += n_parallel

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
    if config["sampler_type"] == "quantum":
        sampler = GroverSampler(func, config["n_digits"], config["n_dim"])

    eval_num_hist = []
    threshold_hist = []
    dist_target_hist = []

    dim = config["n_dim"]
    target = config["target"]

    for i in range(config["max_iter"]):

        try:
            if config["sampler_type"] == "quantum":
                xs, ys, eval_num = sampler.sample(
                    None, None, threshold, n_samples=1, uniform=True,
                    use_optimal_amplify=config["use_optimal_amplify"],
                    oracle_eval_limit=config["eval_limit_per_update"]
                )
                x, y = xs[0, :], ys[0]
            elif config["sampler_type"] == "classical":
                x, y, eval_num = uniform_sampling_classical(
                    func, dim, threshold, config["eval_limit_per_update"])
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
            print(f"""\
--- iter {i} -----------------
accepted: {x}
accepted_val: {y}
threshold: {threshold}
eval_num: {eval_num_hist[-1]}
----------------------------""")

        if dist_target < config["terminate_eps"]:
            break

    if verbose:
        print("total_eval_num: ", sum(eval_num_hist))
    return x, (np.array(threshold_hist), np.array(eval_num_hist), np.array(dist_target_hist), np.array(threshold_hist))


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
        "eval_limit_per_update": 10000,
        "use_optimal_amplify": True,
    }

    def func(x): return (20 + (10*(x[..., 0]-target[0])) ** 2 + (10*(x[..., 1]-target[1])) ** 2 - 10 * np.cos(
        2*np.pi*(10*(x[..., 0]-target[0]))) - 10 * np.cos(2*np.pi*(10*(x[..., 1]-target[1])))) / 40

    result_param, (param_hist, eval_num_hist) = run_grover_minimization(
        func, config, verbose=True)
