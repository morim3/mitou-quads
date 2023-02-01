from typing import Callable, List

import numpy as np
from numpy.typing import NDArray

from sampler import sampling_grover_oracle


def grover_minimization(
    func: Callable[[NDArray], NDArray],
    init_threshold: float,
    iter_num: int,
    target: NDArray,
    n_digits=8,
    dim=2,
    terminate_eps=0.001,
    optimal_amplify_num=False,
    verbose=False
):

    threshold: float = init_threshold
    eval_num_hist = []
    threshold_hist = []

    for i in range(iter_num):


        try:
            x, y, eval_num = sampling_grover_oracle(
                func, None, None, n_digits, dim, threshold, uniform=True, amplify_max=128, optimal_amplify_num=optimal_amplify_num)
        except TimeoutError:
            break

        eval_num_hist.append(eval_num)
        threshold = y
        threshold_hist.append(threshold)
        dist_target = np.linalg.norm(x-target)

        if verbose:
            print("-------")
            print("accepted", x)
            print("accepted_val", y)
            print("iter: ", i)
            print("threshold: ", threshold)
            print("eval_num: ", eval_num_hist[-1])
            print("--------")

        if dist_target < terminate_eps:
            break

    if verbose:
        print("total_eval_num: ", sum(eval_num_hist))
    return x, (threshold_hist, eval_num_hist)


if __name__ == "__main__":
    init_threshold = 1.
    init_mean, init_cov = np.array([0.5, 0.5]), np.array([[1, 0], [0, 1]])

    target = [0.2, 0.8]
    config = {
        "target": target,
        "dim": 2,
        "init_threshold": 1.,
        "n_digits": 8,
        "iter_num": 30,
        "terminate_eps": 0.001,
        "verbose": True
    }


    def func(x): return (20 + (10*(x[..., 0]-target[0])) ** 2 + (10*(x[..., 1]-target[1])) ** 2 - 10 * np.cos(
        2*np.pi*(10*(x[..., 0]-target[0]))) - 10 * np.cos(2*np.pi*(10*(x[..., 1]-target[1])))) / 40

    result_param, (param_hist, eval_num_hist) = grover_minimization(
        func, **config)
