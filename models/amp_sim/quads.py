from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
from numpy.typing import NDArray

from sampler import initalize_normal_state, sampling_grover_oracle
from models.parameters import CMAHyperParam, CMAParam, QuadsParam, QuadsHyperParam, update_quads_params


def get_samples(func, quads_param:QuadsParam, config):
    accepted = []
    accepted_val = []
    mean = quads_param.cma_param.mean
    cov = quads_param.cma_param.cov * quads_param.cma_param.step_size ** 2
    threshold = quads_param.threshold

    n_eval = 0
    for _ in range(config["n_samples"]):
        initial_state = initalize_normal_state(
            config["n_digits"], mean, cov, config["n_dim"])
        x, y, eval_num = sampling_grover_oracle(
            func, mean, cov, config["n_digits"], config["n_dim"], threshold, optimal_amplify_num=config["optimal_amplify_num"], initial_state=initial_state)

        n_eval += eval_num

        accepted.append(x)
        accepted_val.append(y)

    return np.array(accepted), np.concatenate(accepted_val), n_eval


def run_quads(
    func: Callable[[NDArray], NDArray],
    init_param: QuadsParam,
    config,
    verbose=False
):

    hp = QuadsHyperParam(quantile=config["quantile"], smoothing_th=config["smoothing_th"], 
                         cma_hyperparam=CMAHyperParam(config["n_dim"], config["n_samples"], config["smoothing_th"]))

    quads_param = init_param
    eval_num_hist = []
    param_hist = []
    min_func_hist = []
    dist_target_hist = []

    min_val = np.inf

    for i in range(config["iter_num"]):
        # グローバー探索によりしきい値未満のサンプルを得る
        try:
            accepted, accepted_val, n_eval = get_samples(
                func, quads_param, config)
        except TimeoutError:
            break

        # パラメーター更新
        quads_param = update_quads_params(
            accepted, accepted_val, i, quads_param, hp)

        # 履歴の保存
        min_val = min(min_val, np.min(accepted_val))
        dist_target = np.min(np.linalg.norm(
            accepted - config["target"][None], axis=1))
        param_hist.append(quads_param)
        min_func_hist.append(min_val)
        dist_target_hist.append(dist_target)
        eval_num_hist.append(n_eval)

        if verbose:
            print("-------")
            print("accepted", accepted)
            print("accepted_val", accepted_val)
            print("iter: ", i)
            print("updated mu: ", quads_param.cma_param.mean)
            print("cov: ", quads_param.cma_param.cov)
            print("step_size: ", quads_param.cma_param.step_size)
            print("threshold: ", quads_param.threshold)
            print("eval_num: ", eval_num_hist[-1])
            print("--------")

        if dist_target < config["terminate_eps"] or quads_param.cma_param.step_size < config["terminate_step_size"]:
            break

    print("total_eval_num: ", sum(eval_num_hist))
    return quads_param, (param_hist, np.array(eval_num_hist), np.array(min_func_hist))


if __name__ == "__main__":
    init_threshold = 1.
    init_mean, init_cov = np.array([0.2, 0.8]), np.array([[1, 0], [0, 1]])
    init_step_size = 1.
    quads_param = QuadsParam(
        init_threshold, CMAParam(init_mean, init_cov, init_step_size ))
    target = np.array([0.8, 0.2])

    config = {
        "n_dim": 2,
        "n_digits": 8,
        "iter_num": 30,
        "n_samples": 3,
        "terminate_step_size": 0.001,
        "terminate_eps": 0.001,
        "optimal_amplify_num": False,
        "quantile": 0.1,
        "smoothing_th": 0.5,
        "target": target
    }

    def func(x):
        return (20 + np.sum(
            100 * (x - target[None, :]) ** 2 -
            10 * np.cos(2 * np.pi * 10 * (x - target[None, :])), axis=-1)) / 40

    result_param, (param_hist, eval_num_hist, min_func_hist) = run_quads(
        func, quads_param, config, verbose=True)
