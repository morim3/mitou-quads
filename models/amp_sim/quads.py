from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
from numpy.typing import NDArray

from sampler import initalize_normal_state, sampling_grover_oracle


@dataclass
class QuadsParam:
    mean: np.ndarray  # (n_dim, )
    cov: np.ndarray  # (n_dim, n_dim)
    threshold: float
    step_size: float
    cov_path: Optional[np.ndarray] = None
    th_path: float = 0.6
    step_path: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.step_path is None:
            self.step_path = np.zeros_like(self.mean)

        if self.cov_path is None:
            self.cov_path = np.zeros_like(self.mean)


@dataclass
class QuadsHyperParam:
    # パラメータ名はtutrialに準拠
    n_dim: int
    n_samples: int
    quantile: float
    smoothing_th: float
    c_sigma: float = None
    d_sigma: float = None
    c_c: float = None
    c_1: float = None
    c_mu: float = None

    def __post_init__(self):
        n_dim = self.n_dim

        weights = np.array([np.log(self.n_samples+1) - np.log(i)
                           for i in range(1, 1+self.n_samples)])
        self.weights = weights / weights.sum()

        self.mu_eff = 1. / (self.weights ** 2).sum()
        if self.c_sigma is None:
            self.c_sigma = (self.mu_eff + 2) / (n_dim + self.mu_eff + 5)

        if self.d_sigma is None:
            self.d_sigma = 1. + 2 * \
                max(0, np.sqrt((self.mu_eff-1.)/(n_dim + 1))-1) + self.c_sigma

        if self.c_c is None:
            self.c_c = (4 + self.mu_eff / n_dim) / \
                (n_dim + 4 + 2 * self.mu_eff / n_dim)

        if self.c_1 is None:
            self.c_1 = 2.0 / ((n_dim + 1.3)**2 + self.mu_eff)

        if self.c_mu is None:
            self.c_mu = min(1-self.c_1,
                            2.0 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((n_dim + 2) ** 2 + self.mu_eff))

        self.E_normal = np.sqrt(n_dim) * \
            (1 - 1/(4*n_dim) + 1/(21*n_dim**2))


# ref: https://horomary.hatenablog.com/entry/2021/01/23/013508
def update_params(accepted, accepted_val, gen, param: QuadsParam, hp: QuadsHyperParam):
    # accepted: (n_samples, n_dim)
    # accepted_val: (n_samples,)

    sort_indices = np.argsort(accepted_val)
    accepted = accepted[sort_indices]

    diff_scaled = (accepted - param.mean) / param.step_size

    # w_mu: (n_samples,)

    E_normal = hp.E_normal
    smoothing_th = hp.smoothing_th

    mean = np.sum(accepted * hp.weights[:, None], axis=0)
    y = np.sum(diff_scaled * hp.weights[:, None], axis=0)

    diagD, B = np.linalg.eigh(param.cov)
    diagD = np.sqrt(diagD)
    inv_diagD = 1.0 / diagD

    C_ = np.matmul(np.matmul(B, np.diag(inv_diagD)), B.T)

    threshold: float = smoothing_th * param.threshold + \
        (1 - smoothing_th) * np.quantile(accepted_val, q=hp.quantile)

    step_path = (1-hp.c_sigma) * param.step_path + \
        np.sqrt(hp.c_sigma*(2-hp.c_sigma)*hp.mu_eff) * C_ @ y
    step_size = param.step_size * np.exp(hp.c_sigma/hp.d_sigma * (
        np.linalg.norm(step_path) / E_normal - 1))

    """ Covariance matrix adaptation (CMA) """
    left = np.sqrt((param.step_path ** 2).sum()) / \
        np.sqrt(1 - (1 - hp.c_sigma) ** (2 * (gen+1)))
    right = (1.4 + 2 / (hp.n_dim + 1)) * E_normal
    hsigma = 1 if left < right else 0
    d_hsigma = (1 - hsigma) * hp.c_c * (2 - hp.c_c)

    #: p_cの更新
    new_cov_path = (1 - hp.c_c) * param.cov_path
    new_cov_path += hsigma * \
        np.sqrt(hp.c_c * (2 - hp.c_c) * hp.mu_eff) * y
    param.cov_path = new_cov_path

    #: Cの更新
    new_C = (1 + hp.c_1 * d_hsigma - hp.c_1 - hp.c_mu) * param.cov
    new_C += hp.c_1 * np.outer(param.cov_path, param.cov_path)

    new_C += hp.c_mu * np.sum(hp.weights[:, None, None] *
                              diff_scaled[:, None] * diff_scaled[:, :, None], axis=0)

    return QuadsParam(
        mean, new_C, threshold,
        step_size, param.cov_path, step_path)


def get_cmaes_samples(func, quads_param, config):
    accepted = []
    accepted_val = []
    mean = quads_param.mean
    cov = quads_param.cov * quads_param.step_size ** 2
    threshold = quads_param.threshold

    n_eval = 0
    for _ in range(config["sample_num"]):
        initial_state = initalize_normal_state(
            config["n_digits"], mean, cov, config["dim"])
        x, y, eval_num = sampling_grover_oracle(
            func, mean, cov, config["n_digits"], config["dim"], threshold, optimal_amplify_num=config["optimal_amplify_num"], initial_state=initial_state)

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

    quads_param = init_param
    eval_num_hist = []
    param_hist = []
    min_func_hist = []
    dist_target_hist = []

    min_val = np.inf

    for i in range(config["iter_num"]):
        # グローバー探索によりしきい値未満のサンプルを得る
        try:
            accepted, accepted_val, n_eval = get_cmaes_samples(
                func, quads_param, config)
        except TimeoutError:
            break

        # パラメーター更新
        quads_param = update_params(
            accepted, accepted_val, i, quads_param, config["hp"])

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
            print("updated mu: ", quads_param.mean)
            print("cov: ", quads_param.cov)
            print("step_size: ", quads_param.step_size)
            print("threshold: ", quads_param.threshold)
            print("eval_num: ", eval_num_hist[-1])
            print("--------")

        if dist_target < config["terminate_eps"] or quads_param.step_size < config["terminate_step_size"]:
            break

    print("total_eval_num: ", sum(eval_num_hist))
    return quads_param, (param_hist, np.array(eval_num_hist), np.array(min_func_hist))


if __name__ == "__main__":
    init_threshold = 1.
    init_mean, init_cov = np.array([0.5, 0.5]), np.array([[1, 0], [0, 1]])
    init_step_size = 0.1
    quads_param = QuadsParam(
        init_mean, init_cov, init_threshold, init_step_size)
    target = np.array([0.8, 0.2])

    config = {
        "dim": 2,
        "n_digits": 8,
        "iter_num": 30,
        "sample_num": 3,
        "terminate_step_size": 0.001,
        "terminate_eps": 0.001,
        "optimal_amplify_num": False,
        "hp": QuadsHyperParam(n_dim=2, n_samples=3, quantile=0.1, smoothing_th=0.5),
        "target": target
    }

    def func(x):
        return (20 + np.sum(
            100 * (x - target[None, :]) ** 2 -
            10 * np.cos(2 * np.pi * 10 * (x - target[None, :])), axis=-1)) / 40

    result_param, (param_hist, eval_num_hist, min_func_hist) = run_quads(
        func, quads_param, config, verbose=True)
