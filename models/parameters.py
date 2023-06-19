from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CMAParam:
    mean: np.ndarray  # (n_dim, )
    cov: np.ndarray  # (n_dim, n_dim)
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
class QuadsParam:
    threshold: float
    cma_param: CMAParam 


@dataclass
class CMAHyperParam:
    # パラメータ名はtutrialに準拠
    n_dim: int
    n_samples: int
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

@dataclass
class QuadsHyperParam:
    quantile: float
    smoothing_th: float
    cma_hyperparam: CMAHyperParam



def update_quads_params(accepted, accepted_val, gen, param: QuadsParam, hp: QuadsHyperParam):
    threshold: float = hp.smoothing_th * param.threshold + \
        (1 - hp.smoothing_th) * np.quantile(accepted_val, q=hp.quantile)

    new_cma_param = update_cma_params(accepted, accepted_val, gen, param.cma_param, hp.cma_hyperparam)
    return QuadsParam(threshold, new_cma_param)

# ref: https://horomary.hatenablog.com/entry/2021/01/23/013508
def update_cma_params(accepted, accepted_val, gen, param: CMAParam, hp: CMAHyperParam, n_elite=None):
    # accepted: (n_samples, n_dim)
    # accepted_val: (n_samples,)

    n_elite = accepted.shape[0] if n_elite is None else n_elite

    sort_indices = np.argsort(accepted_val)
    accepted = accepted[sort_indices][:n_elite]

    diff_scaled = (accepted - param.mean) / param.step_size

    # w_mu: (n_samples,)

    E_normal = hp.E_normal

    mean = np.sum(accepted * hp.weights[:, None], axis=0)
    y = np.sum(diff_scaled * hp.weights[:, None], axis=0)

    diagD, B = np.linalg.eigh(param.cov)
    diagD = np.sqrt(diagD)
    inv_diagD = 1.0 / diagD

    C_ = np.matmul(np.matmul(B, np.diag(inv_diagD)), B.T)


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

    return CMAParam(
        mean, new_C, step_size, param.cov_path, step_path)

def get_normal_samples(cma_param: CMAParam, n_dim, n_samples, BD=None):
    n_sampled = 0
    sampled = []

    if BD is None:
        diagDD, B = np.linalg.eigh(cma_param.cov)
        diagD = np.sqrt(diagDD)
        BD = np.matmul(B, np.diag(diagD))

    while n_sampled < n_samples:
        Z = np.random.normal(0, 1, size=(n_dim, ))
        Y = np.matmul(BD, Z.T).T
        X = cma_param.mean + cma_param.step_size * Y
        if np.all(np.logical_and(0. <= X, X <= 1. )):
            sampled.append(X)
            n_sampled += 1
            
    return np.stack(sampled)
