
from typing import Callable, List

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from models.amp_sim.sampler import optimal_amplify_num, GroverSampler

def run_adam(func, config, verbose=False):

    threshold: float = config["init_threshold"]
    beta1: float = config["beta1"]
    beta2: float = config["beta2"]
    lr: float = config["init_step_size"]
    eps: float = 1e-12
    
    eval_num_hist = []
    threshold_hist = []
    dist_target_hist = []

    dim = config["n_dim"]
    target = config["target"]

    x = jnp.array(config["init_mean"])
    m = jnp.zeros(dim)
    v = jnp.zeros(dim)

    for i in range(config["max_iter"]):

        eval_num = 2

        grad_fn = jax.grad(lambda z : jnp.sum(func(z)), argnums=[0])

        g = grad_fn(x)[0]
        v = beta2 * v + (1 - beta2) * (g ** 2)
        m = beta1 * m + (1 - beta1) * g
        x = x - lr * (jnp.sqrt(1 - beta2 ** (i + 1) ) / (1 - beta1 ** (i + 1))) * (m / (jnp.sqrt(v) + eps))
        y = jnp.sum(func(x))

        eval_num_hist.append(eval_num)
        threshold = np.array(y)
        threshold_hist.append(threshold)
        dist_target = np.array(jnp.linalg.norm(x-target))
        dist_target_hist.append(dist_target)

        if jnp.mean(jnp.abs(g)) < config["terminate_eps"]:
            break

    if verbose:
        print(f"""\
--- iter {i} -----------------
grad: {g}, 
accepted: {x}
accepted_val: {y}
threshold: {threshold}
total_eval_num: {sum(eval_num_hist)}
----------------------------""")
    return x, (np.array(threshold_hist), np.array(eval_num_hist), np.array(dist_target_hist), np.array(threshold_hist))
