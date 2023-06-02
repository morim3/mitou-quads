from typing import Callable, Optional
import numpy as np
from numpy import random
from numpy.typing import NDArray
import jax.numpy as jnp
import jax
from jax import jit
from jax.scipy import stats
import bisect

def choice(options,probs):
    x = np.random.rand()
    cumsum = np.cumsum(probs)
    ind = bisect.bisect(cumsum, x*cumsum[-1])
    return options[ind]

def get_grid_point(n_bits, range_min, range_max, dim):
    x = [np.linspace(range_min, range_max,
                     2 ** n_bits, endpoint=False, dtype=np.float32)
                     for _ in range(dim)]
    grid = np.stack(
        np.meshgrid(*x, indexing="ij"), axis=-1).reshape(-1, dim)
    return grid

def discrete_normal(n_bits, mean, cov, dim, range_min=0, range_max=1):
    grid = get_grid_point(n_bits, range_min, range_max, dim)
    pdf = stats.multivariate_normal.pdf(grid, mean=mean, cov=cov)
    return pdf / np.sum(pdf)

def init_normal_state(n_digits: int, mu: NDArray, cov: NDArray, dim: int):
    distribution = discrete_normal(n_digits, mu, cov, dim, range_min=0, range_max=1)
    return jnp.sqrt(distribution)

def init_uniform_state(n_digits: int, dim: int):
    distribution = jnp.ones(2**(n_digits*dim), dtype=np.float32) / 2 ** (n_digits * dim)
    return jnp.sqrt(distribution)

class ReflectionGate:
    def __init__(self, reflection_base):
        self.reflection_base = reflection_base

    def forward(self, target):
        inner = target @ self.reflection_base
        reflected = 2 * inner * self.reflection_base - target
        return reflected
    

class DiagonalOracle:
    def __init__(self, test_val: NDArray):
        self.test_val = test_val
    # O(N)
    def forward(self, state_vector: np.ndarray):
        return self.test_val * state_vector

    def inverse(self, state_vector: np.ndarray):
        return self.forward(state_vector)

#
# class ControlZ:
#     def __init__(self, N: int):
#         self.v = jnp.ones(N)
#         self.v[0] = -1
#
#     def forward(self, state_vector: np.ndarray):
#         return state_vector * self.v
#     
#     def inverse(self, state_vector: np.ndarray):
#         return self.forward(state_vector)

def regularize(state_vector: np.ndarray):
    return state_vector / jnp.linalg.norm(state_vector)

def calc_acception_rate(p, func_val, threshold):
    accept_flag = func_val < threshold
    acception_rate = np.sum(p[accept_flag])
    accept_num = np.sum(accept_flag)
    return acception_rate, accept_num

def optimal_amplify_num(p):
    sqrt_rate = np.sqrt(p)
    return int(np.arccos(sqrt_rate) / (2 * np.arcsin(sqrt_rate)))

class GroverSampler:
    def __init__(self,
                 func: Callable[[NDArray], NDArray],
                 n_digits: int,
                 dim: int):
        self.func = func
        self.n_digits = n_digits
        self.dim = dim
        test_points = get_grid_point(n_digits, 0, 1, dim)
        self.func_val = func(test_points)
        
    def sample(self,
               mu: Optional[NDArray],
               cov: Optional[NDArray],
               threshold: float,
               n_samples: int,
               uniform=False,
               amplify_max=128,
               amplify_num_accel_rate=6/5,
               use_optimal_amplify=False,
               oracle_eval_limit: int = 10000,
               initial_state=None,
               verbose=False):
        
        n_digits = self.n_digits
        dim = self.dim
        N = pow(2, n_digits * dim)
        
        if initial_state is None:
            if not uniform:
                if mu is None or cov is None:
                    raise ValueError
                else:
                    initial_state = init_normal_state(n_digits, mu, cov, dim) 
            else:
                initial_state = init_uniform_state(n_digits, dim)

        oracle = DiagonalOracle(np.where(self.func_val < threshold, -1, 1))
        amplify = [jit(oracle.forward), jit(ReflectionGate(initial_state).forward), jit(regularize)]

        amplify_num = 0
        oracle_eval_num = 0

        xs = []
        ys = []

        state_vector = jnp.array(initial_state)
        now_amplify = 0

        while len(ys) < n_samples:
            if use_optimal_amplify:
                p = np.abs(state_vector) ** 2
                p = p / np.sum(p)
                acception_rate, _ = calc_acception_rate(p, self.func_val, threshold)
                actual_amplify_num = optimal_amplify_num(acception_rate)
            else:
                actual_amplify_num = random.randint(0, int(amplify_num)+1) if amplify_num >= 1 else 0 
            
            if verbose:
                print("amp_num", amplify_num, "actual", actual_amplify_num)


            if now_amplify < actual_amplify_num:
                circuit = [] + amplify * (actual_amplify_num - now_amplify)
                now_amplify = actual_amplify_num
            else:
                circuit = [] + amplify * actual_amplify_num
                state_vector = initial_state
                now_amplify = actual_amplify_num

            for i, gate in enumerate(circuit):
                state_vector = gate(state_vector)

            oracle_eval_num += actual_amplify_num

            p = np.abs(state_vector) ** 2
            # p = p / jnp.sum(p) regularize in choice
            if verbose: 
                acception_rate, accept_num = calc_acception_rate(p, self.func_val, threshold)
                print("acception_rate", acception_rate, accept_num)

            x = format(choice(np.arange(N), p), "0" + str(dim * n_digits) + "b")
            x = np.array([int(x[n_digits * i:n_digits * i + n_digits], 2) / 2 ** n_digits  for i in range(dim)])

            if amplify_num == 0:
                amplify_num = 1
            else:
                amplify_num = min(amplify_num * amplify_num_accel_rate, amplify_max)

            y = self.func(x)
            oracle_eval_num += 1
                
            if y < threshold:
                xs.append(x)
                ys.append(y)
                amplify_num = 0
                now_amplify = 0

            if oracle_eval_limit is not None and oracle_eval_limit < oracle_eval_num:
                raise TimeoutError("Oracle evaluate limit exceeded.")
        
        xs = np.array(xs)
        ys = np.array(ys)
        return xs, ys, oracle_eval_num
