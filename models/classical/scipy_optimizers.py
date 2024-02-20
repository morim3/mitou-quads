import numpy as np

from scipy.optimize import minimize, OptimizeResult

from scipy.optimize._optimize import _minimize_neldermead
from scipy.optimize import differential_evolution, basinhopping, dual_annealing


def run_scipy_optimizer(func, config, optimizer="differential_evolution", verbose=False, return_samples=False):

    x_init = config["init_mean"]

    eval_num_hist = []
    param_hist = [x_init]
    func_hist = []
    min_func_hist = []
    dist_target_hist = []

    if return_samples:
        samples_hist = []

    eval_num = 0

    def func_callback(xk, fval):
        nonlocal eval_num
        eval_num += 1
        return fval

    def iter_callback(xk):
        nonlocal eval_num
        eval_num_hist.append(eval_num)
        eval_num = 0
        func_hist.append(func(xk))
        min_func_hist.append(np.min(func_hist))
        dist_target_hist.append(np.linalg.norm(xk - config["target"]))
        if return_samples:
            samples_hist.append(xk)

        if verbose:
            print(f"""\
--- iter {len(eval_num_hist)} -----------------
x: {xk}
f: {func_hist[-1]}
----------------------------""")

        if dist_target_hist[-1] < config["terminate_eps"]:
            return True

    def wrapped_func(x): return func_callback(x, func(x))

    if optimizer == "differential_evolution":
        def callback(xk, convergence): return iter_callback(xk)
        x_final = differential_evolution(wrapped_func, bounds=[(0, 1)] * config["n_dim"],
                                         callback=callback,
                                         x0=x_init,
                                         )
    elif optimizer == "dual_annealing":
        def callback(xk, convergence, context): return iter_callback(xk)
        x_final = dual_annealing(wrapped_func, bounds=[(0, 1)] * config["n_dim"],
                                 callback=callback,
                                 x0=x_init,
                                 )
    elif optimizer == "nelder_mead":
        x_final = minimize(wrapped_func, x_init,
                           method="Nelder-Mead",
                           callback=iter_callback)
    elif optimizer == "lbfgs":
        x_final = minimize(wrapped_func, x_init,
                           method="L-BFGS-B",
                           callback=iter_callback, bounds=[(0, 1)] * config["n_dim"])
    elif optimizer == "basinhopping":

        class RandomDisplacementBounds(object):
            # random displacement with bounds:  see: https://stackoverflow.com/a/21967888/2320035
            # Modified! (dropped acceptance-rejection sampling for a more specialized approach)

            def __init__(self, xmin, xmax, stepsize=0.5):
                self.xmin = xmin
                self.xmax = xmax
                self.stepsize = stepsize

            def __call__(self, x):
                """take a random step but ensure the new position is within the bounds """
                min_step = np.maximum(self.xmin - x, -self.stepsize)
                max_step = np.minimum(self.xmax - x, self.stepsize)

                random_step = np.random.uniform(low=min_step, high=max_step, size=x.shape)
                xnew = x + random_step

                return xnew

        bounds = [(0, 1)] * config["n_dim"]
        bounded_step = RandomDisplacementBounds(np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds]))

        def callback(xk, convergence, context): return iter_callback(xk)
        x_final = basinhopping(wrapped_func, x_init, niter=config["max_iter"],
                               callback=callback, minimizer_kwargs={"method":"L-BFGS-B", "bounds": bounds}, take_step=bounded_step)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    return x_final, (np.array(min_func_hist), np.array(eval_num_hist), np.array(dist_target_hist), param_hist)


if __name__ == "__main__":
    init_threshold = 1.
    init_mean, init_cov = np.array([0.2, 0.8]), np.array([[1, 0], [0, 1]])
    init_step_size = 0.3
    target = np.array([0.8, 0.2])

    config = {
        "n_dim": 2,
        "max_iter": 1000,
        "n_samples": None,
        "terminate_step_size": 0.01,
        "terminate_eps": 0.01,
        "target": target,
        "init_mean": init_mean,
    }

    def func(x):
        return (20 + np.sum(
            100 * (x - target[None, :]) ** 2, axis=-1)) / 40

    result_param, (min_func_hist, eval_num_hist, dist_target_hist, param_hist) = run_scipy_optimizer(
        func, config, verbose=True, optimizer="differential_evolution")

    print(result_param, min_func_hist, eval_num_hist, dist_target_hist)
