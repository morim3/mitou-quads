import numpy as np

from scipy.optimize import minimize, OptimizeResult

from scipy.optimize._optimize import _minimize_neldermead

def run_nelder_mead(func, config, verbose=False, return_samples=False):

    x_init = config["init_mean"]

    # n_elite = int(np.floor(config["n_samples"] / 2))
    # hp = CMAHyperParam(config["n_dim"], n_elite, )

    # cma_param = init_param
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
            raise StopIteration
    
    wrapped_func = lambda x: func_callback(x, func(x))

    x_final = minimize(wrapped_func, x_init,
                       method="Nelder-Mead",
                       options={"maxiter": config["max_iter"]},
                       callback=iter_callback)

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
        "terminate_step_size": 0.001,
        "terminate_eps": 0.001,
        "target": target,
        "init_mean": init_mean,
    }

    def func(x):
        return (20 + np.sum(
            100 * (x - target[None, :]) ** 2, axis=-1)) / 40

    result_param, (min_func_hist, eval_num_hist, dist_target_hist, param_hist) = run_nelder_mead(
        func, config, verbose=True)
    
    print(result_param, min_func_hist, eval_num_hist, dist_target_hist)
