import numpy as np
from models.parameters import CMAParam, CMAHyperParam, update_cma_params, get_normal_samples


def run_cmaes(func, init_cma_param: CMAParam, config, verbose=False):

    cma_param = init_cma_param

    n_elite = int(np.floor(config["n_samples"] / 2))
    hp = CMAHyperParam(config["n_dim"], n_elite, )


    eval_num_hist = []
    param_hist = [init_cma_param]
    min_func_hist = []
    dist_target_hist = []

    min_val = np.inf

    for gen in range(config["iter_num"]):
        sampled = get_normal_samples(cma_param, config["n_dim"], config["n_samples"])
        fitnesses = func(sampled)

        min_fitness = np.min(fitnesses)
        min_val = min(min_val, min_fitness)

        cma_param = update_cma_params(sampled, fitnesses, gen, cma_param, hp, n_elite)  

        dist_target = np.min(np.linalg.norm(
            sampled - config["target"][None], axis=1))

        eval_num_hist.append(fitnesses.shape[0])
        min_func_hist.append(min_val)
        param_hist.append(cma_param)
        dist_target_hist.append(dist_target)


        if verbose:
            print("-------")
            print("iter", gen)
            print("mean: ", cma_param.mean)
            print("cov: ", cma_param.cov)
            print("step_size: ", cma_param.step_size)
            print("eval_num: ", eval_num_hist[-1])
            print("--------")

        if dist_target < config["terminate_eps"] or cma_param.step_size < config["terminate_step_size"]:
            break

    print("total_eval_num: ", sum(eval_num_hist))
    return cma_param, (np.array(min_func_hist), np.array(eval_num_hist), np.array(dist_target_hist), param_hist)

if __name__ == "__main__":
    init_threshold = 1.
    init_mean, init_cov = np.array([0.2, 0.8]), np.array([[1, 0], [0, 1]])
    init_step_size = 0.3
    quads_param = CMAParam(init_mean, init_cov, init_step_size )
    target = np.array([0.8, 0.2])

    config = {
        "n_dim": 2,
        "iter_num": 1000,
        "n_samples": 7,
        "terminate_step_size": 0.001,
        "terminate_eps": 0.001,
        "target": target
    }

    def func(x):
        return (20 + np.sum(
            100 * (x - target[None, :]) ** 2 -
            10 * np.cos(2 * np.pi * 10 * (x - target[None, :])), axis=-1)) / 40

    result_param, (param_hist, eval_num_hist, min_func_hist) = run_cmaes(
        func, quads_param, config, verbose=True)
