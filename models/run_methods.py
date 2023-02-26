import pickle
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
import joblib
import wandb
from models.amp_sim import quads, grover_adaptive
from models.classical import cmaes
from models.parameters import QuadsParam, CMAParam
from utils.objective_functions import objective_functions
from utils.plot_tools import plot_function_surface
from typing import Callable
import os

def get_sample_size(dim):
    return int(4+np.log(dim)*3)

def results_postprocess(result, config):

    eval_total = result["eval_total"]
    converged_to_global = result["converged_to_global"]

    eval_total = np.array(eval_total)
    converged_to_global = np.array(converged_to_global)

    success_rate = np.mean(converged_to_global.astype(float))

    if success_rate > 0:
        mean_eval_success = np.mean(eval_total[converged_to_global])
        std_eval_success = np.std(eval_total[converged_to_global]) 
    else:
        mean_eval_success = None
        std_eval_success = None

    if success_rate < 1:
        mean_eval_failure = np.mean(eval_total[np.logical_not(converged_to_global)])
        std_eval_failure = np.std(eval_total[np.logical_not(converged_to_global)])
    else:
        mean_eval_failure = None
        std_eval_failure = None

    if success_rate == 0:
        mean_eval_to_global = None
    elif success_rate == 1:
        mean_eval_to_global = mean_eval_success
    else:
        mean_eval_to_global = mean_eval_failure * (1-success_rate) / success_rate + mean_eval_success

    result.update({
        "config": config,
        "success_rate": success_rate,
        "mean_eval_success": mean_eval_success,
        "std_eval_success": std_eval_success,
        "mean_eval_failure": mean_eval_failure,
        "std_eval_failure": std_eval_failure,
        "mean_eval_to_global": mean_eval_to_global,
    })
    return result


def wandb_log(result):

    # for trial in range(len(eval_hists)):
    #     wandb.log({
    #         f"eval_num_{trial}": eval_hists[trial],
    #         f"objective_func_{trial}": min_func_hists[trial],
    #         f"dist_target_{trial}": dist_target_hists[trial],
    #         "trial": trial
    #     })


    wandb.log({
                  f"mean_eval_success": result["mean_eval_success"],
                  f"std_eval_success": result["std_eval_success"],
                  f"mean_eval_failure": result["mean_eval_failure"],
                  f"std_eval_failure": result["std_eval_failure"],
                  f"converged_rate": result["success_rate"],
                  f"mean_eval_to_global": result["mean_eval_to_global"],
                  f"eval_total": result["eval_total"], 
                  f"converged_to_global": result["converged_to_global"]
              })

    eval_hists = result["eval_hists"]
    min_func_hists = result["min_func_hists"]

    opt_process = [[i, x, y] for i in range(len(eval_hists)) for (x, y) in zip(eval_hists[i], min_func_hists[i]) ]
    table = wandb.Table(data=opt_process, columns = ["trial", "x", "y"])
    wandb.log({"optimization process" : table})

def run_trials(func, config):
    eval_hists = []
    min_func_hists = []
    dist_target_hists = []
    eval_total = []
    converged_to_global = []

    if args.method == "grover":
        method = grover_adaptive.run_grover_minimization
    elif args.method == "cmaes":
        method = cmaes.run_cmaes
    elif args.method == "quads":
        method = quads.run_quads
    
    def run_trial(func, method, config):
        _, (min_func_hist, eval_num_hist, dist_target_hist, param_hists) = method(func, config, verbose=config["verbose"])
        eval_num_hist = np.cumsum(eval_num_hist)
        eval_total = eval_num_hist[-1]
        converged_to_global = dist_target_hist[-1] < config["terminate_eps"]
        return min_func_hist, eval_num_hist, dist_target_hist, eval_total, converged_to_global

    min_func_hists, eval_hists, dist_target_hists, eval_total, converged_to_global = zip(
        *joblib.Parallel(n_jobs=config["n_jobs"], prefer="threads", verbose=10)(
            joblib.delayed(run_trial)(func, method, config) for _ in range(config["n_trial"]))
    )

    return {
        "eval_hists": eval_hists,
        "min_func_hists": min_func_hists,
        "dist_target_hists": dist_target_hists,
        "eval_total": eval_total,
        "converged_to_global": converged_to_global
    }

def main(args):

    n_dim = args.n_dim

    func, target = objective_functions[args.func](dim=n_dim)
    assert n_dim == target.shape[-1]

    init_mean = np.random.rand(n_dim)
    assert n_dim == init_mean.shape[-1]

    init_cov = np.identity(n_dim) * args.init_normal_std
    init_threshold = func(init_mean)

    if args.method == "grover":
        n_samples = None
    elif args.method == "cmaes":
        n_samples = get_sample_size(n_dim)
    elif args.method == "quads":
        # good sample in cmaes is half better samples
        n_samples = int(get_sample_size(n_dim) / 2 + 1)
    
    config = vars(args)

    config.update({
        "n_samples": n_samples,
        "target": target,
        "init_mean": init_mean,
        "init_cov": init_cov,
        "init_threshold": init_threshold,
    })

    print(f"config: {config}")
 
    with wandb.init(
        entity=args.entity,
        project=args.project_name,
        name=args.name,
        group=args.group,
        config=config,
        mode="disabled" if args.test else "online",
        tags=[args.func, args.method, args.sampler_type] 
    ) as wandb_run:
        
        wandb.log({"func": plot_function_surface(*objective_functions[args.func](dim=2), func_name=args.func, init_mu=init_mean)})
        result = run_trials(func, config)
        result = results_postprocess(result, config)
        wandb_log(result)

        save_path = os.path.join(wandb.run.dir, "result.pickle")
        with open(save_path, mode='wb') as f:
            pickle.dump(result, f)

        wandb.save(save_path)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--project_name")
    parser.add_argument("--name", default=None)
    parser.add_argument("--group", default=None)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--func", default="rastrigin", help="test function to optimize")
    parser.add_argument("--n_dim", default=3, type=int, help="number of dimension")
    parser.add_argument("--method", default="quads", choices=["grover", "cmaes", "quads"], help="method used in optimization")
    parser.add_argument("--sampler_type", default="quantum", choices=["quantum", "classical"],
                        help="type of sampler (quantum: sample by quantum simulator, classical: sample by classical algorithm)")
    parser.add_argument("--n_digits", default=8, type=int,
                        help="number of digits quantizing the function space")
    parser.add_argument("--max_iter", default=100, type=int,
                        help="maximum number of optimization iterations")
    parser.add_argument("--test", action="store_true",
                        help="Run in smoke-test mode")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--n_trial", default=100, type=int)
    parser.add_argument("--n_jobs", default=10, type=int)
    parser.add_argument("--terminate_step_size", default=0.01, type=np.float32)
    parser.add_argument("--terminate_eps", default=0.01, type=np.float32)
    parser.add_argument("--quantile", default=0.2, type=np.float32)
    parser.add_argument("--smoothing_th", default=0.5, type=np.float32)
    parser.add_argument("--use_optimal_amplify", default=False, type=bool)
    parser.add_argument("--eval_limit_per_update", default=10000, type=int)
    parser.add_argument('--init_normal_std', type=np.float32, default=1)
    parser.add_argument('--init_step_size', type=np.float32, default=0.5)
    args = parser.parse_args()
    
    main(args)

