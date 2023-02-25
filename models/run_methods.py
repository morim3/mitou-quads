import pickle
import numpy as np
from tqdm import tqdm
import wandb
from models.amp_sim import quads, grover_adaptive
from models.classical import cmaes
from models.parameters import QuadsParam, CMAParam
from utils import objective_functions

def get_objective_function(name, **kwargs):
    return objective_functions.__getattribute__(f"get_{name}")(**kwargs)

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
    wandb.log({"optimization process" : wandb.plot.line(table, "x", "y",
               title="Optimization Process")})


def run_quads(func, config):
    eval_hists = []
    min_func_hists = []
    dist_target_hists = []
    eval_total = []
    converged_to_global = []
    
    quads_param = QuadsParam(
        config["init_threshold"], CMAParam(config["init_mean"], config["init_cov"], config["init_step_size"] ))

    for trial in tqdm(range(config["n_trial"])):
        _, (min_func_hist, eval_num_hist, dist_target_hist, _) = quads.run_quads(func, quads_param, config, verbose=False)
        eval_num_hist = np.cumsum(eval_num_hist)
        eval_hists.append(eval_num_hist)
        min_func_hists.append(min_func_hist)
        dist_target_hists.append(dist_target_hist)
        eval_total.append(eval_num_hist[-1])
        converged_to_global.append(dist_target_hist[-1] < config["terminate_eps"])

    return {
        "eval_hists": eval_hists,
        "min_func_hists": min_func_hists,
        "dist_target_hists": dist_target_hists,
        "eval_total": eval_total,
        "converged_to_global": converged_to_global
    }

def run_grover(func, config):
    eval_hists = []
    min_func_hists = []
    dist_target_hists = []
    eval_total = []
    converged_to_global = []

    for trial in tqdm(range(config["n_trial"])):
        _, (min_func_hist, eval_num_hist, dist_target_hist) = grover_adaptive.run_grover_minimization(func, config, False)

        eval_num_hist = np.cumsum(eval_num_hist)
        min_func_hists.append(np.array(min_func_hist))
        eval_hists.append(np.array(eval_num_hist))
        dist_target_hists.append(np.array(dist_target_hist))
        eval_total.append(eval_num_hist[-1])
        converged_to_global.append(dist_target_hist[-1] < config["terminate_eps"])

    return {
        "eval_hists": eval_hists,
        "min_func_hists": min_func_hists,
        "dist_target_hists": dist_target_hists,
        "eval_total": eval_total,
        "converged_to_global": converged_to_global
    }

def run_cmaes(func, config):
    eval_hists = []
    min_func_hists = []
    dist_target_hists = []
    eval_total = []
    converged_to_global = []

    cmaes_param = CMAParam(config["init_mean"], config["init_cov"], config["init_step_size"])

    for trial in tqdm(range(config["n_trial"])):
        _, (min_func_hist, eval_num_hist, dist_target_hist, _) = cmaes.run_cmaes(func, cmaes_param, config)

        eval_num_hist = np.cumsum(eval_num_hist)
        eval_hists.append(eval_num_hist)
        min_func_hists.append(min_func_hist)
        dist_target_hists.append(dist_target_hist)
        eval_total.append(eval_num_hist[-1])
        converged_to_global.append(dist_target_hist[-1] < config["terminate_eps"])

    return {
        "eval_hists": eval_hists,
        "min_func_hists": min_func_hists,
        "dist_target_hists": dist_target_hists,
        "eval_total": eval_total,
        "converged_to_global": converged_to_global
    }

def log_function_shape(func_name, ):
    import matplotlib.pyplot as plt
    func, target = objective_functions.__getattribute__(f"get_{func_name}")(dim=2, )

    X, Y = np.meshgrid(np.linspace(0, 1, 500), np.linspace(0, 1, 500))
    grid = np.stack([X, Y], axis=-1).reshape((-1, 2))
    func_value = func(grid).reshape((500, 500))
    fig, ax = plt.subplots(dpi=100)
    ax.imshow(func_value)
    ax.set_title(f"{func_name} function")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.scatter([target[0]], [target[1]], marker='*', s=1)

    wandb.log({"func": fig})

def main(args):

    n_dim = args.n_dim

    func, target = get_objective_function(args.func, dim=n_dim)
    assert n_dim == target.shape[-1]

    init_mean = args.init_normal_mean
    if len(init_mean) == 1:
        init_mean = init_mean * n_dim
    init_mean = np.array(init_mean)
    assert n_dim == init_mean.shape[-1]

    init_cov = np.identity(n_dim) * args.init_normal_std
    init_threshold = func(init_mean)

    if args.method == "grover":
        n_samples = None
    elif args.method == "cmaes":
        n_samples = get_sample_size(n_dim)
    elif args.method == "quads":
        # good sample in cmaes is half better samples
        n_samples = int(get_sample_size(n_dim) / 2) + 1
    
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
        project=args.project_name,
        config=config,
        mode="disabled" if args.test else "online",
        tags=[args.func, args.method, args.sampler_type] 
    ) as wandb_run:
        
        artifact = wandb.Artifact("experiment-result", type="result")
        
        log_function_shape(args.func, )

        if args.method == "grover":
            result = run_grover(func, config)
        elif args.method == "cmaes":
            result = run_cmaes(func, config)
        elif args.method == "quads":
            result = run_quads(func, config)
        
        result = results_postprocess(result, config)
        wandb_log(result)

        with artifact.new_file(f"result.pickle", mode='wb') as f:
            pickle.dump(result, f)

        wandb.log_artifact(artifact)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--project_name", default="mitou-quads")
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
    parser.add_argument("--verbose", action="store_false")
    parser.add_argument("--n_trial", default=100, type=int)
    parser.add_argument("--terminate_step_size", default=0.01, type=np.float32)
    parser.add_argument("--terminate_eps", default=0.01, type=np.float32)
    parser.add_argument("--quantile", default=0.2, type=np.float32)
    parser.add_argument("--smoothing_th", default=0.5, type=np.float32)
    parser.add_argument("--optimal_amplify_num", default=False, type=bool)
    parser.add_argument("--eval_limit_one_sample", default=10000, type=int)
    parser.add_argument('--init_normal_mean', nargs='+', type=np.float32, default=[0.8])
    parser.add_argument('--init_normal_std', type=np.float32, default=1)
    parser.add_argument('--init_step_size', type=np.float32, default=0.5)
    args = parser.parse_args()
    
    main(args)

