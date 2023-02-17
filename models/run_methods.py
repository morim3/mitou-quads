import numpy as np
from tqdm import tqdm
from models.amp_sim import quads, grover_adaptive
from models.classical import cmaes
from models.parameters import QuadsParam, QuadsHyperParam, CMAParam, CMAHyperParam
from utils import objective_functions
import wandb
import json


def get_sample_size(dim):
    return int(4+np.log(dim)*3)

def wandb_log(eval_hists, min_func_hists, dist_target_hists, eval_total, converged_to_global):

    for trial in range(len(eval_hists)):
        wandb.log({
            f"eval_num_{trial}": eval_hists[trial],
            f"objective_func_{trial}": min_func_hists[trial],
            f"dist_target_{trial}": dist_target_hists[trial],
            "trial": trial
        })

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

    if success_rate == 1:
        mean_eval_to_global = mean_eval_success
    elif success_rate == 0:
        mean_eval_to_global = None
    else:
        mean_eval_to_global = mean_eval_failure / success_rate + mean_eval_success

    wandb.log({
                  f"mean_eval_success": mean_eval_success,
                  f"std_eval_success": std_eval_success,
                  f"mean_eval_failure": mean_eval_failure,
                  f"std_eval_failure": std_eval_failure,
                  f"converged_rate": success_rate,
                  f"mean_eval_to_global": mean_eval_to_global
              })

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

    for trial in tqdm(range(config["trial_num"])):
        _, (min_func_hist, eval_num_hist, dist_target_hist, _) = quads.run_quads(func, quads_param, config, verbose=False)
        eval_num_hist = np.cumsum(eval_num_hist)
        eval_hists.append(eval_num_hist)
        min_func_hists.append(min_func_hist)
        dist_target_hists.append(dist_target_hist)
        eval_total.append(eval_num_hist[-1])
        converged_to_global.append(dist_target_hist[-1] < config["terminate_eps"])


    wandb_log(eval_hists, min_func_hists, dist_target_hists, eval_total, converged_to_global)

    return eval_hists, min_func_hists, dist_target_hists, eval_total, converged_to_global

def run_grover(func, config, ):
    min_func_hists = []
    eval_hists = []
    dist_target_hists = []
    eval_total = []
    converged_to_global = []

    for trial in tqdm(range(config["trial_num"])):
        _, (min_func_hist, eval_num_hist, dist_target_hist, _) = grover_adaptive.run_grover_minimization(func, config, False)

        eval_num_hist = np.cumsum(eval_num_hist)
        min_func_hists.append(np.array(min_func_hist))
        eval_hists.append(np.array(eval_num_hist))
        dist_target_hists.append(np.array(dist_target_hist))
        eval_total.append(eval_num_hist[-1])
        converged_to_global.append(dist_target_hist[-1] < config["terminate_eps"])

    wandb_log(eval_hists, min_func_hists, dist_target_hists, eval_total, converged_to_global)

    return eval_hists, min_func_hists, dist_target_hists, eval_total, converged_to_global

def run_cmaes(func, config):
    eval_hists = []
    min_func_hists = []
    dist_target_hists = []
    eval_total = []
    converged_to_global = []

    cmaes_param = CMAParam(config["init_mean"], config["init_cov"], config["init_step_size"])

    for trial in tqdm(range(config["trial_num"])):
        _, (min_func_hist, eval_num_hist, dist_target_hist, _) = cmaes.run_cmaes(func, cmaes_param, config)

        eval_num_hist = np.cumsum(eval_num_hist)
        eval_hists.append(eval_num_hist)
        min_func_hists.append(min_func_hist)
        dist_target_hists.append(dist_target_hist)
        eval_total.append(eval_num_hist[-1])
        converged_to_global.append(dist_target_hist[-1] < config["terminate_eps"])


    wandb_log(eval_hists, min_func_hists, dist_target_hists, eval_total, converged_to_global)

    return eval_hists, min_func_hists, dist_target_hists, eval_total, converged_to_global

def log_func(func_name, other_param):
    import matplotlib.pyplot as plt
    if "target" in other_param:
        func, target = objective_functions.__getattribute__(f"get_{func_name}")(dim=2, target=target)
    else:
        func, target = objective_functions.__getattribute__(f"get_{func_name}")(dim=2, )

    X, Y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    grid = np.stack([X, Y], axis=-1).reshape((-1, 2))
    func_value = func(grid).reshape((100, 100))
    fig, ax = plt.subplots()
    ax.contour(X, Y, func_value)
    ax.set_title("two dimensional function")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    wandb.log({"func": fig})



def main(args):


    other_param = json.loads(args.other_param)
    func, target = objective_functions.__getattribute__(f"get_{args.func}")(**other_param)

    log_func(args.func, args.other_param)

    n_dim = target.shape[-1]
    

    init_normal_mean = np.array(args.init_normal_mean)
    init_threshold = func(args.init_normal_mean)
    init_cov = np.identity(n_dim) * args.init_normal_std

    if n_dim != init_normal_mean.shape[-1]:
        raise TypeError("args.init_normal mean != n_dim of func")

    if args.method == "cmaes":
        n_samples = get_sample_size(n_dim)
    elif args.method == "quads":
        # good sample in cmaes is half better samples
        n_samples = int(get_sample_size(n_dim) / 2) + 1
    else:
        n_samples = None

    config = {
        "sampler_type": args.sampler_type,
        "method": args.method,
        "n_dim": n_dim,
        "n_digits": args.n_digits,
        "iter_num": args.iter_num,
        "trial_num": args.trial_num,
        "n_samples": n_samples,
        "terminate_step_size": args.terminate_step_size,
        "terminate_eps": args.terminate_eps,
        "optimal_amplify_num": args.optimal_amplify_num,
        "quantile": args.quantile,
        "smoothing_th": args.smoothing_th,
        "target": target,
        "eval_limit_one_sample": args.eval_limit_one_sample,
        "init_mean": init_normal_mean,
        "init_cov": init_cov,
        "init_threshold": init_threshold,
        "init_step_size": args.init_step_size,
    }
 
    wandb.init(
        project="mitou-quads",
        config=config,
        name=args.name,
        notes = args.notes
    )


    if args.method == "cmaes":
        run_cmaes(func, config)
    elif args.method == "grover":
        run_grover(func, config)
    elif args.method == "quads":
        run_quads(func, config)
    else:
        raise NotImplementedError


    wandb.finish()




if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--name", default="default")
    parser.add_argument("--notes", default="")
    parser.add_argument("--func", default="rastrigin")
    parser.add_argument("--other_param", default="{}")
    parser.add_argument("--method", default="quads")
    parser.add_argument("--sampler_type", default="quantum")
    parser.add_argument("--n_digits", default=8, type=int)
    parser.add_argument("--iter_num", default=100, type=int)
    parser.add_argument("--trial_num", default=100, type=int)
    parser.add_argument("--terminate_step_size", default=0.001, type=np.float32)
    parser.add_argument("--terminate_eps", default=0.01, type=np.float32)
    parser.add_argument("--quantile", default=0.2, type=np.float32)
    parser.add_argument("--smoothing_th", default=0.5, type=np.float32)
    parser.add_argument("--optimal_amplify_num", default=False, type=bool)
    parser.add_argument("--eval_limit_one_sample", default=10000, type=int)
    parser.add_argument('--init_normal_mean', nargs='+', type=np.float32)
    parser.add_argument('--init_normal_std', type=np.float32, default=1)
    parser.add_argument('--init_step_size', type=np.float32, default=0.5)
    args = parser.parse_args()
    main(args)




