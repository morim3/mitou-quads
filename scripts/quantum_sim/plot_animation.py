from models.amp_sim import quads, grover_adaptive
from models.classical import cmaes
from models.parameters import QuadsParam, CMAParam
import matplotlib.pyplot as plt
import numpy as np
from utils.objective_functions import objective_functions
from models.run_methods import get_sample_size, parse_args
import os
from scipy.stats import multivariate_normal
from typing import List
import matplotlib.cm as cm
import matplotlib.patches as pat
from utils.mplsetting import get_custom_rcparams

plt.rcParams.update(get_custom_rcparams())

def run_method(args):

    n_dim = 2
    func, target = objective_functions[args.func](dim=n_dim, use_jax=False)
    init_cov = np.identity(n_dim) * args.init_normal_std

    n_samples = get_sample_size(n_dim)

    if args.method == "quads":
        n_samples = int(n_samples / 2 + 1)

    init_mean = np.random.rand(n_dim)

    config = {}
    config.update(vars(args))

    init_threshold = func(init_mean)

    config.update({
        "n_dim": n_dim,
        "n_samples": n_samples,
        "target": target,
        "init_cov": init_cov,
        "init_mean": init_mean,
        "init_threshold": init_threshold,
    })

    if args.method == "quads":
        _, (_, _, _, param_hists, samples_hist) = quads.run_quads(func, config, verbose=config["verbose"], return_samples=True)
        param_hists = [param.cma_param for param in param_hists]
    elif args.method == "cmaes":
        _, (_, _, _, param_hists, samples_hist) = cmaes.run_cmaes(func, config, verbose=config["verbose"], return_samples=True)


    return param_hists, samples_hist


def plot_animation(param_hists: List[CMAParam], sample_hists, func, output_path):
    func, target = objective_functions[func](dim=2, use_jax=False)
    #plot function
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = func(np.stack([X, Y], axis=-1))

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=100)
    ax.pcolormesh(X, Y, Z)

    ax.scatter(target[0], target[1], marker="*", color="red", s=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    #plot normal distributions by elipse
    # and plot samples

    for i in range(len(param_hists)-1):
        param_hist = param_hists[i]
        mean = param_hist.mean
        cov = param_hist.cov
        step_size = param_hist.step_size
        # ax.contour(X, Y, multivariate_normal.pdf(np.stack([X, Y], axis=-1), mean, cov*step_size**2), 4,
        #            colors=cm.hsv(i/len(param_hists)), alpha=0.5, linewidths=1)
        lambda_, v = np.linalg.eig(cov*step_size**2)
        lambda_ = np.sqrt(lambda_)
        ell = pat.Ellipse(xy=(mean[0], mean[1]),
                      width=lambda_[0], height=lambda_[1],
                      angle=np.rad2deg(np.arccos(v[0, 0])),
                      color=cm.hsv(i/len(param_hists)), alpha=0.9, lw=3)
        ell.set_facecolor('none')
        ax.add_artist(ell)

        ax.scatter(sample_hists[i][:, 0], sample_hists[i][:, 1], marker=".", color=cm.hsv(i/len(param_hists)), alpha=0.9, s=200, ec="black", lw=0.5)

    fig.savefig(output_path)


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-o", type=str, default="quads_animation.png")
    parser = parse_args(parser)

    args = parser.parse_args()

    # when no param_hist.npy, run quads
    # if not os.path.exists("hists.npy"):
    #     param_hist, samples_hist = run_quads(args)
    #     np.save("hists.npy", (param_hist, samples_hist))
    # else:
    #     param_hist, samples_hist = np.load("hists.npy")

    param_hist, samples_hist = run_method(args)

    plot_animation(param_hist, samples_hist, args.func, args.o)


if __name__ == "__main__":
    main()

