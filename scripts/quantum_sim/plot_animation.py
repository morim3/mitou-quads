import os
import random
from typing import List

import matplotlib.cm as cm
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import BoundaryNorm, Normalize
import matplotlib.gridspec as gridspec
import matplotlib.patches as pat
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.stats import multivariate_normal

from models.amp_sim import grover_adaptive, quads
from models.classical import cmaes
from models.parameters import CMAParam, QuadsParam
from models.run_methods import get_sample_size, parse_args
from utils.mplsetting import get_custom_rcparams
from utils.objective_functions import objective_functions

plt.rcParams.update(get_custom_rcparams())

np.random.seed(11)
random.seed(11)

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

    fig = plt.figure(figsize=(10, 10), dpi=100)
    gs = gridspec.GridSpec(2, 2, width_ratios=[20, 1], height_ratios=[20, 1])

    ax = fig.add_subplot(gs[0, 0])
    ax_iter = fig.add_subplot(gs[1, 0])
    ax_cbar = fig.add_subplot(gs[0, 1])
    # fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=100)
    im_func = ax.pcolormesh(X, Y, Z, cmap="gray")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xlabel("Variable 1")
    ax.set_ylabel("Variable 2")

    #plot normal distributions by elipse
    # and plot samples

    cmap = cm.jet
    cmap_lin = np.linspace(0, 1, len(param_hists)-1)
    for i in range(len(param_hists)-1):
        cmap_i = cmap(cmap_lin[i])
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
                      color=cmap_i, alpha=0.9, lw=3)
        ell.set_facecolor('none')
        ax.add_artist(ell)

        ax.scatter(sample_hists[i][:, 0], sample_hists[i][:, 1], marker=".", color=cmap_i, alpha=0.9, s=200, ec="black", lw=0.5)

    # norm = Normalize(vmin=1/2, vmax=len(param_hists)-1/2)


    bounds = np.arange(len(param_hists))
    norm = BoundaryNorm(bounds, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([]) 
    cbar_iter = fig.colorbar(sm, cax=ax_iter, ticks=np.arange(len(param_hists))+.5, orientation="horizontal")
    cbar_iter.set_label('Iteration')
    cbar_iter.ax.set_xticklabels(np.arange(len(param_hists)))

    cbar_func = fig.colorbar(im_func, cax=ax_cbar)
    cbar_func.set_label('Function Value')

    ax.scatter(target[0], target[1], marker="*", color="red", s=500)
    ax.set_aspect('equal')

    fig.tight_layout()
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

