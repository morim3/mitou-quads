import matplotlib.pyplot as plt
import wandb
import os
import numpy as np
import pickle

import json
import glob
from matplotlib.colors import hsv_to_rgb
from matplotlib.legend_handler import HandlerTuple

plt.rcParams["font.family"] = "Noto Sans CJK JP"   # 使用するフォント
plt.rcParams["font.size"] = 15

color = [hsv_to_rgb((260.0/360.0, 0.5, 0.85)), hsv_to_rgb((120.0 / 360.0, 0.5, 0.7)), hsv_to_rgb((30.0/360.0, 0.5, 0.95))]

def load_experiments(api, entity, func, versions={
        "grover": "latest",
        "cmaes": "latest", 
        "quads": "latest"}):

    prefix = f"{entity}/mitou-quads"
    experiments = []

    def get_result(method):
        artifact = api.artifact(
            f"{prefix}/result-quantum-{method}-{func}:{versions[method]}")
        return f"{artifact.download()}/result.pickle"

    with open(get_result("grover"), mode='rb') as f:
        experiments.append({
            "name": "グローバー適応探索",
            "data": pickle.load(f),
            "zorder": 10000,
            "marker": "d",
            "color": hsv_to_rgb((260.0/360.0, 0.5, 0.85)),
        })

    with open(get_result("cmaes"), mode='rb') as f:
        experiments.append({
            "name": "CMA-ES",
            "data": pickle.load(f),
            "zorder": 30000,
            "marker": "o",
            "color": hsv_to_rgb((120.0/360.0, 0.5, 0.7)),
        })

    with open(get_result("quads"), mode='rb') as f:
        proposed_quantum = pickle.load(f)
        # proposed_quantum["min_func_hists"] = [m[1:] for m in proposed_quantum["min_func_hists"]]
        experiments.append({
            "name": "QuADS(提案手法)",
            "data": proposed_quantum,
            "zorder": 20000,
            "marker": "s",
            "color": hsv_to_rgb((30.0/360.0, 0.5, 0.95)),
        })

    return experiments


def eval_to_func_val(experiments):
    global_threshold = 1.0e-3

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), dpi=300)

    ax = axes[0]

    ax.set_ylim(1e-3, 10)
    ax.set_xlim(0, 3000)

    draw_all_terminal = True

    handles_marker = [None] * 3
    handles_tragectory = [None] * 3 

    for exp_id, e in enumerate(experiments):
        data = e["data"]
        name = e["name"]
        label = f"exp_{exp_id}"
        eval_hists = data["eval_hists"]
        min_func_hists = data["min_func_hists"]

        min_func_hists = [ np.clip(np.array(min_func_hist), global_threshold, None) for min_func_hist in min_func_hists]

        # seq_len = int(np.ceil(np.max(term_eval_nums)))

        ax.set_yscale('log')
        for i in range(len(eval_hists)):
            r, g, b = [*e["color"]]
            handles_tragectory[exp_id], = ax.plot(
                eval_hists[i], min_func_hists[i], c=[r, g, b, 0.1], zorder=e["zorder"] + i, lw=4)

            # if min_func_hists[i][-1] <= global_threshold:
            #     min_func_hists[i][-1] *= (0.88) ** (exp_id + 1)
            
            if draw_all_terminal or min_func_hists[i][-1] > global_threshold:
                handles_marker[exp_id] = ax.scatter(
                    [eval_hists[i][-1]], [min_func_hists[i][-1]],
                    edgecolors=["black"], c=[[r, g, b]], s=60, alpha=0.7,
                    # marker=data["marker"],
                    marker="d" if data["converged_to_global"][i] else "o",
                    zorder=100_000 + e["zorder"], clip_on=False)

    ax.get_xaxis().set_tick_params(pad=6)

    # print(handles_marker, handles_tragectory)

    ax = axes[1]

    xlim = 3000
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0, xlim)

    handles_all = []
    handles_good = []

    for exp_id, e in enumerate(experiments):
        data = e["data"]
        eval_hists = data["eval_hists"]
        min_func_hists = [m[1:] for m in data["min_func_hists"]]
        eval_sums = [e[-1] for e in data["eval_hists"]]
        final_min_vals = [m[-1] for m in data["min_func_hists"]]

        print(data["converged_to_global"])

        good_eval_sums = [e for e, c in zip(eval_sums, data["converged_to_global"]) if c]

        def increase_curve(x, xlim, ylim):
            y = np.array([0] + np.linspace(0, ylim, len(x)).tolist() + [ylim])
            x = np.array([0] + sorted(x) + [xlim])
            return x, y
        
        cum_rate = increase_curve(eval_sums, xlim, 1.0)
        global_conv_rate = float(len(good_eval_sums)) / float(len(eval_sums))
        good_cum_rate = increase_curve(good_eval_sums, xlim, global_conv_rate)
        

        handle_all, = ax.plot(*cum_rate, c=e["color"], lw=2, ls="dashed", dashes=[1.5,1.0])
        handle_good, = ax.plot(*good_cum_rate, c=e["color"], lw=2, ls="solid")
        handles_tragectory[exp_id] = handle_good
        handles_all.append(handle_all)
        handles_good.append(handle_good)

    axes[0].legend(zip(handles_marker, handles_tragectory),
        [e["name"] for e in experiments],
        handler_map={tuple: HandlerTuple(ndivide=None)},
        fontsize=9, loc="upper right", bbox_to_anchor=(0.9, 0.88))

    axes[0].set_xlabel("関数評価回数")
    axes[0].set_ylabel("\n".join("関数評価値"), rotation=0, loc="center")
    axes[0].yaxis.set_label_coords(-0.09, 0.4)

    axes[1].legend([tuple(handles_all), tuple(handles_good)],
        ["収束全体", "大域的収束"],
        handler_map={tuple: HandlerTuple(ndivide=None)},
        fontsize=9, loc="lower right", bbox_to_anchor=(0.9, 0.1),
        handlelength = 8)

    axes[1].set_xlabel("関数評価回数")
    axes[1].set_ylabel("\n".join("サンプル割合"), rotation=0, loc="center")
    axes[1].yaxis.set_label_coords(-0.09, 0.4)
    fig.tight_layout()
    return fig, axes


# def eval_to_converge(results):
#     fig, ax = plt.subplots(figsize=(8, 4.5))
#
#     for exp_id, (name, (summary, table)) in enumerate(results.items()):
#
#         handle_all, = ax.plot(eval_sums, eval_sums_y, c=e["color"], lw=2, ls="dashed", dashes=[1.5,1.0])
#         handle_good, = ax.plot(good_eval_sums, good_eval_sums_y, c=e["color"], lw=2, ls="solid")
#
#     ax.set_xlabel("関数評価回数")
#     ax.set_ylabel("\n".join("関数評価値"), rotation=0, loc="center")
#     ax.yaxis.set_label_coords(-0.09, 0.4)
#
#     ax.legend([tuple(handles_all), tuple(handles_good)],
#         ["収束全体", "大域的収束"],
#         handler_map={tuple: HandlerTuple(ndivide=None)},
#         fontsize=9, loc="lower right", bbox_to_anchor=(0.9, 0.1),
#         handlelength = 8)
#
#     ax.set_xlabel("関数評価回数")
#     ax.set_ylabel("\n".join("サンプル割合"), rotation=0, loc="center")
#     ax.yaxis.set_label_coords(-0.09, 0.4)
#     fig.tight_layout()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", type=str)
    parser.add_argument("--func", type=str, default="rastrigin")
    parser.add_argument("--v_quads", type=str, default="latest")
    parser.add_argument("--v_grover", type=str, default="latest")
    parser.add_argument("--v_cmaes", type=str, default="latest")
    # parser.add_argument("--output")
    args = parser.parse_args()

    api = wandb.Api()

    experiments = load_experiments(api, args.entity, args.func, versions={
        "grover": args.v_grover,
        "cmaes": args.v_cmaes,
        "quads": args.v_quads
    })

    fig, axes = eval_to_func_val(experiments)
    plt.savefig("outputs/fig1.pdf")

