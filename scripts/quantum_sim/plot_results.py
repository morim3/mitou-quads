import matplotlib.pyplot as plt
import wandb
import os
import numpy as np

import json
import glob
from matplotlib.colors import hsv_to_rgb
from matplotlib.legend_handler import HandlerTuple

plt.rcParams["font.family"] = "Noto Sans CJK JP"   # 使用するフォント
plt.rcParams["font.size"] = 15

color = [hsv_to_rgb((260.0/360.0, 0.5, 0.85)), hsv_to_rgb((120.0 / 360.0, 0.5, 0.7)), hsv_to_rgb((30.0/360.0, 0.5, 0.95))]

def eval_to_func_val(results):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for exp_id, (name, (_, table)) in enumerate(results.items()):

        table = [[trial, eval, float(func[0]) if isinstance(func, list) else func] for trial, eval, func in table["data"]]
        table = np.array(table)
        max_trial = int(np.max(table[:, 0]))
        ax.set_yscale('log')
        for i in range(max_trial):
            table_i = table[table[:, 0] == i]
            base_zorder = exp_id * 1000 + i
            r, g, b = [*color[exp_id]]
            ax.plot(table_i[:, 1], table_i[:, 2], c=[
                    r, g, b, 0.2], zorder=base_zorder, lw=2)

            if i == 0:
                ax.scatter([table_i[:, 1][-1]], [table_i[:, 2][-1]],
                           edgecolors=["black"], c=[[r, g, b]], s=50,
                           zorder=10000 + base_zorder, alpha=0.7, label=name)
            else:
                ax.scatter([table_i[:, 1][-1]], [table_i[:, 2][-1]],
                           edgecolors=["black"], c=[[r, g, b]], s=50,
                           zorder=10000 + base_zorder, alpha=0.7,)

    ax.legend(fontsize=15, loc="upper right", bbox_to_anchor=(0.9, 0.88)).get_frame().set_alpha(1)

    ax.set_xlabel("関数評価回数")
    ax.set_ylabel("\n".join("関数評価値"), rotation=0, loc="center")
    ax.yaxis.set_label_coords(-0.09, 0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=500)


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
    parser.add_argument("--quads_id")
    parser.add_argument("--grover_id")
    parser.add_argument("--cmaes_id")
    parser.add_argument("--output")
    args = parser.parse_args()

    results = {}
    for name, id in [("グローバー適応探索", args.grover_id),("提案手法", args.quads_id),  ("CMA-ES", args.cmaes_id)]:
        api = wandb.Api()
        artifact = api.artifact(
            f'morim3/mitou-quads/run-{id}-optimizationprocess_table:v0', )

        globbed = glob.glob(f'./artifacts/run-{id}-optimizationprocess_table*')
        print(globbed)
        if len(globbed) == 0:
            artifact_dir = artifact.download()
        else:
            artifact_dir = globbed[0]

        artifact_dir = os.path.join(artifact_dir, "optimization process_table.table.json")

        summary = api.run(f"morim3/mitou-quads/{id}")

        with open(artifact_dir, "r") as f:
            table = json.load(f)

        results[name] = (summary, table)

    eval_to_func_val(results)

