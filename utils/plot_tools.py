import numpy as np
import matplotlib.pyplot as plt


def plot_function_surface(func, target=None, func_name="", ax=None):
    X, Y = np.meshgrid(np.linspace(0, 1, 500), np.linspace(0, 1, 500))
    grid = np.stack([X, Y], axis=-1).reshape((-1, 2))
    func_value = func(grid).reshape((500, 500))
    if ax is None:
        fig, ax = plt.subplots(dpi=300)
    ax.imshow(func_value)
    ax.set_title(f"{func_name} function")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.scatter([target[0]], [target[1]], marker='*', s=1)
    return ax

def plot_optimization_dynamics(eval_hists, min_val_hists, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        plt.yscale('log')

    term_eval_nums = np.array([np.cumsum(eval_hist)[-1] for eval_hist in eval_hists])
    term_vals = np.array([min_val_hist[-1] for min_val_hist in min_val_hists])

    seq_len = int(np.ceil(np.max(term_eval_nums)))
    x = np.arange(seq_len)

    ys = []

    for eval_hist, min_val_hist in zip(eval_hists, min_val_hists):
        eval_num_hist = np.cumsum(eval_hist)
        y = np.interp(np.arange(np.ceil(eval_num_hist[-1])), eval_num_hist, min_val_hist)
        y = np.concatenate([y, np.zeros(len(x) - len(y))])
        ys.append(y)

    ys = np.array(ys)

    median, q25, q75, q10, q90 = [], [], [], [], []

    for i in range(seq_len):
        y = ys[ys[:, i] > 0, i]
        if len(y) == 1:
            y = np.concatenate([y, y])
        median.append(np.median(y))
        q25.append(np.quantile(y, 0.25))
        q75.append(np.quantile(y, 0.75))
        q10.append(np.quantile(y, 0.10))
        q90.append(np.quantile(y, 0.90))

    median = np.array(median)
    q10 = np.array(q10)
    q90 = np.array(q90)
    q25 = np.array(q25)
    q75 = np.array(q75)

    # for i in range(len(eval_hists)):
    #     ax.plot(np.cumsum(eval_hists[i]), min_val_hists[i])
    #     ax.plot(x, ys[i])

    ax.fill_between(x, q10, q90, color="cyan")
    ax.fill_between(x, q25, q75, color="magenta")
    ax.plot(x, median, color="black", lw=2)
    ax.scatter(term_eval_nums, term_vals)

    return ax

def plot_optimization_statistics(eval_hists, min_val_hists, ax=None, c_fill1="blue", c_quantile="cyan", a_fill1=0.9, a_quantile=0.9, c_median="black", zorder=1, label="", seq_len=-1):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        plt.yscale('log')

    term_eval_nums = np.array([np.cumsum(eval_hist)[-1] for eval_hist in eval_hists])
    if seq_len == -1:
        seq_len = int(np.max(term_eval_nums))
    x = np.arange(seq_len)

    ys = []

    for eval_hist, min_val_hist in zip(eval_hists, min_val_hists):
        eval_num_hist = np.cumsum(eval_hist)
        y = np.interp(np.arange(eval_num_hist[-1]), eval_num_hist, min_val_hist)
        y = np.concatenate([y, np.ones(len(x) - len(y))*y[-1]])
        ys.append(y)

    ys = np.array(ys)

    median, q25, q75, q10, q90 = [], [], [], [], []

    for i in range(seq_len):
        y = ys[:, i]
        if len(y) == 1:
            y = np.concatenate([y, y])
        median.append(np.median(y))
        q25.append(np.quantile(y, 0.25))
        q75.append(np.quantile(y, 0.75))
        q10.append(np.quantile(y, 0.10))
        q90.append(np.quantile(y, 0.90))

    median = np.array(median)
    q10 = np.array(q10)
    q90 = np.array(q90)
    q25 = np.array(q25)
    q75 = np.array(q75)

    # for i in range(len(eval_hists)):
    #     ax.plot(np.cumsum(eval_hists[i]), min_val_hists[i])
    #     ax.plot(x, ys[i])

    ax.fill_between(x, q10, q90, color=c_fill1, alpha=a_fill1, zorder=zorder, label=label)
    ax.plot(x, q25, color=c_quantile, alpha=a_quantile, lw=2, zorder=zorder)
    ax.plot(x, q75, color=c_quantile, alpha=a_quantile, lw=2, zorder=zorder)
    ax.plot(x, median, color=c_median, lw=2, zorder=zorder)

    return ax
