def get_costom_rcparams():
    # based on seaborn-paper context
    customrc = {
        # 'axes.labelsize': 8.8,
        # 'axes.titlesize': 9.6,
        # 'xtick.labelsize': 8,
        # 'ytick.labelsize': 8,
        'legend.fontsize': 15,
        'font.size': 15,

        # default grid.linewidth=0.8 is too bold
        'grid.linewidth': 0.2,
        'lines.linewidth': 1.4,
        'patch.linewidth': 0.24,
        'lines.markersize': 5.6,
        'lines.markeredgewidth': 0,

        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.4,
        'ytick.minor.width': 0.4,

        'xtick.minor.visible': True,
        'ytick.minor.visible': True,

        'xtick.major.pad': 5.6,
        'ytick.major.pad': 5.6,

        # font
        'font.family': 'Times New Roman',
        'mathtext.fontset': 'stix',  # Computer Modern

        # spines
        'axes.spines.top': False,
        'axes.spines.right': False,

        # grid
        'axes.axisbelow': True,

        # ticks
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    }
    return customrc

