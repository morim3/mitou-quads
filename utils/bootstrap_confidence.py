import numpy as np

def bootstrap_confidence(samples, statistics_fun, n_bootstrap=2000, alpha=0.05): 

    bootstrap_statistics = np.array([])
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(np.arange(len(samples)), size=len(samples))
        bootstrap_statistics = np.append(bootstrap_statistics, statistics_fun(samples[bootstrap_sample]))

    bootstrap_statistics = np.sort(bootstrap_statistics)

    return bootstrap_statistics[int((n_bootstrap+1) * alpha)-1], bootstrap_statistics[int((n_bootstrap+1) * (1-alpha))-1]

