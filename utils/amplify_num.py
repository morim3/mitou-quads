import numpy as np

def optimal_amplify_num(p):
    sqrt_rate = np.sqrt(p)
    return int(np.arccos(sqrt_rate) / (2 * np.arcsin(sqrt_rate)))

def upper_bound_amplify_num(p, lam=6/5):
    # Boyer, M., Brassard, G., Høyer, P., & Tapp, A. (1998). Tight bounds on quantum searching. Fortschritte der Physik: Progress of Physics, 46(4‐5), 493-505.
    m_0 = 1 / (2 * np.sqrt((1-p) * p))
    return 1/2 * lam / (lam-1) * m_0 + 2 * lam / (4 - 3 * lam) * m_0
