import numpy as np

def get_rastrigin(target, square_term=1, const_term=200):
    def fun(x,):
        ## sumじゃないとだめ、meanはおかしい
        return np.sum(square_term * (10*(x - target[None, :])) ** 2 +
            10 * (1 - np.cos(2 * np.pi * 10 * (x - target[None, :]))), axis=-1) / const_term
    return fun

def get_squared(target, square_term=1):
    def squared(x):
        return  square_term * np.sum((x - target[..., :]) ** 2, axis=-1)
    return squared

def get_styblinski_tang():
    
    def base_fun(x, s):
        return np.sum(x**4 - 16 * x**2 + 5 * x, axis=-1) / 2 / s
    def styblinski_tang(x):
        optimal_point = -2.90353402777118 * np.ones_like(y)
        # x in [0,1] -> y in [-5, 5]
        y = (x - 0.5) * 10
        # offset optimal value
        return base_fun(y) - base_fun(optimal_point)
    return styblinski_tang

def rosenbrock(x):
    return np.sum((x[:, 1:] - x[:, :-1] ** 2) ** 2 * 100 + (1 - x[:, :-1])**2)
