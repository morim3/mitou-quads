import numpy as np

def get_rastrigin(dim=3, target=None, square_term=1, ):
    if target is None:
        target = np.ones(dim, dtype=np.float32) * 0.5

    def fun(x,):
        x = (x-target[None]) * 10.24
        return np.sum(square_term * x ** 2 +
            10 * (1 - np.cos(2 * np.pi * x)), axis=-1)
    return fun, target

def get_ackley(dim=3, target=None):
    if target is None:
        target = np.ones(dim, dtype=np.float32) * 0.5

    def fun(x, ):
        x = (x - target[None]) * 32.768 * 2
        return 20 - 20 * np.exp(-0.2*np.sqrt(np.mean(x**2, axis=-1))) + np.e - np.exp(np.mean(np.cos(2*np.pi*x ), axis=-1))

    return fun, target

def get_squared(dim=3, target=None, square_term=1):
    if target is None:
        target = np.ones(dim, dtype=np.float32)*0.75
    def squared(x):
        return  square_term * np.sum((x - target[..., :]) ** 2, axis=-1)
    return squared, target

def get_styblinski_tang(dim=3, target=None):

    target = np.array([-2.90353402777118 / 10 + 0.5] * dim, dtype=np.float32)

    def base_fun(x):
        return np.sum(x**4 - 16 * x**2 + 5 * x, axis=-1) / 2
    def styblinski_tang(x):
        optimal_point = -2.90353402777118 * np.ones_like(x)

        # x in [0,1] -> y in [-5, 5]
        # optimal_point = 0.20964659722
        y = (x - 0.5) * 10
        # offset optimal value
        return base_fun(y) - base_fun(optimal_point)
    return styblinski_tang, target

def get_easom(dim=2, target=None):
    target = np.array([(np.pi+100)/200, (np.pi+100)/200], dtype=np.float32)
    def fun(x):
        x = x * 200 - 100
        return - np.cos(x[:, 0])*np.cos(x[:, 1])*np.exp(-((x[:, 0]-np.pi)**2+(x[:, 1]-np.pi)**2)) + 1
        
    return fun, target

def get_schwefel(dim=3, target=None):
    target = (np.ones(dim, dtype=np.float32)*420.9687 + 500) / 1000
    def fun(x):
        x = x * 1000 -500
        return - np.sum(x*np.sin(np.sqrt(np.abs(x))), axis=-1)

    return fun, target

def rosenbrock(x):
    return np.sum((x[:, 1:] - x[:, :-1] ** 2) ** 2 * 100 + (1 - x[:, :-1])**2)
