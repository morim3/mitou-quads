import time
import numpy as np
from collections import Counter
import random
import jax.random
from jax import numpy as jnp
import bisect


def choice(options,probs):
    x = np.random.rand()
    cumsum = np.cumsum(probs)
    ind = bisect.bisect(cumsum, x*cumsum[-1])
    return options[ind]

N = 2 ** 24

options = range(N)
weights = [random.random() for _ in range(N)]
runs = 10

print("start")
#
# now = time.time()
# temp = []
# for i in range(runs):
#     op = random.choices(options,weights=probs)
#     temp.append(op)
# print(time.time()-now)
#
probs = np.array(weights)
options = np.arange(N)
now = time.time()
temp = []
for i in range(runs):
    op = choice(options,probs)
    temp.append(op)

print(time.time()-now)
#
#
# print("")
# now = time.time()
# temp = []
# for i in range(runs):
#     op = np.random.choice(options, p=probs)
#     temp.append(op)
# temp = Counter(temp)
# print(time.time()-now)
#
print("")
now = time.time()
temp = []
for i in range(runs):
    op = np.random.choice(options, p=probs)
    temp.append(op)
print(time.time()-now)

probs = jnp.array(probs)
options = jnp.arange(N)
key = jax.random.PRNGKey(0)
print("")
now = time.time()
temp = []
for i in range(runs):
    key, subkey = jax.random.split(key)
    op = jax.random.choice(subkey, options, p=probs)
    temp.append(op)
print(time.time()-now)
