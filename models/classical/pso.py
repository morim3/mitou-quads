import numpy as np

from pyswarms.utils.functions import single_obj as fx
import pyswarms as ps

def run_particle_swarm_optimization(func, config, verbose):

    n_particles = config['n_samples']
    n_dimensions = config['n_dim']
    w = config['w']
    c1 = config['c1']
    c2 = config['c2']
    f_tol_rel = config['f_tol_rel']
    tol_iter = config['tol_iter']
    max_iter = config['max_iter']
    target = config['target']

    bounds = ([0] * n_dimensions, [1] * n_dimensions)

    options = {'c1': c1, 'c2': c2, 'w':w}
    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=n_dimensions, options=options, bounds=bounds, ftol=f_tol_rel, ftol_iter=tol_iter)
    best_cost, best_pos = optimizer.optimize(func, iters=max_iter, verbose=verbose)

    # (n_iter, n_particles, n_dim)
    pos_hist = np.array(optimizer.pos_history).reshape(-1, n_dimensions)

    ## get first iter where the distance to target is less than 0.01
    dist_target_hist = np.linalg.norm(pos_hist - target[None, :], axis=-1)
    idxes = np.where(dist_target_hist < 0.01)
    if len(idxes[0]) > 0:
        idx = idxes[0][0]
        return pos_hist[idx], ([], [idx], [dist_target_hist[idx]], [])
    else:
        return best_pos, ([], [pos_hist.shape[0]], [dist_target_hist[-1]], [])



# def run_particle_swarm_optimization(func, config):
#     # Initialize the particles
#
#     n_particles = config['n_samples']
#     n_dimensions = config['n_dim']
#     w = config['w']
#     c1 = config['c1']
#     c2 = config['c2']
#     tol_rel_fun = config['tol_rel_fun']
#     tol_iter = config['tol_iter']
#     target = config['target']
#
#     bounds = ([0] * n_dimensions, [1] * n_dimensions)
#
#     eval_num_hist = []
#     min_func_hist = []
#     dist_target_hist = []
#     param_hist = []
#
#     particles = np.random.uniform(bounds[0], bounds[1], (n_particles, n_dimensions))
#     
#     velocities = np.random.uniform(-1, 1, (n_particles, n_dimensions)) * 0.01
#     personal_best_positions = particles
#     personal_best_scores = np.full(n_particles, np.inf)
#     global_best_position = np.zeros(n_dimensions)
#     global_best_score = np.inf
#
#     criteria_counter = 0
#     global_best_score = np.inf
#     stop = False
#     while not stop:
#
#         prev_best_score = global_best_score
#         scores = [func(p)[0] for p in particles]
#         for i in range(n_particles):
#             # Evaluate the objective function
#             score = scores[i]
#             if score < personal_best_scores[i]:
#                 personal_best_scores[i] = score
#                 personal_best_positions[i] = particles[i]
#
#         swarm_best_score = np.min(personal_best_scores)
#         if swarm_best_score < global_best_score:
#             global_best_score = swarm_best_score
#             global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
#
#         for i in range(n_particles):
#             r1 = np.random.rand(n_dimensions)
#             r2 = np.random.rand(n_dimensions)
#             velocities[i] = w * velocities[i] + c1 * r1 * (personal_best_positions[i] - particles[i]) + c2 * r2 * (global_best_position - particles[i])
#             particles[i] += velocities[i]
#
#             dist_target_hist.append(np.linalg.norm(particles[i] - target))
#             if dist_target_hist[-1] < config["terminate_eps"]:
#                 stop = True
#                 break
#
#             # particles[i] = np.maximum(particles[i], bounds[0])
#             # particles[i] = np.minimum(particles[i], bounds[1])
#
#         if global_best_score != np.inf and (global_best_score - prev_best_score) / (1+abs(global_best_score)) > tol_rel_fun:
#             criteria_counter = 0
#
#         # Check the stopping criteria
#         criteria_counter += 1
#         if criteria_counter > tol_iter:
#             break                       
#         
#
#         eval_num_hist.append(n_particles)
#
#     return global_best_position, (np.array(min_func_hist), np.array(eval_num_hist), np.array(dist_target_hist), param_hist)

if __name__ == "__main__":

    n_dim = 5
    target = np.ones(n_dim) * 0.5
    config = {
        "n_samples": 10,
        "n_dim": n_dim,
        "w": 0.9,
        "c1": 0.5,
        "c2": 0.3,
        "f_tol_rel": 1e-4,
        "tol_iter": 50,
        "max_iter": 1000,
        "target": target,
        "terminate_eps": 0.01
    }

    def func(x):
        return np.sum((x - target[None,:]) ** 2, axis=-1)

    result_param, (min_func_hist, eval_num_hist, dist_target_hist, param_hist) = run_particle_swarm_optimization(func, config)
    print(dist_target_hist[-1])
    print(eval_num_hist[-1])


