import os
import torch
import random
import numpy as np 
from sklearn.cluster import KMeans

def ADE_FDE(y_, y, batch_first=False):
    # average displacement error
    # final displacement error
    # y_, y: S x L x N x 2
    if torch.is_tensor(y):
        err = (y_ - y).norm(dim=-1)
    else:
        err = np.linalg.norm(np.subtract(y_, y), axis=-1)
    if len(err.shape) == 1:
        fde = err[-1]
        ade = err.mean()
    elif batch_first:
        fde = err[..., -1]
        ade = err.mean(-1)
    else:
        fde = err[..., -1, :]
        ade = err.mean(-2)
    return ade, fde

def clustering(y, n_samples):
    # y: S x L x 2
    traj1d = np.reshape(y[...,:2], (*y.shape[:-2],-1))
    clustering = KMeans(n_samples, n_init=1).fit(traj1d)
    return np.reshape(clustering.cluster_centers_, (n_samples, *traj1d.shape[1:-1], -1, 2)), \
        [np.count_nonzero(clustering.labels_ == i) for i in range(n_samples)]

def seed(seed: int):
    rand = seed is None
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder="big")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = not rand
    torch.backends.cudnn.benchmark = rand

def get_rng_state(device):
    return (
        torch.get_rng_state(), 
        torch.cuda.get_rng_state(device) if torch.cuda.is_available and "cuda" in str(device) else None,
        np.random.get_state(),
        random.getstate(),
        )

def set_rng_state(state, device):
    torch.set_rng_state(state[0])
    if state[1] is not None: torch.cuda.set_rng_state(state[1], device)
    np.random.set_state(state[2])
    random.setstate(state[3])
