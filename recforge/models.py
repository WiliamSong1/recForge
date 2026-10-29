"""Rating MF and metadata-aware neural retrieval; all scores remain uncalibrated."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MatrixFactorization(nn.Module):
    def __init__(self, users, items, dimension, mean=3.5):
        super().__init__()
        self.user = nn.Embedding(users, dimension)
        self.item = nn.Embedding(items, dimension)
        self.ub = nn.Embedding(users, 1)
        self.ib = nn.Embedding(items, 1)
        self.register_buffer("mean", torch.tensor(float(mean)))
        nn.init.normal_(self.user.weight, std=0.05)
        nn.init.normal_(self.item.weight, std=0.05)
        nn.init.zeros_(self.ub.weight)
        nn.init.zeros_(self.ib.weight)

    def forward(self, u, i):
        return self.mean + self.ub(u).squeeze(-1) + self.ib(i).squeeze(-1) + (self.user(u) * self.item(i)).sum(-1)


class TwoTower(nn.Module):
    def __init__(self, users, items, genres, dimension):
        super().__init__()
        self.user_id = nn.Embedding(users, dimension)
        self.item_id = nn.Embedding(items, dimension)
        self.register_buffer("genres", torch.as_tensor(genres, dtype=torch.float32))
        self.user_net = nn.Sequential(nn.Linear(dimension, dimension * 2), nn.ReLU(), nn.Linear(dimension * 2, dimension))
        self.item_net = nn.Sequential(nn.Linear(dimension + genres.shape[1], dimension * 2), nn.ReLU(), nn.Linear(dimension * 2, dimension))

    def users(self, u):
        return F.normalize(self.user_net(self.user_id(u)), dim=-1)

    def items(self, i):
        return F.normalize(self.item_net(torch.cat([self.item_id(i), self.genres[i]], -1)), dim=-1)

    def forward(self, u, i):
        return (self.users(u) * self.items(i)).sum(-1)


def sample_negatives(users, seen, eligible, rng):
    """Reject all observed training interactions, including disliked items.

    Future positives are unavailable at training time and are intentionally not consulted.
    Users who have consumed the entire eligible catalog must be removed by the caller.
    """
    eligible = np.asarray(eligible)
    if not len(eligible):
        raise ValueError("No eligible negative items")
    if any(len(seen[int(u)].intersection(eligible)) == len(eligible) for u in np.unique(users)):
        raise ValueError("User has no unseen negative items")
    negatives = rng.choice(eligible, len(users))
    bad = np.array([int(i) in seen[int(u)] for u, i in zip(users, negatives)])
    while bad.any():
        negatives[bad] = rng.choice(eligible, int(bad.sum()))
        bad = np.array([int(i) in seen[int(u)] for u, i in zip(users, negatives)])
    return negatives
