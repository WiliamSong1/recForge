import copy
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from .data import histories
from .models import MatrixFactorization, TwoTower, sample_negatives


def seed_everything(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def train_models(frame, genres, n_users, config, validation=None):
    seed_everything(config["seed"], config["threads"])
    device = config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(config["seed"])
    n_items = len(genres)
    seen = histories(frame, n_users)
    eligible = np.sort(frame.i.unique())
    positive = frame[frame.rating >= config["positive_rating"]]
    positive = positive[positive.u.map(lambda u: len(seen[u]) < len(eligible))]
    if positive.empty:
        raise ValueError("No positive examples with available negatives")
    mf = MatrixFactorization(n_users, n_items, config["dimension"], frame.rating.mean()).to(device)
    tower = TwoTower(n_users, n_items, genres, config["dimension"]).to(device)
    logs = []
    for name, model, examples in [("mf", mf, frame), ("two_tower", tower, positive)]:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        u = examples.u.to_numpy()
        i = examples.i.to_numpy()
        y = examples.rating.to_numpy(dtype=np.float32)
        best, state, stale = float("inf"), None, 0
        # Fixed validation negatives; warm users and items only; do not consult test outcomes.
        val = None
        if validation is not None:
            val = validation[validation.u.isin(frame.u.unique()) & validation.i.isin(eligible)].copy()
            if name == "two_tower":
                val = val[(val.rating >= config["positive_rating"]) & val.u.map(lambda x: len(seen[x]) < len(eligible))]
                val_seen = [s.copy() for s in seen]
                for vu, vi in val[["u", "i"]].itertuples(index=False, name=None):
                    val_seen[vu].add(vi)
                val = val[val.u.map(lambda x: len(val_seen[x].intersection(eligible)) < len(eligible))]
                vn = sample_negatives(val.u.to_numpy(), val_seen, eligible, np.random.default_rng(123)) if len(val) else []
        for epoch in range(config["epochs"]):
            start = time.perf_counter()
            model.train()
            order = rng.permutation(len(u))
            negatives = sample_negatives(u, seen, eligible, rng) if name == "two_tower" else None
            total = 0.0
            for offset in range(0, len(order), config["batch_size"]):
                idx = order[offset:offset + config["batch_size"]]
                bu = torch.as_tensor(u[idx], device=device)
                bi = torch.as_tensor(i[idx], device=device)
                if name == "mf":
                    loss = F.mse_loss(model(bu, bi), torch.as_tensor(y[idx], device=device))
                else:
                    bn = torch.as_tensor(negatives[idx], device=device)
                    loss = F.softplus(-(model(bu, bi) - model(bu, bn)) / 0.1).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(idx)
            model.eval()
            metric = total / len(u)
            validation_loss = None
            if val is not None and len(val):
                losses = []
                with torch.inference_mode():
                    for start_idx in range(0, len(val), config["batch_size"]):
                        v = val.iloc[start_idx:start_idx + config["batch_size"]]
                        vu, vi = torch.as_tensor(v.u.to_numpy(), device=device), torch.as_tensor(v.i.to_numpy(), device=device)
                        if name == "mf":
                            loss = F.mse_loss(model(vu, vi), torch.as_tensor(v.rating.to_numpy(dtype=np.float32), device=device), reduction="sum")
                        else:
                            neg = torch.as_tensor(vn[start_idx:start_idx + len(v)], device=device)
                            loss = F.softplus(-(model(vu, vi) - model(vu, neg)) / 0.1).sum()
                        losses.append(loss.item())
                validation_loss = sum(losses) / len(val)
                metric = validation_loss
            logs.append({"model": name, "epoch": epoch + 1, "train_loss": total / len(u), "validation_loss": validation_loss,
                         "seconds": time.perf_counter() - start})
            print(logs[-1], flush=True)
            if metric < best - 1e-5:
                best, state, stale = metric, copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
            if validation_loss is not None and stale >= config["patience"]:
                break
        model.load_state_dict(state)
        model.cpu().eval()
    return mf, tower, logs
