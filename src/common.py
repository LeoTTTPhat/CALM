"""common.py -- caching, metrics and small shared utilities."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
CACHE = os.path.join(RES, "cache")
for _d in (RES, FIG, CACHE):
    os.makedirs(_d, exist_ok=True)


def _h(*a):
    return hashlib.sha1(json.dumps(a, sort_keys=True, default=str).encode()).hexdigest()[:16]


def cached(tag, *key, builder=None, force=False):
    """Disk-memoise an expensive object under results/cache/."""
    p = os.path.join(CACHE, f"{tag}_{_h(*key)}.pkl")
    if os.path.exists(p) and not force:
        with open(p, "rb") as f:
            return pickle.load(f)
    obj = builder()
    with open(p, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return obj


def save_json(name, obj):
    p = os.path.join(RES, name)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=_default)
    print(f"  -> {os.path.relpath(p, ROOT)}")
    return p


def load_json(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def err_metrics(yhat, y):
    e = np.asarray(yhat, np.float64) - np.asarray(y, np.float64)
    return dict(rmse=float(np.sqrt((e ** 2).mean())),
                mae=float(np.abs(e).mean()),
                bias=float(e.mean()),
                p90=float(np.percentile(np.abs(e), 90)),
                p99=float(np.percentile(np.abs(e), 99)))


def agg(runs, keys=("rmse", "mae", "bias", "p90", "p99")):
    """Mean +- std over repeated seeds."""
    out = {}
    for k in keys:
        v = np.array([r[k] for r in runs if k in r], dtype=float)
        if len(v):
            out[k] = float(v.mean())
            out[k + "_sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    out["n_seeds"] = len(runs)
    return out


class Timer:
    def __enter__(self):
        self.t = time.perf_counter(); return self

    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t


def paired_bootstrap(a, b, y, n_boot=2000, seed=0, stat="rmse"):
    """Paired bootstrap CI on the RMSE difference (a - b); negative favours a."""
    rng = np.random.default_rng(seed)
    ea = (np.asarray(a) - np.asarray(y)) ** 2
    eb = (np.asarray(b) - np.asarray(y)) ** 2
    n = len(ea)
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        d[i] = np.sqrt(ea[idx].mean()) - np.sqrt(eb[idx].mean())
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(delta=float(np.sqrt(ea.mean()) - np.sqrt(eb.mean())),
                ci_lo=float(lo), ci_hi=float(hi),
                p_better=float((d < 0).mean()))
