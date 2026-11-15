"""
test_lit_baselines.py -- Fit the literature baselines on the
highrise scenario (seed 1) and report test RMSE and wall time.

    cd src && python test_lit_baselines.py [name ...]

Reference values from the previous run (highrise): CALM about 3.6-3.7 dB,
random forest about 3.9 dB, kriging about 4.9 dB, 3GPP prior about 8 dB.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

torch.set_num_threads(4)

import dataset  # noqa: E402
from scenario import Scenario  # noqa: E402
from lit_baselines import LAST_INFO, LIT_BASELINES, run_lit_baseline  # noqa: E402


def rmse(pred, y):
    return float(np.sqrt(np.mean((np.asarray(pred, np.float64)
                                  - np.asarray(y, np.float64)) ** 2)))


def main(names):
    t0 = time.time()
    sc = Scenario("highrise", seed=1, scfg=dataset.SamplingConfig(seed=1))
    print(f"scenario built in {time.time() - t0:.1f}s")
    tr, te = sc.tables["train"], sc.tables["test"]
    tabs = {"cal": sc.tables["cal"], "test": te}
    print(f"train rows={len(tr['y'])}  test rows={len(te['y'])}  sites={sc.B}")
    print(f"{'3gpp prior':10s} test RMSE = {rmse(te['prior'], te['y']):.3f} dB")

    rows = []
    for name in names:
        t = time.time()
        pred = run_lit_baseline(name, sc, tabs, tr, seed=0)
        dt = time.time() - t
        assert set(pred) == set(tabs), (name, set(pred))
        for k in tabs:
            assert pred[k].shape == tabs[k]["y"].shape, (name, k)
            assert np.isfinite(pred[k]).all(), (name, k)
        r = rmse(pred["test"], te["y"])
        info = LAST_INFO.get(name, {})
        extra = ""
        if name == "phygat":
            extra = (f"  regime={info.get('regime')}  "
                     f"kriging-prior={rmse(info['pred_kriging_prior']['test'], te['y']):.3f}  "
                     f"residual-only={rmse(info['pred_residual_only']['test'], te['y']):.3f}")
        print(f"{name:10s} test RMSE = {r:.3f} dB   cal RMSE = "
              f"{rmse(pred['cal'], tabs['cal']['y']):.3f} dB   wall = {dt:.0f}s   "
              f"epochs = {info.get('epochs_run')}   params = {info.get('n_params')}"
              + extra, flush=True)
        rows.append((name, r, dt))

    print("\nsummary (highrise, seed 1)")
    for name, r, dt in rows:
        print(f"  {name:10s} {r:6.3f} dB  {dt:6.0f} s")


if __name__ == "__main__":
    main(sys.argv[1:] or LIT_BASELINES)
