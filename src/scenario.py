"""
scenario.py -- Assembles a complete PACT-RM scenario (city, ground truth,
crowdsensed context, probe splits, graphs and feature tensors) and caches it.

A scenario exposes, for every probe split in {train, cal, test} and every base
station b, a flat table of (query, BS) examples:

    Xq   (N, 8)     physics covariates of the query                (query feats)
    Xn   (N, k, 6)  physics + observed residual of each neighbour   (node feats)
    Xe   (N, k, 8)  geometric / temporal edge covariates            (edge feats)
    Xc   (N, 4)     cross-BS obstruction context at the query
    prior(N,)       3GPP TR 36.777 prior path loss (the physics anchor)
    y    (N,)       ground-truth path loss at the current round
    r    (N,)       residual target, y - prior

Everything downstream -- learners, baselines, conformal layer and controller --
consumes this single representation.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict

import numpy as np

import dataset as ds
from citygen import make_city
from propagation import GroundTruthField, RadioConfig

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "cache")
ALTITUDES = (100.0, 150.0, 200.0, 250.0, 300.0)


def _key(env, seed, scfg, rcfg, alts):
    blob = json.dumps(dict(env=env, seed=seed, s=asdict(scfg), r=asdict(rcfg),
                           a=list(alts)), sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


class Scenario:
    def __init__(self, env, seed=1, scfg=None, rcfg=None, alts=ALTITUDES,
                 verbose=True, city_kw=None, city=None, pl_det_override=None):
        self.env, self.seed = env, seed
        self.city_kw = dict(city_kw or {})
        self.scfg = scfg or ds.SamplingConfig(seed=seed)
        self.rcfg = rcfg or RadioConfig()
        self.alts = tuple(alts)
        self.verbose = verbose
        # Optional injection of an externally generated city and of a
        # deterministic path-loss map (ray tracing, Sec. VII-B): the drifting
        # shadowing, the prior and every downstream stage are unchanged.
        self._city_in = city
        self._pl_det_override = pl_det_override
        self._build()

    # ------------------------------------------------------------------ #
    def _build(self):
        self.city = (self._city_in if self._city_in is not None else
                     make_city(self.env, seed=self.seed, **self.city_kw))
        self.gt = GroundTruthField(self.city, self.rcfg, self.alts,
                                   seed=self.seed, n_rounds=self.scfg.n_rounds)
        if self._pl_det_override is not None:
            self.gt.pl_det = np.asarray(self._pl_det_override, np.float32)
        self.B = self.city.bs.shape[0]

        rng = np.random.default_rng(self.seed + 31337)
        meas, cal, tst = ds.draw_samples(self.gt, self.scfg)

        # Three-way split of the probe pool: train / cal / test.
        # `draw_samples` already separated cal from the rest; we further split
        # the remainder into train and test.
        ntr = len(tst.li) // 2
        perm = rng.permutation(len(tst.li))
        tr_i, te_i = perm[:ntr], perm[ntr:]

        def sub(s, i):
            return ds.Sample(s.li[i], s.ix[i], s.iy[i], s.xyz[i], s.round_idx[i])

        self.meas = meas
        self.splits = dict(train=sub(tst, tr_i), cal=cal, test=sub(tst, te_i))

        # Hold out 10% of the training probes for early stopping, so that the
        # calibration split is never used for model selection.
        nva = len(tr_i) // 10
        self.splits["val"] = sub(tst, tr_i[:nva])
        self.splits["train"] = sub(tst, tr_i[nva:])

        self.y_meas = ds.observe(self.gt, meas, rng, self.rcfg.meas_noise_db)
        self.prior_meas = np.stack(
            [ds.gather(self.gt.pl_prior, meas, b) for b in range(self.B)], axis=1)
        self.r_meas = (self.y_meas - self.prior_meas).astype(np.float32)

        self.tables = {k: self._table(v, label_seed=i)
                       for i, (k, v) in enumerate(self.splits.items())}
        if self.verbose:
            n = {k: len(v["y"]) for k, v in self.tables.items()}
            print(f"[scenario:{self.env}] meas={len(meas.li)} "
                  f"train={n['train']} val={n['val']} cal={n['cal']} "
                  f"test={n['test']} (rows = probes x {self.B} BS)")

    # ------------------------------------------------------------------ #
    def _table(self, q: ds.Sample, label_seed=0, exclude_self=False,
               y_obs_override=None):
        gt, meas, k = self.gt, self.meas, self.scfg.k_neighbors
        cur = gt.n_rounds - 1
        nbr, nbr_d = ds.build_graph(meas, q, k, metric_z=self.scfg.metric_z,
                                    exclude_self=exclude_self)

        Xq, Xn, Xe, Xg, prior, y = [], [], [], [], [], []
        pl_now = gt.pl_true
        for b in range(self.B):
            xq, pr = ds.physics_features(gt, q, b)
            xe = ds.edge_features(gt, meas, q, nbr, nbr_d, b, cur)

            # neighbour node features for this BS
            mi = ds.Sample(meas.li[nbr], meas.ix[nbr], meas.iy[nbr],
                           meas.xyz[nbr], meas.round_idx[nbr])
            flat = ds.Sample(mi.li.ravel(), mi.ix.ravel(), mi.iy.ravel(),
                             mi.xyz.reshape(-1, 3), mi.round_idx.ravel())
            xn_flat, _ = ds.physics_features(gt, flat, b)
            xn = xn_flat.reshape(nbr.shape[0], k, -1)
            # keep [prior, plos, log_d3d, h, sin_elev] and prepend the residual
            keep = np.stack([xn[..., 4], xn[..., 5], xn[..., 0],
                             xn[..., 2], xn[..., 3]], axis=-1)
            rn = self.r_meas[nbr, b][..., None] / 10.0
            xn = np.concatenate([rn, keep], axis=-1).astype(np.float32)

            Xq.append(xq); Xn.append(xn); Xe.append(xe); prior.append(pr)
            Xg.append(ds.geometry_features(gt, q, b))
            y.append(pl_now[b, q.li, q.ix, q.iy])

        Xq = np.concatenate(Xq, 0); Xn = np.concatenate(Xn, 0)
        Xe = np.concatenate(Xe, 0); prior = np.concatenate(prior, 0)
        Xg = np.concatenate(Xg, 0)
        y = np.concatenate(y, 0).astype(np.float32)

        # Cross-BS obstruction context: how shadowed is this neighbourhood,
        # aggregated over *all* base stations (a location blocked for one BS is
        # often blocked for others).
        rn_all = self.r_meas[nbr]                          # (Nq, k, B)
        w = np.exp(-nbr_d / 50.0)[..., None]
        wsum = w.sum(1) + 1e-6
        ctx = np.concatenate([
            (rn_all * w).sum(1) / wsum,                    # (Nq, B) weighted mean
        ], axis=1)
        ctx_stats = np.stack([ctx.mean(1), ctx.std(1), ctx.min(1), ctx.max(1)],
                             axis=1).astype(np.float32) / 10.0
        Xc = np.tile(ctx_stats, (self.B, 1))

        # `y` is the latent (noise-free, current-round) path loss and is the
        # evaluation target.  `y_obs` is what a labelled probe actually
        # reports: the same quantity plus the residual fading of the reports.
        # All training and calibration uses `y_obs`.
        if y_obs_override is not None:
            y_obs = np.asarray(y_obs_override, np.float32)
        elif self.scfg.probe_label_noise:
            rng_l = np.random.default_rng(self.seed * 1000 + 977 + label_seed)
            y_obs = (y + rng_l.normal(0.0, self.rcfg.meas_noise_db, y.shape)
                     ).astype(np.float32)
        else:
            y_obs = y.copy()
        return dict(Xq=Xq, Xn=Xn, Xe=Xe, Xc=Xc, Xg=Xg, prior=prior, y=y,
                    y_obs=y_obs, r_obs=(y_obs - prior).astype(np.float32),
                    r=(y - prior).astype(np.float32),
                    nbr=np.tile(nbr, (self.B, 1)),
                    nbr_d=np.tile(nbr_d, (self.B, 1)),
                    bs_id=np.repeat(np.arange(self.B), len(q.li)),
                    xyz=np.tile(q.xyz, (self.B, 1)).astype(np.float32),
                    li=np.tile(q.li, self.B))

    def table_for(self, q):
        """Build a feature table for an arbitrary set of query locations."""
        return self._table(q, label_seed=99)

    def report_table(self):
        """Reports used as their own training targets (no labelled probes).

        Each report is a query whose neighbours are the *other* reports; its
        label is the aged, noisy report itself, re-ordered to (site-major)
        table rows.  This is the zero-probe training budget of Sec. VI.
        """
        yo = self.y_meas.T.reshape(-1)          # site-major, matches _table
        return self._table(self.meas, exclude_self=True, y_obs_override=yo)

    # ------------------------------------------------------------------ #
    def summary(self):
        v = self.gt.validate_against_3gpp()
        s = self.gt.stats()
        c = self.city.summary()
        return dict(env=self.env, seed=self.seed,
                    alpha_star=c["realised_alpha"],
                    h_mean=c["mean_building_height"],
                    h_p95=c["p95_building_height"],
                    h_max=c["max_building_height"],
                    los_pct=100 * v["los_fraction"],
                    bias_los=v["bias_los_db"], bias_nlos=v["bias_nlos_db"],
                    prior_rmse=s["resid_rmse_db"],
                    n_meas=len(self.meas.li),
                    n_bs=self.B)


def dims(k=12):
    return dict(q=len(ds.QUERY_FEATS), n=len(ds.NODE_FEATS),
                e=len(ds.EDGE_FEATS), c=4, g=len(ds.GEOM_FEATS), k=int(k))


if __name__ == "__main__":
    import time
    from citygen import STUDY_ENVS
    for env in STUDY_ENVS:
        t0 = time.time()
        sc = Scenario(env, seed=1)
        s = sc.summary()
        tt = sc.tables["test"]
        print(f"  {env:9s} priorRMSE={s['prior_rmse']:.2f} dB  "
              f"|r| mean={np.abs(tt['r']).mean():.2f}  "
              f"Xq{tt['Xq'].shape} Xn{tt['Xn'].shape} Xe{tt['Xe'].shape} "
              f"[{time.time()-t0:.1f}s]")
    print("dims:", dims())
