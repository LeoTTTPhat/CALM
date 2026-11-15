"""
trainer.py -- Training / inference driver shared by PACT-RM and every neural
ablation, so that all variants see identical optimisation settings.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from models import (PACTRM, PACTConfig, Standardiser, mask_geometry,
                    masked_physics_inputs)


def _batches(n, bs, rng, shuffle=True):
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


class Learner:
    """Wraps a PACTRM instance with standardisation and the target convention.

    When `use_physics` is true the network regresses the residual w.r.t. the
    3GPP prior; otherwise it regresses the path loss directly.  This is the
    single switch that separates *physics anchoring* from *physics as a
    penalty* (`physics_penalty > 0`) and from no physics at all.
    """

    def __init__(self, dims, cfg: PACTConfig, seed=0, device="cpu"):
        torch.manual_seed(seed)
        self.cfg = cfg
        self.device = device
        self.net = PACTRM(dims["q"], dims["n"], dims["e"], dims["c"], cfg,
                          dg=dims.get("g", 0),
                          k_nbr=dims.get("k", 12)).to(device)
        self.std = Standardiser()
        self.y_offset = 0.0
        self.y_scale = 1.0
        self.n_params = sum(p.numel() for p in self.net.parameters())

    # ------------------------------------------------------------------ #
    def _prep(self, tab, fit=False):
        t = masked_physics_inputs(tab, self.cfg.use_physics,
                                  self.cfg.prior_input)
        if not self.cfg.use_geom_feats:
            t = mask_geometry(t)
        if fit:
            self.std.fit(t)
        z = self.std.apply(t)
        z["D"] = np.asarray(tab["nbr_d"], np.float32)
        z["P"] = np.asarray(tab["prior"] - self.y_offset, np.float32)
        return {k: torch.as_tensor(v, device=self.device) for k, v in z.items()}

    def _fwd(self, Z, sl):
        return self.net(Z["Xq"][sl], Z["Xn"][sl], Z["Xe"][sl], Z["Xc"][sl],
                        Z["Xg"][sl], D=Z["D"][sl], prior=Z["P"][sl])

    def _target(self, tab):
        """Regression target.

        Anchored: the residual w.r.t. the 3GPP prior.  Unanchored: the path
        loss *centred* on the training mean.  Centring is what makes the
        comparison fair -- an unanchored learner forced to reproduce a
        ~100 dB offset would lose for a trivial reason -- and it is also the
        target Theorem 1 analyses (the constant predictor is free).
        """
        # Training and validation use the *observed* probe labels.
        y = tab.get("y_obs", tab["y"])
        r = tab.get("r_obs", tab["r"])
        if self.cfg.use_physics:
            return torch.as_tensor(r, device=self.device)
        return torch.as_tensor((y - self.y_offset) / self.y_scale,
                               device=self.device)

    def fit(self, train, val=None, epochs=80, bs=1024, lr=3e-3, wd=1e-4,
            patience=6, beta_nll=0.2, verbose=False, seed=0):
        rng = np.random.default_rng(seed)
        y_tr = train.get("y_obs", train["y"])
        if not self.cfg.use_physics and self.y_offset == 0.0:
            self.y_offset = float(np.mean(y_tr))
            if self.cfg.target_std:
                # Same dispersion as the anchored residual target, so that
                # any remaining gap is not a matter of target scale.
                r_tr = train.get("r_obs", train["r"])
                self.y_scale = float(np.std(y_tr) / max(np.std(r_tr), 1e-6))
        Z = self._prep(train, fit=True)
        Y = self._target(train)
        prior = torch.as_tensor(train["prior"] - self.y_offset,
                                device=self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        Zv = Yv = None
        if val is not None:
            Zv, Yv = self._prep(val), self._target(val)

        best, best_state, bad = np.inf, None, 0
        hist = []
        n = len(Y)
        for ep in range(epochs):
            self.net.train()
            tot = 0.0
            for bidx in _batches(n, bs, rng):
                b = torch.as_tensor(bidx, device=self.device)
                out, s, _ = self._fwd(Z, b)
                loss = F.mse_loss(out, Y[b])
                if s is not None:
                    resid = (Y[b] - out).detach()
                    loss = loss + beta_nll * (s + resid ** 2 / (2 * torch.exp(2 * s))).mean()
                if self.cfg.physics_penalty > 0 and not self.cfg.use_physics:
                    loss = loss + self.cfg.physics_penalty * F.mse_loss(out, prior[b])
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
                tot += float(loss.detach()) * len(bidx)
            sched.step()
            tr = tot / n
            if Zv is not None and (ep % 2 == 1 or ep == epochs - 1):
                v = float(F.mse_loss(self.predict_raw(Zv), Yv))
                hist.append((ep, tr, v))
                if v < best - 1e-4:
                    best, bad = v, 0
                    best_state = {k: t.detach().clone()
                                  for k, t in self.net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= patience:
                        break
                if verbose and ep % 10 == 0:
                    print(f"    ep{ep:3d} train={tr:.3f} val={v:.3f}")
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.history = hist
        return self

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_raw(self, Z, bs=8192, mc_dropout=False):
        """Raw network output.  `mc_dropout` keeps dropout active, which is
        what makes an MC-dropout ensemble stochastic; without it every pass is
        identical and the baseline degenerates to zero variance."""
        self.net.train(mc_dropout)
        outs = []
        n = len(Z["Xq"])
        for i in range(0, n, bs):
            o, _, _ = self._fwd(Z, slice(i, i + bs))
            outs.append(o)
        return torch.cat(outs)

    @torch.no_grad()
    def predict(self, tab, bs=8192):
        """Returns (path-loss prediction in dB, heteroscedastic log-scale)."""
        self.net.eval()
        Z = self._prep(tab)
        outs, ss = [], []
        n = len(Z["Xq"])
        for i in range(0, n, bs):
            o, s, _ = self._fwd(Z, slice(i, i + bs))
            outs.append(o.cpu().numpy())
            ss.append((s.cpu().numpy() if s is not None else np.zeros(len(o))))
        yhat = np.concatenate(outs)
        yhat = (tab["prior"] + yhat if self.cfg.use_physics
                else yhat * self.y_scale + self.y_offset)
        return yhat.astype(np.float32), np.concatenate(ss).astype(np.float32)
