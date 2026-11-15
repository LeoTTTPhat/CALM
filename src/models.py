"""
models.py -- The PACT-RM estimator and its neural ablations.

PACT-RM predicts the *residual* of the standardised 3GPP TR 36.777 prior:

    PL_hat(x, b) = PL_prior(x, b) + r_theta(x, b)

r_theta is an edge-conditioned graph attention network over the k-nearest
crowdsensed reports, with a temporal gate that discounts aged reports.  A second
head predicts a log-scale s_theta used as the difficulty estimator of the
normalised conformal layer (it never affects the point prediction: the scale
head is trained on a detached mean).

Ablations are selected through `PACTConfig` flags so that every variant shares
one code path, which removes a common source of unfair comparison.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

AGE_COL = 5          # index of the "age" edge feature in EDGE_FEATS


@dataclass
class PACTConfig:
    d_model: int = 48
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.05
    use_physics: bool = True     # residual parameterisation + prior features
    use_graph: bool = True       # attend to neighbouring reports
    use_temporal: bool = True    # age gate on the attention logits
    use_context: bool = True     # cross-BS obstruction context
    heteroscedastic: bool = True
    use_oracle_geom: bool = False  # headroom reference only; needs exact 3-D map
    # Remove the covariates the standardised prior is itself a function of
    # (ranges, altitude, elevation angle).  This isolates *when* anchoring can
    # help: a residual parameterisation adds information only if the prior is
    # not already reconstructible from the learner's own inputs.
    use_geom_feats: bool = True
    physics_penalty: float = 0.0  # >0 reproduces a PINN-style soft anchor
    # ---- separating information from parameterisation ---------------------
    # Whether the prior-derived covariates (g, p_LoS, neighbour priors, the
    # edge prior difference and the residual-based cross-site context) are
    # inputs.  None follows `use_physics` (the original convention).
    prior_input: bool | None = None
    target_std: bool = False      # unanchored: standardise the target
    linear_bypass: bool = False   # unanchored: learnable a*g added to output
    # Aggregation over the k reports: 'attention' (CALM), 'mean', 'max',
    # 'kernel' (fixed exp(-d/50 m) weights) or 'concat' (distance-ordered
    # concatenation).  All variants see identical neighbours and features.
    aggregator: str = "attention"

    @property
    def has_prior_input(self) -> bool:
        return self.use_physics if self.prior_input is None else self.prior_input


def mlp(i, h, o, p=0.0):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, o))


class EdgeAttentionLayer(nn.Module):
    """One round of edge-conditioned attention with an explicit temporal gate."""

    def __init__(self, d, p, use_temporal=True, n_heads=4):
        super().__init__()
        assert d % n_heads == 0, "d_model must be divisible by n_heads"
        self.h = n_heads
        self.dh = d // n_heads
        # Message: one affine map of the concatenated (neighbour, edge) state.
        self.msg = nn.Sequential(nn.Linear(2 * d, d), nn.GELU())
        # Additive (GAT-style) multi-head attention scoring: three rank-H
        # projections instead of a full MLP over the 3d concatenation.
        self.a_q = nn.Linear(d, n_heads, bias=False)
        self.a_n = nn.Linear(d, n_heads, bias=False)
        self.a_e = nn.Linear(d, n_heads, bias=False)
        self.proj = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.use_temporal = use_temporal
        # per-head log-rate of the exponential staleness discount, learned
        self.log_tau = nn.Parameter(torch.zeros(n_heads))

    def forward(self, hq, hn, he, age, mask=None):
        # hq (N, d) ; hn, he (N, k, d) ; age (N, k)
        N, k, _ = hn.shape
        s = F.leaky_relu(self.a_q(hq)[:, None, :] + self.a_n(hn) + self.a_e(he),
                         0.2)                                        # (N, k, H)
        if self.use_temporal:
            s = s - F.softplus(self.log_tau)[None, None, :] * age[..., None]
        if mask is not None:
            s = s.masked_fill(~mask[..., None], float("-inf"))
        a = torch.softmax(s, dim=1)                                  # (N, k, H)
        m = self.msg(torch.cat([hn, he], dim=-1)).view(N, k, self.h, self.dh)
        agg = (a[..., None] * m).sum(1).reshape(N, -1)               # (N, d)
        return self.norm(hq + self.proj(agg)), a


class PoolLayer(nn.Module):
    """Query-conditioned set aggregation without learned attention.

    Messages depend on the query state, the neighbour state and the edge state
    exactly as in the attention layer; only the weighting differs."""

    def __init__(self, d, k, kind="mean"):
        super().__init__()
        self.kind = kind
        self.msg = nn.Sequential(nn.Linear(3 * d, d), nn.GELU())
        in_p = k * d if kind == "concat" else d
        self.proj = nn.Linear(in_p, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, hq, hn, he, dist=None):
        N, k, d = hn.shape
        m = self.msg(torch.cat([hn, he, hq[:, None, :].expand(-1, k, -1)], -1))
        if self.kind == "mean":
            agg = m.mean(1)
        elif self.kind == "max":
            agg = m.max(1).values
        elif self.kind == "kernel":
            w = torch.exp(-dist / 50.0)
            w = w / (w.sum(1, keepdim=True) + 1e-9)
            agg = (w[..., None] * m).sum(1)
        elif self.kind == "concat":
            agg = m.reshape(N, k * d)
        else:
            raise ValueError(self.kind)
        return self.norm(hq + self.proj(agg)), None


class PACTRM(nn.Module):
    def __init__(self, dq, dn, de, dc, cfg: PACTConfig, dg: int = 0,
                 k_nbr: int = 12):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        in_q = dq + (dc if cfg.use_context else 0) + (dg if cfg.use_oracle_geom else 0)
        self.enc_q = mlp(in_q, 2 * d, d, cfg.dropout)
        self.enc_n = mlp(dn, 2 * d, d, cfg.dropout)
        self.enc_e = mlp(de, 2 * d, d, cfg.dropout)
        if not cfg.use_graph:
            self.layers = nn.ModuleList()
        elif cfg.aggregator == "attention":
            self.layers = nn.ModuleList(
                [EdgeAttentionLayer(d, cfg.dropout, cfg.use_temporal, cfg.n_heads)
                 for _ in range(cfg.n_layers)])
        else:
            self.layers = nn.ModuleList(
                [PoolLayer(d, k_nbr, cfg.aggregator)
                 for _ in range(cfg.n_layers)])
        self.head = mlp(d, 2 * d, 1, cfg.dropout)
        self.scale = mlp(d, d, 1, cfg.dropout) if cfg.heteroscedastic else None
        self.bypass = nn.Parameter(torch.zeros(())) if cfg.linear_bypass else None

    def forward(self, Xq, Xn, Xe, Xc, Xg=None, D=None, prior=None):
        x = torch.cat([Xq, Xc], dim=-1) if self.cfg.use_context else Xq
        if self.cfg.use_oracle_geom and Xg is not None:
            x = torch.cat([x, Xg], dim=-1)
        hq = self.enc_q(x)
        attn = None
        if self.cfg.use_graph:
            hn, he = self.enc_n(Xn), self.enc_e(Xe)
            age = Xe[..., AGE_COL]
            for lay in self.layers:
                if self.cfg.aggregator == "attention":
                    hq, attn = lay(hq, hn, he, age)
                else:
                    hq, attn = lay(hq, hn, he, D)
        out = self.head(hq).squeeze(-1) * 10.0        # dB
        if self.bypass is not None and prior is not None:
            out = out + self.bypass * prior
        s = None
        if self.scale is not None:
            s = self.scale(hq.detach()).squeeze(-1)
        return out, s, attn


# ---------------------------------------------------------------------- #
class Standardiser:
    def __init__(self):
        self.stats = {}

    KEYS = ("Xq", "Xn", "Xe", "Xc", "Xg")

    def fit(self, tab):
        for k in self.KEYS:
            v = tab[k].reshape(-1, tab[k].shape[-1])
            self.stats[k] = (v.mean(0), v.std(0) + 1e-6)
        return self

    def apply(self, tab):
        out = {}
        for k in self.KEYS:
            m, s = self.stats[k]
            out[k] = ((tab[k] - m) / s).astype(np.float32)
        return out

    def state(self):
        return {k: (v[0].tolist(), v[1].tolist()) for k, v in self.stats.items()}

    def load(self, st):
        self.stats = {k: (np.asarray(a, np.float32), np.asarray(b, np.float32))
                      for k, (a, b) in st.items()}
        return self


GEOM_Q_COLS = (0, 1, 2, 3, 6, 7)     # log d3D, log d2D, h, sin(elev), dz, z_b
GEOM_N_COLS = (3, 4, 5)              # log d3D, h, sin(elev) of the neighbour


def mask_geometry(tab):
    """Blind the learner to the covariates the prior is a function of."""
    out = dict(tab)
    Xq = tab["Xq"].copy()
    for c in GEOM_Q_COLS:
        Xq[:, c] = 0.0
    Xn = tab["Xn"].copy()
    for c in GEOM_N_COLS:
        Xn[..., c] = 0.0
    out["Xq"], out["Xn"] = Xq, Xn
    return out


def masked_physics_inputs(tab, use_physics: bool, prior_input=None):
    """Remove the prior-derived covariates when the physics anchor is ablated.

    Columns 4 (prior) and 5 (LoS probability) of the query features and column
    1 (prior) of the node features are the only places the standardised model
    enters the network, so zeroing them yields a genuinely physics-free learner.
    """
    prior_input = use_physics if prior_input is None else prior_input
    if use_physics and prior_input:
        return tab
    out = dict(tab)
    Xq = tab["Xq"].copy(); Xn = tab["Xn"].copy()
    Xe = tab["Xe"].copy(); Xc = tab["Xc"].copy()
    if not use_physics:
        # The neighbour value channel lives in the learner's target space: an
        # unanchored learner observes raw reports, y_j/10 = r_j/10 +
        # 10 * (prior_j/100), not residuals.
        Xn[..., 0] = Xn[..., 0] + 10.0 * Xn[..., 1]
    if not prior_input:
        # Remove every prior-derived covariate: g and p_LoS at the query and
        # at the neighbours, the edge prior difference, and the cross-site
        # context, which is built from residuals and would otherwise smuggle
        # the prior back in.
        Xq[:, 4] = 0.0; Xq[:, 5] = 0.0
        Xn[..., 1] = 0.0; Xn[..., 2] = 0.0
        Xe[..., 6] = 0.0
        Xc[:] = 0.0
    out["Xq"], out["Xn"], out["Xe"], out["Xc"] = Xq, Xn, Xe, Xc
    return out


def to_torch(d, device="cpu"):
    return {k: torch.as_tensor(v, device=device) for k, v in d.items()}
