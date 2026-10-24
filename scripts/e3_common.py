"""Load E3 results and derive Morris measures and detectability indices."""
import warnings

import numpy as np
from common import ROOT
from carboseis.analysis import morris_effects


def load():
    path = ROOT / "build" / "e3" / "results.npz"
    if not path.exists():
        raise SystemExit("E3 has not been run yet: run 'make e3' (scripts/e3_run.py) first.")
    d = np.load(path, allow_pickle=True)
    r_planned = int(d["r_planned"]) if "r_planned" in d.files else None
    if "trajs" not in d.files:
        raise SystemExit(f"E3 is still in its first phase: {len(d['seeds'])} seed replicates are done "
                         "but no Morris trajectory yet. Wait for the 'E3 finished' message of "
                         "e3_run.py (see build/e3/status.txt), then run 'make figures-e3'.")
    trajs, Y, delta = d["trajs"], d["Y"], float(d["delta"])
    partial = r_planned is not None and len(trajs) < r_planned
    if partial:
        print(f"WARNING: E3 is incomplete ({len(trajs)} of {r_planned} trajectories); "
              "the figures are marked PARTIAL and must not be used in the paper.")
    mu, sigma, n_eff = morris_effects(list(trajs), list(Y), delta, return_counts=True)
    seeds = d["seeds"]
    ok = ~np.isnan(seeds).all(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        var_seed = np.nanvar(seeds[ok], axis=0, ddof=1)
    # mu is the mean absolute output change over one Morris step of size delta (in units
    # of the normalized range); for a parameter uniform over its range the first-order
    # variance is (slope)^2 / 12 with slope = mu / delta.
    var_j = (mu / delta) ** 2 / 12.0
    with np.errstate(divide="ignore", invalid="ignore"):
        D = var_j / var_seed[None, :]
    n_fail = int(np.isnan(Y).all(axis=2).sum())
    D_lo, D_hi = bootstrap_D(trajs, Y, seeds[ok], delta)
    dom = failure_domain(trajs, Y, d["lo"], d["hi"])
    return dict(params=list(d["params"]), stats=list(d["stats"]), mu=mu, sigma=sigma, D=D,
                n_eff=n_eff, n_traj=len(trajs), n_seeds=int(ok.sum()), n_failed=n_fail,
                n_evals=int(Y.shape[0] * Y.shape[1]), engine=str(d["engine"]), seeds=seeds[ok],
                delta=delta, partial=partial, r_planned=r_planned, D_lo=D_lo, D_hi=D_hi,
                robust=D_lo > 1.0, domain=dom)


def detectability(trajs, Y, seeds, delta):
    mu, _, _ = morris_effects(list(trajs), list(Y), delta, return_counts=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        vs = np.nanvar(seeds, axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return (mu / delta) ** 2 / 12.0 / vs[None, :]


def bootstrap_D(trajs, Y, seeds, delta, B=2000, seed=1):
    """5th and 95th percentiles of D when trajectories and seed replicates are resampled with
    replacement.  A parameter is called robustly detectable when the 5th percentile exceeds one."""
    rng = np.random.default_rng(seed)
    nt, ns = len(trajs), len(seeds)
    Ds = []
    for _ in range(B):
        ti, si = rng.integers(0, nt, nt), rng.integers(0, ns, ns)
        Ds.append(detectability(trajs[ti], Y[ti], seeds[si], delta))
    Ds = np.array(Ds)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(Ds, 5, axis=0), np.nanpercentile(Ds, 95, axis=0)


def failure_domain(trajs, Y, lo, hi, transport_min=3.0, growth_min=2000.0):
    """Failed CarboKitten runs by distinct combination of the forward-run parameters, split into the
    corner with high transport and high euphotic growth and the rest of the box."""
    X = trajs.reshape(-1, trajs.shape[-1])
    V = lo + X * (hi - lo)
    fail = np.isnan(Y.reshape(-1, Y.shape[-1])).all(axis=1)
    combos = {}
    for v, f in zip(V[:, :5], fail):
        combos.setdefault(tuple(np.round(v, 6)), []).append(bool(f))
    corner = {k: any(f) for k, f in combos.items() if k[4] >= transport_min and k[3] >= growth_min}
    rest = {k: any(f) for k, f in combos.items() if not (k[4] >= transport_min and k[3] >= growth_min)}
    return dict(n_combos=len(combos), n_failed=sum(any(f) for f in combos.values()),
                corner=len(corner), corner_failed=sum(corner.values()),
                rest=len(rest), rest_failed=sum(rest.values()),
                transport_min=transport_min, growth_min=growth_min)
