"""Metrics of the example application (manuscript Sect. 6): zones, E1 reflector metrics,
E2 event tracking and age span, E3 seismic summary statistics, Morris screening."""
from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import hilbert

from .pipeline import depth_to_twt


# ----------------------------------------------------------------- zones
def zones(em, interior_tol_m=15.0, margin_width_m=300.0, min_thickness_m=10.0, basin_tol_m=None):
    """Trace classes: 0 none, 1 interior, 2 margin, 3 slope, and 4 basin if ``basin_tol_m`` is
    given (slope traces whose carbonate top lies within ``basin_tol_m`` of the deepest top).
    Without ``basin_tol_m``, class 3 contains slope and basin (as used by E3)."""
    top = em.layer_top_depth.min(axis=0)
    base = em.layer_base_depth[0]
    thick = base - top
    interior = (top - em.platform_top_depth) < interior_tol_m
    cls = np.zeros(len(em.x), int)
    cls[interior] = 1
    xi = em.x[interior]
    if xi.size:
        dist = np.minimum(np.abs(em.x - xi.min()), np.abs(em.x - xi.max()))
        margin = (~interior) & (dist <= margin_width_m) & (thick > min_thickness_m)
        cls[margin] = 2
        cls[(~interior) & (~margin) & (thick > min_thickness_m)] = 3
        if basin_tol_m is not None:
            cls[(cls == 3) & (top > np.nanmax(top) - basin_tol_m)] = 4
    return cls


def sample_at(ts, twt, half_window_s=0.004, absolute=True):
    """Largest-magnitude amplitude within +-window of twt; twt is (n, ntr)."""
    twt = np.atleast_2d(twt)
    dt = ts.t[1] - ts.t[0]
    offs = np.arange(-int(round(half_window_s / dt)), int(round(half_window_s / dt)) + 1)
    out = np.full(twt.shape, np.nan)
    for j in range(twt.shape[1]):
        k = np.round(twt[:, j] / dt).astype(int)
        idx = np.clip(k[:, None] + offs[None, :], 0, ts.data.shape[0] - 1)
        seg = ts.data[idx, j]
        v = seg[np.arange(len(k)), np.argmax(np.abs(seg), axis=1)]
        out[:, j] = np.abs(v) if absolute else v
    return out


def platform_window(ts, em, cls_mask, start_s=0.010):
    """Boolean (nt, ntr) mask of the carbonate interval below the top on selected traces."""
    top = depth_to_twt(em, em.layer_top_depth.min(axis=0)[None, :])[0]
    base = depth_to_twt(em, em.layer_base_depth[0][None, :])[0]
    T = ts.t[:, None]
    return (T > top[None, :] + start_s) & (T < base[None, :] - 0.010) & cls_mask[None, :]


# ----------------------------------------------------------------- E1
def e1_metrics(ts, em, fs_layers, cls):
    """Reflection strength and lateral continuity along flooding-surface time-lines."""
    interior = cls == 1
    win = platform_window(ts, em, interior)
    rms = float(np.sqrt(np.mean(ts.data[win] ** 2))) if win.any() else np.nan
    depths = em.layer_top_depth[np.asarray(fs_layers, int)]
    twt = depth_to_twt(em, depths)
    # evaluate only inside the same window as the RMS (10 ms below the top, 10 ms above the
    # base), so that the top and base reflections of the carbonate never enter the metric
    top = depth_to_twt(em, em.layer_top_depth.min(axis=0)[None, :])[0]
    base = depth_to_twt(em, em.layer_base_depth[0][None, :])[0]
    inside = (twt > top[None, :] + 0.010) & (twt < base[None, :] - 0.010)
    amp = np.where(inside, sample_at(ts, twt), np.nan)[:, interior]
    ok = np.isfinite(amp)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        strength = np.nanmean(amp, axis=1) / rms
        continuity = np.where(ok.any(axis=1), (amp > rms).sum(axis=1) / np.maximum(ok.sum(axis=1), 1), np.nan)
    keep = ok.mean(axis=1) >= 0.5 if amp.size else np.zeros(0, bool)    # surface inside on half the traces
    strength, continuity = np.where(keep, strength, np.nan), np.where(keep, continuity, np.nan)
    return {"strength": strength, "continuity": continuity, "rms": rms,
            "mean_strength": float(np.nanmean(strength)), "mean_continuity": float(np.nanmean(continuity))}


def e1_reference_horizons(ts, em, fs_layers, cls, n=60, exclude_m=5.0, margin_m=10.0):
    """Same metrics as ``e1_metrics`` along reference horizons that are not flooding surfaces.

    Reference horizons are time-lines chosen uniformly in depth (not in time, which would
    over-sample hiatal surfaces) between the platform top and base, using the median depth
    of each time-line over the interior traces.  Time-lines within ``exclude_m`` of a
    flooding surface are skipped.  Returns the strength and continuity of each horizon.
    """
    interior = cls == 1
    if not interior.any():
        return {"strength": np.array([]), "continuity": np.array([])}
    dmed = np.median(em.layer_top_depth[:, interior], axis=1)          # (nl,), decreasing with k
    d_fs = dmed[np.asarray(fs_layers, int)]
    lo, hi = np.nanmin(dmed) + margin_m, np.nanmax(dmed) - margin_m
    ks = []
    for d in np.linspace(lo, hi, n):
        if d_fs.size and np.min(np.abs(d_fs - d)) < exclude_m:
            continue
        ks.append(int(np.nanargmin(np.abs(dmed - d))))
    ks = np.unique(ks)
    m = e1_metrics(ts, em, ks, cls)
    return {"strength": m["strength"], "continuity": m["continuity"], "layers": ks}


# ----------------------------------------------------------------- event tracking (E2)
def pick_peaks(ts, mask, rel_threshold=0.5):
    d = np.where(mask, ts.data, 0.0)
    thr = rel_threshold * np.sqrt(np.mean(ts.data[mask] ** 2)) if mask.any() else np.inf
    pk = np.zeros_like(d, bool)
    pk[1:-1] = (d[1:-1] > d[:-2]) & (d[1:-1] >= d[2:]) & (d[1:-1] > thr)
    tr = np.zeros_like(d, bool)
    tr[1:-1] = (d[1:-1] < d[:-2]) & (d[1:-1] <= d[2:]) & (d[1:-1] < -thr)
    return pk, tr


def track_events(picks, max_jump=1, min_traces=8):
    """Greedy left-to-right linking of picks into events (lists of (it, ix))."""
    nt, nx = picks.shape
    events, active = [], {}
    for ix in range(nx):
        rows = np.flatnonzero(picks[:, ix])
        new_active, used = {}, set()
        for it in rows:
            best = None
            for d in range(0, max_jump + 1):
                for cand in (it - d, it + d):
                    if cand in active and cand not in used:
                        best = cand
                        break
                if best is not None:
                    break
            if best is not None:
                ev = active[best]
                used.add(best)
            else:
                ev = []
                events.append(ev)
            ev.append((it, ix))
            new_active[it] = ev
        active = new_active
    return [np.array(e) for e in events if len(e) >= min_traces]


def age_section(ts, em):
    from .seismic import depth_to_time
    return depth_to_time(em, em.fields["age"], ts.t[1] - ts.t[0], nearest=True)


def event_age_stats(events, ages, layer_dt_Myr, cls=None, min_points=5):
    """Per event and zone: age span (kyr, 5-95 percentile range), the fraction of points
    within one output interval of the segment's median age, and the resolution floor, i.e.
    the median age range covered by one time sample along the event (kyr).
    Rows: (span, frac, zone, n, floor)."""
    A = ages.data
    # age range of one time sample: half the age difference between the neighbouring samples
    per_sample = np.full(A.shape, np.nan)
    per_sample[1:-1] = 0.5 * np.abs(A[2:] - A[:-2])
    rows = []
    for ev in events:
        a = A[ev[:, 0], ev[:, 1]]
        fl = per_sample[ev[:, 0], ev[:, 1]]
        z = cls[ev[:, 1]] if cls is not None else np.zeros(len(ev), int)
        for zone in np.unique(z):
            sel = (z == zone) & np.isfinite(a)
            if zone == 0 or sel.sum() < min_points:
                continue
            aa = a[sel]
            span = (np.percentile(aa, 95) - np.percentile(aa, 5)) * 1e3
            frac = float(np.mean(np.abs(aa - np.median(aa)) <= layer_dt_Myr))
            floor = float(np.nanmedian(fl[sel])) * 1e3 if np.isfinite(fl[sel]).any() else np.nan
            rows.append((span, frac, int(zone), int(sel.sum()), floor))
    return np.array(rows) if rows else np.zeros((0, 5))


# ----------------------------------------------------------------- E3 statistics
STAT_NAMES = ["top-carbonate amplitude", "continuous intra-platform reflectors",
              "platform RMS amplitude", "platform width in TWT", "margin dip",
              "internal dominant frequency", "high-amplitude continuous facies"]


def summary_statistics(ts, em):
    cls = zones(em)
    interior = cls == 1
    top_d = em.layer_top_depth.min(axis=0)
    top_t = depth_to_twt(em, top_d[None, :])[0]
    s1 = float(np.nanmean(sample_at(ts, top_t[None, :], absolute=False)[0, interior]))
    win = platform_window(ts, em, interior)
    pk, _ = pick_peaks(ts, win)
    ev = track_events(pk, min_traces=max(4, int(0.5 * interior.sum())))
    s2 = float(len(ev))
    s3 = float(np.sqrt(np.mean(ts.data[win] ** 2))) if win.any() else np.nan
    shallow = top_t < top_t[interior].min() + 0.020 if interior.any() else np.zeros_like(interior)
    s4 = float(shallow.sum() * (em.x[1] - em.x[0]))
    winm = platform_window(ts, em, (cls == 2) | (cls == 3), start_s=0.0)
    pkm, _ = pick_peaks(ts, winm)
    evm = track_events(pkm, max_jump=3, min_traces=4)
    dt = ts.t[1] - ts.t[0]
    dips = [abs(np.polyfit(e[:, 1] * (em.x[1] - em.x[0]) * 1e-3, e[:, 0] * dt * 1e3, 1)[0]) for e in evm]
    s5 = float(np.median(dips)) if dips else 0.0                      # ms per km
    seg = np.where(win, ts.data, 0.0)
    spec = np.abs(np.fft.rfft(seg, axis=0)).mean(axis=1)
    f = np.fft.rfftfreq(seg.shape[0], dt)
    s6 = float(f[np.argmax(spec[1:]) + 1])
    env = np.abs(hilbert(ts.data, axis=0))
    cont = np.zeros_like(env)
    cont[:, 1:-1] = np.minimum(np.abs(np.sign(ts.data[:, :-2]) + np.sign(ts.data[:, 2:])), 1)
    hi = (env > np.median(env[win])) & (cont > 0) & win if win.any() else win
    s7 = float(hi.sum() / max(win.sum(), 1))
    return np.array([s1, s2, s3, s4, s5, s6, s7])


# ----------------------------------------------------------------- Morris screening
def morris_trajectories(k, r, levels=4, seed=0):
    rng = np.random.default_rng(seed)
    delta = levels / (2.0 * (levels - 1))
    grid = np.arange(levels // 2) / (levels - 1)
    out = []
    for _ in range(r):
        x = rng.choice(grid, size=k)
        traj = [x.copy()]
        for i in rng.permutation(k):
            x = x.copy()
            x[i] = x[i] + delta if x[i] + delta <= 1 else x[i] - delta
            traj.append(x)
        out.append(np.array(traj))
    return out, delta


def morris_effects(trajs, Y, delta, return_counts=False):
    """Morris measures from trajectories.

    ``Y`` holds one (k+1, nstat) array of outputs per trajectory.  The elementary
    effect of a step is stored as the output change over one Morris step of size
    ``delta`` (in units of the normalized range), i.e. ``(dy / dx) * delta``.
    Steps touching a failed evaluation (NaN) are skipped statistic by statistic.
    Returns mu_star and sigma, both (k, nstat), and optionally the number of
    effects used, (k, nstat).
    """
    k = trajs[0].shape[1]
    ee = [[] for _ in range(k)]
    for X, y in zip(trajs, Y):
        y = np.asarray(y, float)
        for s in range(min(k, len(y) - 1)):
            i = int(np.flatnonzero(X[s + 1] != X[s])[0])
            ee[i].append((y[s + 1] - y[s]) / (X[s + 1, i] - X[s, i]) * delta)
    nstat = np.asarray(Y[0]).shape[1]
    mu = np.full((k, nstat), np.nan)
    sd = np.full((k, nstat), np.nan)
    n = np.zeros((k, nstat), int)
    for i, e in enumerate(ee):
        if not e:
            continue
        e = np.array(e)
        n[i] = np.isfinite(e).sum(0)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mu[i] = np.nanmean(np.abs(e), 0)
            sd[i] = np.nanstd(e, 0)
    return (mu, sd, n) if return_counts else (mu, sd)
