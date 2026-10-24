"""E1 diagnostics: why flooding surfaces do or do not stand out -> tables/e1_diagnostics.csv.

(a) Layer scale, before any wavelet: reflection coefficient across each flooding surface
    (positive when impedance increases downward),
    averaged over 3 m of compacted rock on either side, against the same contrast at
    horizons placed at random stratigraphic heights; sign of the flooding-surface contrast;
    fraction of flooding surfaces that fall on a hiatus; change of depositional water depth
    across the flooding surfaces.
(b) Seismic scale: percentile of the median flooding surface among the reference horizons
    at 30 and 60 Hz (point-spread-function imaging).
(c) Meteoric profile: the same percentile for a step-like and an exponential meteoric
    profile and meteoric-zone depths of 5, 10 and 20 m (30 Hz).
No new CarboKitten run is needed.
"""
import csv

import numpy as np

from common import TAB, chain, grid, scenario
from e1_common import SCENARIOS
from carboseis.analysis import e1_metrics, e1_reference_horizons, zones
from carboseis.grid import flooding_surface_frames

rows = []


def add(part, scen, setting, metric, value):
    rows.append([part, scen, setting, metric, value])
    print(f"{part} {scen:3s} {setting:22s} {metric:38s} {value:.4f}", flush=True)


def interior_columns(g, em):
    cls = zones(em)
    it = np.clip(np.round((g.x - em.x[0]) / (em.x[1] - em.x[0])).astype(int), 0, len(em.x) - 1)
    return np.flatnonzero(cls[it] == 1)


def contrast(Z, H, k, ix, w=3.0):
    Hc = H[:, ix]
    c = np.cumsum(Hc)
    top_k = c[k]
    below = (c > top_k - w) & (c <= top_k) & (Hc > 0)
    above = (c - Hc >= top_k) & (c - Hc < top_k + w) & (Hc > 0)
    if not below.any() or not above.any():
        return np.nan
    za = np.average(Z[above, ix], weights=Hc[above])
    zb = np.average(Z[below, ix], weights=Hc[below])
    return (zb - za) / (zb + za)          # seismic convention: positive when impedance increases downward


def seismic_percentiles(res, g):
    cls = zones(res.em)
    fs = flooding_surface_frames(g.t, g.sea_level)
    m = e1_metrics(res.l2, res.em, fs, cls)
    r = e1_reference_horizons(res.l2, res.em, fs, cls)
    s, c = np.nanmedian(m["strength"]), np.nanmedian(m["continuity"])
    rs, rc = r["strength"][np.isfinite(r["strength"])], r["continuity"][np.isfinite(r["continuity"])]
    return float(np.mean(rs < s)), float(np.mean(rc < c))


# (a) layer scale and (b) frequency
for name in ("DC", "D0", "D1", "D2"):
    ov, mode = SCENARIOS[name]
    sc = scenario(ov)
    g = grid(sc)
    res = chain(sc, grid_obj=g, property_mode=mode)
    L = res.layers
    Z, H = L["vp"] * L["rho"], L["hc"]
    cols = interior_columns(g, res.em)
    fs = flooding_surface_frames(g.t, g.sea_level)
    fsr = np.array([[contrast(Z, H, k, ix) for ix in cols] for k in fs])
    rng = np.random.default_rng(0)
    ref = []
    for ix in cols:
        c = np.cumsum(H[:, ix])
        for h in rng.uniform(5.0, c[-1] - 5.0, 40):
            ref.append(contrast(Z, H, int(np.searchsorted(c, h)), ix))
    ref = np.abs(np.array(ref))
    ref = ref[np.isfinite(ref)]
    a_fs = float(np.nanmedian(np.abs(fsr)))
    add("a", name, "layer scale", "median |R| at flooding surfaces", a_fs)
    add("a", name, "layer scale", "median |R| at reference horizons", float(np.median(ref)))
    add("a", name, "layer scale", "flooding-surface percentile", float(np.mean(ref < a_fs)))
    add("a", name, "layer scale", "mean reflection coefficient at flooding surfaces (seismic sign)", float(np.nanmean(fsr)))
    if name == "D0":
        hiatus = np.mean([np.median(g.H[max(k - 2, 0):k + 3][:, cols].sum(0)) < 0.05 for k in fs])
        wd = g.wd[:, cols]
        dwd = np.nanmedian([np.median(wd[k + 1:k + 6]) - np.median(wd[max(k - 6, 0):k - 1]) for k in fs])
        add("a", "all", "layer scale", "fraction of flooding surfaces on a hiatus", float(hiatus))
        add("a", "all", "layer scale", "water-depth change across flooding surfaces (m)", float(dwd))
    for f0 in (30.0, 60.0):
        r60 = chain(sc, grid_obj=g, property_mode=mode, f0=f0)
        ps, pc = seismic_percentiles(r60, g)
        add("b", name, f"{f0:.0f} Hz", "strength percentile", ps)
        add("b", name, f"{f0:.0f} Hz", "continuity percentile", pc)

# (c) meteoric profile and depth
for prof in ("step", "exponential"):
    for dm in (5.0, 10.0, 20.0):
        for name in ("D1", "D2"):
            ov = dict(SCENARIOS[name][0])
            ov.update({"grid.meteoric_profile": prof, "grid.meteoric_zone_m": dm})
            sc = scenario(ov)
            g = grid(sc)
            ps, pc = seismic_percentiles(chain(sc, grid_obj=g, property_mode="process"), g)
            add("c", name, f"{prof} {dm:.0f} m", "strength percentile", ps)
            add("c", name, f"{prof} {dm:.0f} m", "continuity percentile", pc)

with open(TAB / "e1_diagnostics.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["part", "scenario", "setting", "metric", "value"])
    for r in rows:
        w.writerow(r[:4] + [f"{r[4]:.4f}"])
