"""Base-case numbers quoted in Sect. 6.2, by zone (interior, margin, slope) -> tables/base_case.csv.

Zones are those of the examples (analysis.zones).  Layer quantities are weighted by
compacted rock volume.  The top-carbonate reflection is measured on the depth model
(reflection coefficient) and on both seismic levels (signed amplitude at the top).
"""
import csv

import numpy as np

from common import TAB, TEXTURE_LABELS, chain, grid, scenario
from carboseis.analysis import platform_window, sample_at, zones
from carboseis.pipeline import depth_to_twt

sc = scenario()
g = grid(sc)
res = chain(sc, grid_obj=g)
em, L = res.em, res.layers
cls_tr = zones(em, basin_tol_m=20.0)
# zone of each CarboKitten column = zone of the nearest seismic trace
it = np.clip(np.round((g.x - em.x[0]) / (em.x[1] - em.x[0])).astype(int), 0, len(em.x) - 1)
cls_col = cls_tr[it]
w = L["hc"]                                                   # compacted thickness weights
tau_c = sc["diagenesis"]["tau_crit_Myr"]


def wmean(v, m):
    ww = w * m
    return float((v * ww).sum() / ww.sum()) if ww.sum() > 0 else np.nan


def wmedian(v, m):
    sel = (m & (w > 0)).ravel()
    if not sel.any():
        return np.nan
    vv, ww = v.ravel()[sel], w.ravel()[sel]
    o = np.argsort(vv)
    c = np.cumsum(ww[o])
    return float(vv[o][np.searchsorted(c, 0.5 * c[-1])])


# top-carbonate reflection coefficient on the depth model (5 m above / below the top)
top_d = em.layer_top_depth.min(axis=0)
iz = np.clip((top_d / em.dz).astype(int), 5, em.vp.shape[0] - 6)
Z = em.vp * em.rho
za = np.array([Z[i - 5:i - 1, j].mean() for j, i in enumerate(iz)])
zb = np.array([Z[i + 1:i + 5, j].mean() for j, i in enumerate(iz)])
rc_top = (zb - za) / (zb + za)
top_t = depth_to_twt(em, top_d[None, :])[0]
amp1 = sample_at(res.l1, top_t[None, :], absolute=False)[0]
amp2 = sample_at(res.l2, top_t[None, :], absolute=False)[0]

# dip of the carbonate top in depth (degrees), from the seismic-trace grid
dip_top = np.degrees(np.arctan(np.abs(np.gradient(top_d, em.x))))

rows = {}
for zc, zn in ((1, "interior"), (2, "margin"), (3, "slope"), (4, "basin")):
    col = (cls_col == zc)[None, :] & np.ones_like(w, bool)
    tr = cls_tr == zc
    r = {"columns (50 m)": int((cls_col == zc).sum()),
         "uncompacted thickness, median (m)": float(np.median(g.H.sum(0)[cls_col == zc])) if (cls_col == zc).any() else np.nan,
         "compacted thickness, median (m)": float(np.median(w.sum(0)[cls_col == zc])) if (cls_col == zc).any() else np.nan,
         "depositional water depth, median (m)": wmedian(g.wd, col),
         "volume deposited above fair-weather wave base": wmean((g.wd < sc["texture"]["fair_weather_wave_base_m"]).astype(float), col),
         f"volume with exposure > tau_crit ({tau_c} Myr)": wmean((g.exposure > tau_c).astype(float), col),
         "porosity after diagenesis, mean": wmean(L["phi"], col),
         "porosity before diagenesis, mean": wmean(L["phi_mech"], col),
         "Vp, mean (m/s)": wmean(L["vp"], col),
         "density, mean (g/cm3)": wmean(L["rho"], col),
         "impedance, mean (m/s g/cm3)": wmean(L["vp"] * L["rho"], col),
         "top reflection coefficient, median": float(np.median(rc_top[tr])) if tr.any() else np.nan,
         "top reflection positive (hard kick), fraction of traces": float(np.mean(rc_top[tr] > 0)) if tr.any() else np.nan,
         "top amplitude Level 1, median": float(np.nanmedian(amp1[tr])) if tr.any() else np.nan,
         "top amplitude Level 2 / Level 1, median": float(np.nanmedian(amp2[tr] / amp1[tr])) if tr.any() else np.nan,
         "dip of carbonate top, 90th percentile (deg)": float(np.percentile(dip_top[tr], 90)) if tr.any() else np.nan}
    q = res.q * w[:, None, :]
    tot = q[:, :, cls_col == zc].sum()
    for t, name in enumerate(TEXTURE_LABELS):
        r[f"texture fraction: {name}"] = float(q[:, t, cls_col == zc].sum() / tot) if tot > 0 else np.nan
    fv = (g.h.sum(0))[:, cls_col == zc].sum(1) if (cls_col == zc).any() else np.zeros(g.h.shape[1])
    for f, name in enumerate(g.facies_names):
        r[f"facies fraction: {name}"] = float(fv[f] / fv.sum()) if fv.sum() > 0 else np.nan
    rows[zn] = r
win = platform_window(res.l1, em, cls_tr == 1)
seg = np.where(win, res.l1.data, 0.0)
spec = np.abs(np.fft.rfft(seg, axis=0)).mean(axis=1)
f = np.fft.rfftfreq(seg.shape[0], res.l1.t[1] - res.l1.t[0])
peak_f = float(f[np.argmax(spec[1:]) + 1])

keys = list(rows["interior"])
with open(TAB / "base_case.csv", "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["quantity", "interior", "margin", "slope", "basin"])
    for k in keys:
        wr.writerow([k] + [f"{rows[z][k]:.4g}" for z in ("interior", "margin", "slope", "basin")])
    wr.writerow(["internal dominant frequency, Level 1 (Hz)", f"{peak_f:.1f}", "", "", ""])
    wr.writerow(["final water depth at platform centre (m)",
                 f"{float(g.sea_level[-1] - g.z_top[-1].max()):.1f}", "", "", ""])
print(open(TAB / "base_case.csv").read())
