"""Lithofacies, compaction and rule-based diagenesis (manuscript Sects. 3.2-3.4).

All arrays are layer-wise, shape (nl, nx), or (nl, n_textures, nx) for texture
fractions.  Depths are burial depths below the sea floor in metres.
"""
from __future__ import annotations

import numpy as np


def _sigmoid(u):
    return 1.0 / (1.0 + np.exp(-np.clip(u, -50, 50)))


def depth_class_weights(wd, fwwb, swb, width):
    """Smooth weights of the shallow / mid / deep water-depth classes."""
    s1, s2 = _sigmoid((wd - fwwb) / width), _sigmoid((wd - swb) / width)
    return 1.0 - s1, s1 - s2, s2


def texture_fractions(p, wd, cfg):
    """Eq. (4): q[k, tau, x] = sum_f p[k, f, x] M[f, tau](wd[k, x])."""
    ws, wm, wdp = depth_class_weights(wd, cfg["fair_weather_wave_base_m"],
                                      cfg["storm_wave_base_m"], cfg["transition_width_m"])
    Ms, Mm, Md = (np.asarray(cfg[k], float) for k in ("shallow", "mid", "deep"))
    nf = p.shape[1]
    if Ms.shape[0] != nf:
        raise ValueError(f"texture membership has {Ms.shape[0]} rows but output has {nf} facies")
    q = (np.einsum("kfx,ft->ktx", p * ws[:, None, :], Ms)
         + np.einsum("kfx,ft->ktx", p * wm[:, None, :], Mm)
         + np.einsum("kfx,ft->ktx", p * wdp[:, None, :], Md))
    return q


def compact(H, q, cfg, burial_top):
    """Eq. (5): solid-conserving Athy compaction, solved from the top of each column down.

    H: depositional thickness (nl, nx); q: texture fractions; burial_top: burial
    depth (m below sea floor) of each column top.  Returns compacted thickness,
    mean porosity and mid-layer burial depth, all (nl, nx).
    """
    phi0_t = np.asarray(cfg["phi0"], float)
    c_t = np.asarray(cfg["c_per_km"], float) * 1e-3
    qs = q.sum(axis=1)
    w = np.where(qs[:, None, :] > 0, q / np.where(qs > 0, qs, 1)[:, None, :], 1.0 / len(phi0_t))
    phi0 = np.einsum("ktx,t->kx", w, phi0_t)
    c = np.einsum("ktx,t->kx", w, c_t)
    s = H * (1.0 - phi0)                                   # solid thickness
    nl, nx = H.shape
    hc = np.zeros_like(H)
    zmid = np.zeros_like(H)
    z1 = np.asarray(burial_top, float).copy()
    for k in range(nl - 1, -1, -1):                       # youngest layer first
        p0, ck, sk = phi0[k], c[k], s[k]
        z2 = z1 + sk / np.maximum(1.0 - p0 * np.exp(-ck * z1), 1e-6)
        for _ in range(6):                                 # Newton on the solid-thickness equation
            g = (z2 - z1) - p0 / ck * (np.exp(-ck * z1) - np.exp(-ck * z2)) - sk
            z2 = z2 - g / (1.0 - p0 * np.exp(-ck * z2))
        z2 = np.maximum(z2, z1)
        hc[k] = z2 - z1
        zmid[k] = 0.5 * (z1 + z2)
        z1 = z2
    with np.errstate(invalid="ignore", divide="ignore"):
        phi = np.where(hc > 1e-9, 1.0 - s / np.where(hc > 1e-9, hc, 1.0),
                       phi0 * np.exp(-c * zmid))
    return hc, np.clip(phi, 0.0, 0.95), zmid, s


def diagenesis(phi_mech, q, exposure, zmid, depth_below_platform_top, cfg):
    """Sect. 3.4: rule-based diagenesis.  Returns porosity per pore type
    (interparticle, stiff, crack) and mineral fractions (calcite, dolomite, clay)."""
    pt = np.asarray(cfg["pore_types"], float)               # (n_tex, 3)
    qs = q.sum(axis=1)
    w = np.where(qs[:, None, :] > 0, q / np.where(qs > 0, qs, 1)[:, None, :], 0.0)
    frac = np.einsum("ktx,tp->kpx", w, pt)
    frac /= np.maximum(frac.sum(axis=1, keepdims=True), 1e-12)
    phi_ip = phi_mech * frac[:, 0]
    phi_st = phi_mech * frac[:, 1]
    phi_cr = phi_mech * frac[:, 2]
    grainy = w[:, 0] + w[:, 1] + 0.5 * w[:, 2]               # boundstone + grainstone + half packstone
    if cfg.get("meteoric", True):
        m = exposure / (exposure + cfg["tau_crit_Myr"])
        phi_st = phi_st + cfg["dphi_mold"] * m * grainy
        phi_ip = np.maximum(phi_ip - cfg["dphi_cem"] * m * grainy, 0.0)
    if cfg.get("burial_cementation", True):
        excess = np.maximum(zmid - cfg["z_ch_m"], 0.0)
        phi_ip = phi_ip * np.exp(-cfg["c_ch_per_km"] * 1e-3 * excess)
    dol = np.zeros_like(phi_mech)
    if cfg.get("dolomitization", False):
        zone = depth_below_platform_top < cfg["dolomite_zone_m"]
        dol = np.where(zone, cfg["dolomite_fraction"], 0.0)
        phi_ip = np.where(zone, np.maximum(phi_ip + cfg["dphi_dolomite"], 0.0), phi_ip)
    if cfg.get("deep_leaching", False):
        zone = depth_below_platform_top < cfg["leaching_zone_m"]
        phi_st = np.where(zone, phi_st + cfg["dphi_vug"], phi_st)
    clay = w[:, 5] * cfg["clay_in_argillaceous"] if w.shape[1] > 5 else np.zeros_like(phi_mech)
    calcite = np.clip(1.0 - dol - clay, 0.0, 1.0)
    return {"phi_ip": phi_ip, "phi_stiff": phi_st, "phi_crack": phi_cr,
            "phi": phi_ip + phi_st + phi_cr, "calcite": calcite, "dolomite": dol, "clay": clay}
