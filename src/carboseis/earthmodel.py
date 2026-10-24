"""Earth-model assembly (manuscript Sect. 3.6).

The compacted carbonate layers are embedded below water, a clastic overburden
and an onlapping background fill, above a basement.  Properties are averaged
onto a regular depth grid exactly: Backus (harmonic) averages of the P-wave and
shear moduli and arithmetic averages of density over each cell.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WATER, OVERBURDEN, BACKGROUND, CARBONATE, BASEMENT = range(5)


@dataclass
class EarthModel:
    x: np.ndarray            # (ntr,) trace positions, m
    z: np.ndarray            # (nz,) cell-centre depths below sea surface, m
    dz: float
    vp: np.ndarray           # (nz, ntr) m/s
    vs: np.ndarray
    rho: np.ndarray          # g/cm3
    unit: np.ndarray         # (nz, ntr) int unit code
    layer: np.ndarray        # (nz, ntr) carbonate layer index, -1 elsewhere
    fields: dict             # extra cell fields (age, texture, facies, phi)
    layer_top_depth: np.ndarray   # (nl, ntr) depth of the top of every carbonate layer
    layer_base_depth: np.ndarray  # (nl, ntr)
    platform_top_depth: float

    @property
    def impedance(self):
        return self.rho * self.vp


def clastic_props(depth, water_depth, overburden, cfg):
    b = np.maximum(depth - water_depth, 0.0)
    vp = np.minimum(cfg["clastic_vp0_m_s"] + cfg["clastic_vp_gradient_per_s"] * b, cfg["clastic_vp_max_m_s"])
    bg = depth > water_depth + overburden
    vp = np.where(bg, vp + cfg["background_vp_offset_m_s"], vp)
    rho = 0.31 * vp ** 0.25                                   # Gardner, g/cm3
    vs = np.maximum((vp - 1360.0) / 1.16, 300.0)              # Castagna mudrock line
    return vp, vs, rho, np.where(bg, BACKGROUND, OVERBURDEN)


def layer_depths(elev_bot, hc, platform_top_elev, cfg):
    top_datum = cfg["water_depth_m"] + cfg["overburden_m"]
    d_top = top_datum + (platform_top_elev - (elev_bot + hc))
    d_bot = top_datum + (platform_top_elev - elev_bot)
    return d_top, d_bot


def interp_layers(x_src, x_dst, arr, nearest=False):
    """Interpolate (nl, nx) layer arrays along x."""
    if nearest:
        idx = np.clip(np.searchsorted(x_src, x_dst), 1, len(x_src) - 1)
        idx = np.where(np.abs(x_src[idx - 1] - x_dst) < np.abs(x_src[idx] - x_dst), idx - 1, idx)
        return arr[..., idx]
    out = np.empty(arr.shape[:-1] + (len(x_dst),))
    flat = arr.reshape(-1, arr.shape[-1])
    o = out.reshape(-1, len(x_dst))
    for i in range(flat.shape[0]):
        o[i] = np.interp(x_dst, x_src, flat[i])
    return out


def assemble(x_traces, d_top, d_bot, vp, vs, rho, cell_fields, cfg, platform_top_depth):
    """Build the depth-grid earth model.  All layer arrays are (nl, ntr) on the
    trace positions; ``cell_fields`` maps names to (nl, ntr) layer values."""
    WD, Hob, dz = cfg["water_depth_m"], cfg["overburden_m"], cfg["dz_m"]
    bvp, bvs, brho = cfg["basement"]
    zmax = float(np.max(d_bot)) + cfg["basement_thickness_m"]
    nz = int(np.ceil(zmax / dz))
    edges = np.arange(nz + 1) * dz
    zc = 0.5 * (edges[1:] + edges[:-1])
    ntr = len(x_traces)
    out = {k: np.zeros((nz, ntr)) for k in ("vp", "vs", "rho")}
    unit = np.zeros((nz, ntr), int)
    layer = -np.ones((nz, ntr), int)
    fields = {k: np.full((nz, ntr), np.nan) for k in cell_fields}
    nl = d_top.shape[0]
    for j in range(ntr):
        top_c = d_top[:, j].min()          # top of carbonate column = top of youngest layer
        base_c = d_bot[0, j]
        e_cl = edges[(edges > WD) & (edges < top_c)]
        b_cl = np.concatenate([[WD], e_cl, [top_c]])
        m_cl = 0.5 * (b_cl[1:] + b_cl[:-1])
        vcl, scl, rcl, ucl = clastic_props(m_cl, WD, Hob, cfg)
        order = np.arange(nl - 1, -1, -1)
        thick = d_bot[order, j] - d_top[order, j]
        keep = thick > 1e-9
        lay = order[keep]
        b_cb = np.concatenate([[top_c], d_bot[lay, j]])
        bounds = np.concatenate([[0.0, WD], b_cl[1:], b_cb[1:], [zmax + dz]])
        p_vp = np.concatenate([[1500.0], vcl, vp[lay, j], [bvp]])
        p_vs = np.concatenate([[0.0], scl, vs[lay, j], [bvs]])
        p_rho = np.concatenate([[1.03], rcl, rho[lay, j], [brho]])
        p_unit = np.concatenate([[WATER], ucl, np.full(len(lay), CARBONATE), [BASEMENT]])
        p_layer = np.concatenate([[-1], -np.ones(len(vcl), int), lay, [-1]])
        th = np.diff(bounds)
        M = p_rho * (p_vp * 1e-3) ** 2
        G = p_rho * (p_vs * 1e-3) ** 2
        cum = lambda v: np.concatenate([[0.0], np.cumsum(th * v)])
        Fi = np.diff(np.interp(edges, bounds, cum(1.0 / M))) / dz
        Fr = np.diff(np.interp(edges, bounds, cum(p_rho))) / dz
        Fg = np.diff(np.interp(edges, bounds, cum(np.where(G > 0, 1.0 / np.maximum(G, 1e-12), 1e12)))) / dz
        out["rho"][:, j] = Fr
        out["vp"][:, j] = np.sqrt(1.0 / Fi / Fr) * 1e3
        out["vs"][:, j] = np.where(Fg < 1e6, np.sqrt(1.0 / np.maximum(Fg, 1e-12) / Fr) * 1e3, 0.0)
        pid = np.clip(np.searchsorted(bounds, zc, side="right") - 1, 0, len(p_unit) - 1)
        unit[:, j] = p_unit[pid]
        layer[:, j] = p_layer[pid]
        lk = layer[:, j]
        for k, arr in cell_fields.items():
            fields[k][:, j] = np.where(lk >= 0, arr[np.maximum(lk, 0), j], np.nan)
    return EarthModel(x=np.asarray(x_traces), z=zc, dz=dz, vp=out["vp"], vs=out["vs"], rho=out["rho"],
                      unit=unit, layer=layer, fields=fields, layer_top_depth=d_top,
                      layer_base_depth=d_bot, platform_top_depth=platform_top_depth)
