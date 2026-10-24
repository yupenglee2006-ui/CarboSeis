"""Run the complete CarboSeis chain for one scenario (manuscript Sects. 2 and 3).

Steps 1-7 of Fig. 1: stratigraphic grid, texture, compaction, diagenesis, rock physics,
earth model and seismic.  ``run_chain`` reads the CarboKitten output named in the scenario
unless a stratigraphic grid is passed, so the petrophysical description can be changed and
the seismic regenerated without a new forward run.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import earthmodel as emod
from .grid import StratGrid, build_grid
from .io import read_section
from .properties import compact, diagenesis, texture_fractions
from .rockphysics import elastic_properties
from .seismic import add_noise, level1, level2, twt_edges


@dataclass
class ChainResult:
    grid: StratGrid
    q: np.ndarray
    layers: dict              # layer-wise arrays on CarboKitten columns
    em: emod.EarthModel
    l1: object = None
    l2: object = None
    l2_depth: np.ndarray = None
    extra: dict = field(default_factory=dict)


def constant_impedance(p, table):
    """Conventional reference: one Vp and density per facies, mixed by facies fraction."""
    vp_f = np.array([r[0] for r in table], float)
    rho_f = np.array([r[1] for r in table], float)
    vp = np.einsum("kfx,f->kx", p, vp_f)
    rho = np.einsum("kfx,f->kx", p, rho_f)
    return vp, vp / 1.9, rho


def run_chain(sc: dict, grid: StratGrid | None = None, f0: float | None = None,
              levels=(1, 2), property_mode: str = "process") -> ChainResult:
    """Run steps 1-7 for scenario ``sc`` (a dict from :func:`carboseis.load_scenario`).

    ``grid``: a stratigraphic grid built earlier from the same forward run (skips reading the
    HDF5 file); ``f0``: dominant frequency overriding ``[seismic]``; ``levels``: imaging levels
    to compute (1 = 1-D convolution, 2 = point-spread function); ``property_mode``:
    ``"process"`` (texture, compaction, diagenesis and rock physics) or ``"constant"`` (one
    impedance per facies from ``[constant_impedance]``, the conventional reference of E1).
    """
    em_cfg, seis = sc["earthmodel"], sc["seismic"]
    if grid is None:
        path = sc["carbokitten"]["output"] if not sc.get("_use_standin") else sc["standin"]["output"]
        grid = build_grid(read_section(path, sc["grid"]["group"]), sc["grid"]["meteoric_zone_m"],
                           sc["grid"].get("meteoric_profile", "step"))
    g = grid
    q = texture_fractions(g.p, g.wd, sc["texture"])
    top_unc = g.z_top[-1]
    burial_top = em_cfg["overburden_m"] + (top_unc.max() - top_unc)
    hc, phi_mech, zmid, solid = compact(g.H, q, sc["compaction"], burial_top)
    elev_bot = g.z_base[None, :] + np.cumsum(hc, axis=0) - hc
    plat = float((elev_bot + hc)[-1].max())
    d_top, d_bot = emod.layer_depths(elev_bot, hc, plat, em_cfg)
    datum = em_cfg["water_depth_m"] + em_cfg["overburden_m"]
    div = diagenesis(phi_mech, q, g.exposure, zmid, d_top - datum, sc["diagenesis"])
    rp = sc["rockphysics"]
    sg = None
    if rp.get("gas_column_m", 0.0) > 0:
        sg = np.where(0.5 * (d_top + d_bot) < datum + rp["gas_column_m"], rp["gas_saturation"], 0.0)
    if property_mode == "constant":
        vp, vs, rho = constant_impedance(g.p, sc["constant_impedance"]["vp_rho"])
    else:
        vp, vs, rho = elastic_properties(div, rp, sg)
    thick = hc > 1e-6
    tex_dom = np.where(thick, np.argmax(q, axis=1), -1).astype(float)
    fac_dom = np.where(thick, np.argmax(g.p, axis=1), -1).astype(float)
    age = np.broadcast_to(g.t[:, None], hc.shape).astype(float)
    layers = dict(hc=hc, phi_mech=phi_mech, zmid=zmid, d_top=d_top, d_bot=d_bot, vp=vp, vs=vs,
                  rho=rho, tex_dom=tex_dom, fac_dom=fac_dom, age=age, **div)
    x = g.x
    xt = np.arange(x[0], x[-1] + 1e-6, em_cfg["trace_spacing_m"])
    li = lambda a, nearest=False: emod.interp_layers(x, xt, a, nearest)
    cell = {"age": li(age), "phi": li(div["phi"]), "texture": li(tex_dom, True),
            "facies": li(fac_dom, True)}
    em = emod.assemble(xt, li(d_top), li(d_bot), li(vp), li(vs), li(rho), cell, em_cfg, datum)
    res = ChainResult(grid=g, q=q, layers=layers, em=em)
    if 1 in levels:
        res.l1 = level1(em, seis, f0)
    if 2 in levels and seis.get("psf", True):
        res.l2, res.l2_depth = level2(em, seis, f0)
    snr = float(seis.get("noise_snr", 0.0) or 0.0)
    if snr > 0:                                   # band-limited random noise (Sect. 3.7)
        ncfg = dict(seis, **({"dominant_frequency_Hz": f0} if f0 is not None else {}))
        seed = int(seis.get("noise_seed", 0))
        if res.l1 is not None:
            res.l1 = add_noise(res.l1, snr, ncfg, seed)
        if res.l2 is not None:
            res.l2 = add_noise(res.l2, snr, ncfg, seed + 1)
    return res


def depth_to_twt(em, depths):
    """Two-way time (s) of given depths (n, ntr) on each trace."""
    te = twt_edges(em.vp, em.dz)
    zt = np.arange(em.vp.shape[0]) * em.dz
    out = np.empty_like(depths, dtype=float)
    for j in range(em.vp.shape[1]):
        out[:, j] = np.interp(depths[:, j], zt, te[:, j])
    return out
