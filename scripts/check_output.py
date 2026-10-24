"""Sanity report for a CarboKitten output file before it is used for the paper figures.

    python scripts/check_output.py [FILE.h5] [--json build/check_output.json] [--strict]

Prints, and optionally writes as JSON, the facts that catch the known pitfalls
(COMPATIBILITY.md): file layout, platform relief (centre above flanks), final water
depth, the V1 consistency of the reconstructed time-lines, water depth at deposition
and exposure time in the platform interior, the dominant-texture budget, and the
largest single-interval deposit (a numerical blow-up shows up there).
It also reports the Julia, CarboKitten and bridge versions that the bridge recorded in the file
(v4.0.1) and warns if they are missing or if Julia has another minor version than the bridge.
With --strict the script exits with status 1 when there is any warning.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from common import ROOT, TEXTURE_LABELS, bridge_julia_version, ck_bridge, run_provenance, same_minor, scenario
from carboseis.grid import build_grid, flooding_surface_frames
from carboseis.io import read_section
from carboseis.properties import texture_fractions


def pct(a, q=(5, 50, 95)):
    a = np.asarray(a)[np.isfinite(a)]
    return [round(float(v), 2) for v in np.percentile(a, q)] if a.size else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?")
    ap.add_argument("--json", default=None)
    ap.add_argument("--strict", action="store_true", help="exit with status 1 if there are warnings")
    a = ap.parse_args()
    sc = scenario()
    path = a.file or (sc["standin"]["output"] if sc["_use_standin"] else sc["carbokitten"]["output"])
    group = sc["grid"]["group"]
    rep = {"file": str(Path(path).name), "size_MB": round(Path(path).stat().st_size / 2**20, 1)}
    with h5py.File(path, "r") as f:
        rep["groups"] = {k: sorted(f[k].keys()) for k in f.keys() if isinstance(f[k], h5py.Group)
                         and k != "input"}
        att = f["input"].attrs
        rep["standin"] = bool(att.get("carboseis_standin", att.get("seiskitten_standin", False)))
        rep["slice_attr"] = repr(f[group].attrs.get("slice"))
    sec = read_section(path, group)
    g = build_grid(sec, sc["grid"]["meteoric_zone_m"], sc["grid"].get("meteoric_profile", "step"))
    H = g.H.sum(axis=0)
    plat = H > 0.5 * H.max()                       # traces with more than half the maximum thickness
    ic = int(np.argmax(H))
    rep["section"] = {"traces": int(len(sec.x)), "frames": int(len(sec.t)),
                      "t_start_Myr": float(sec.t[0]), "t_end_Myr": float(sec.t[-1])}
    rep["relief_m"] = {
        "initial_topography_centre": round(float(sec.eta0[ic]), 1),
        "initial_topography_flank_min": round(float(sec.eta0.min()), 1),
        "final_elevation_centre": round(float(sec.bathymetry[-1, ic]), 1),
        "final_elevation_flank_min": round(float(sec.bathymetry[-1].min()), 1),
        "final_sea_level": round(float(sec.sea_level[-1]), 1),
        "final_water_depth_centre": round(float(sec.sea_level[-1] - sec.bathymetry[-1, ic]), 1),
        "uncompacted_thickness_centre": round(float(H[ic]), 1),
        "uncompacted_thickness_flank_min": round(float(H.min()), 1),
        "centre_above_flanks": bool(sec.eta0[ic] > sec.eta0.min()),
    }
    rep["V1_max_abs_m"] = float(np.abs(g.z_top[-1] - sec.bathymetry[-1]).max())
    layers = (g.H > 0.01) & plat[None, :]
    rep["platform_interior"] = {
        "traces": int(plat.sum()),
        "water_depth_at_deposition_p5_p50_p95_m": pct(g.wd[layers]),
        "fraction_of_layers_deposited_above_fwwb": round(float(
            (g.wd[layers] < sc["texture"]["fair_weather_wave_base_m"]).mean()), 3) if layers.any() else None,
        "exposure_time_p50_p95_max_Myr": pct(g.exposure[layers], (50, 95, 100)),
        "fraction_of_layers_with_exposure_gt_tau_crit": round(float(
            (g.exposure[layers] > sc["diagenesis"]["tau_crit_Myr"]).mean()), 3) if layers.any() else None,
    }
    q = texture_fractions(g.p, g.wd, sc["texture"])                 # (nl, ntex, nx)
    vol = (q * g.H[:, None, :])[:, :, plat].sum(axis=(0, 2))
    rep["texture_budget_platform"] = {t: round(float(v / vol.sum()), 3) for t, v in zip(TEXTURE_LABELS, vol)}
    fvol = g.h.sum(axis=(0, 2))
    rep["facies_budget_section"] = {n: round(float(v / fvol.sum()), 3) for n, v in zip(g.facies_names, fvol)}
    dep = sec.deposition.sum(axis=1)[1:]
    rep["max_single_interval_deposit_m"] = round(float(dep.max()), 2)
    rep["flooding_surfaces"] = int(len(flooding_surface_frames(g.t, g.sea_level)))
    warn = []
    if not rep["relief_m"]["centre_above_flanks"]:
        warn.append("platform relief inverted: elevation and water depth exchanged? (bridge NOTES.md B2)")
    if rep["V1_max_abs_m"] > 1e-3:
        warn.append("V1 fails: time-lines and recomputed elevation disagree (COMPATIBILITY.md #6)")
    if rep["max_single_interval_deposit_m"] > 25.0:
        warn.append("deposit > 25 m in one output interval: outside the stable transport range (bridge NOTES.md B4)")
    if rep["standin"]:
        warn.append("this is the Python stand-in, not CarboKitten output")
    else:
        prov = run_provenance(path)
        try:
            want = bridge_julia_version(ck_bridge())
        except SystemExit:
            want = None
        rep["provenance"] = {**prov, "bridge_manifest_julia": want}
        if prov["julia"] is None:
            warn.append("Julia version not recorded: run made with a bridge older than 0.1.0.dev3; "
                        "regenerate it before mixing it with new runs")
        elif want and not same_minor(prov["julia"], want):
            warn.append(f"made with Julia {prov['julia']}, but the bridge is resolved for Julia {want}; "
                        "runs from different Julia minor versions must not be mixed")
    rep["warnings"] = warn
    txt = json.dumps(rep, indent=1)
    print(txt)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(txt)
    if a.strict and warn:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
