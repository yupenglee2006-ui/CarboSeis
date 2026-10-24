"""Write a chain result to one HDF5 file (manuscript Sect. 2, outputs).

Layout of the file written by :func:`write_result` (and by ``carboseis run``)::

    /                     attributes: carboseis_version, created_utc, scenario_name,
                          forward_model_file, forward_model_julia_version,
                          forward_model_carbokitten_version, forward_model_bridge_version,
                          standin, property_mode, units_note
    /scenario             attributes: file_text (scenario file as read, if given),
                          effective_json (all parameters after command-line overrides)
    /layers               layer-wise quantities on the CarboKitten columns,
                          shape (n_layers, n_columns) unless stated
    /earth_model          the depth grid, shape (n_depth, n_traces)
    /seismic/level1       1-D convolution in two-way time, data (n_time, n_traces)
    /seismic/level2       point-spread-function image in two-way time (and in depth)

Every dataset carries a ``units`` attribute and, where useful, a ``description``.
Lengths are in m, times in Myr (geological) or s (two-way time), velocities in m/s,
density in g/cm3 and acoustic impedance in (m/s)(g/cm3).
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import h5py
import numpy as np

from .seismic import reflectivity

UNIT_NAMES = {0: "water", 1: "overburden", 2: "background fill", 3: "carbonate", 4: "basement"}


def _put(group, name, data, units="", description="", compress=True):
    data = np.asarray(data)
    kw = {}
    if compress and data.size > 1024 and data.dtype.kind in "fiub":
        kw = dict(compression="gzip", compression_opts=4, shuffle=True)
    ds = group.create_dataset(name, data=data, **kw)
    if units:
        ds.attrs["units"] = units
    if description:
        ds.attrs["description"] = description
    return ds


def _time_section(group, ts, cfg, f0, description):
    _put(group, "t", ts.t, "s", "two-way time")
    _put(group, "x", ts.x, "m", "trace position")
    _put(group, "data", ts.data.astype(np.float32), "", description)
    group.attrs["wavelet"] = str(cfg.get("wavelet", "ricker"))
    group.attrs["dominant_frequency_Hz"] = float(f0 if f0 is not None else cfg["dominant_frequency_Hz"])
    group.attrs["noise_snr"] = float(cfg.get("noise_snr", 0.0) or 0.0)


def write_result(path, res, sc: dict, scenario_text: str | None = None, f0: float | None = None,
                 property_mode: str = "process") -> Path:
    """Save the chain result ``res`` of scenario ``sc`` to ``path`` (HDF5) and return the path."""
    from . import __version__

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    g, L, em = res.grid, res.layers, res.em
    seis = sc["seismic"]
    src = sc["standin"]["output"] if sc.get("_use_standin") else sc["carbokitten"]["output"]
    with h5py.File(path, "w") as f:
        f.attrs["carboseis_version"] = __version__
        f.attrs["created_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
        f.attrs["scenario_name"] = str(sc.get("meta", {}).get("name", ""))
        f.attrs["forward_model_file"] = str(src)
        try:                                     # versions recorded by the bridge (>= 0.1.0.dev3)
            with h5py.File(src, "r") as ck:
                for k in ("carboseis_julia_version", "carboseis_carbokitten_version", "carboseis_bridge_version"):
                    if k in ck["input"].attrs:
                        v = ck["input"].attrs[k]
                        f.attrs["forward_model_" + k.replace("carboseis_", "")] = (
                            v.decode() if isinstance(v, bytes) else str(v))
        except (OSError, KeyError):
            pass
        f.attrs["standin"] = bool(g.standin)
        f.attrs["property_mode"] = property_mode
        f.attrs["units_note"] = ("lengths m; geological time Myr; two-way time s; velocity m/s; "
                                 "density g/cm3; impedance (m/s)(g/cm3)")

        s = f.create_group("scenario")
        if scenario_text is not None:
            s.attrs["file_text"] = scenario_text
        s.attrs["effective_json"] = json.dumps({k: v for k, v in sc.items() if not k.startswith("_")},
                                               default=str, indent=1)

        ly = f.create_group("layers")
        ly.attrs["facies_names"] = [str(n) for n in g.facies_names]
        ly.attrs["texture_names"] = [str(n) for n in sc["texture"]["names"]]
        ly.attrs["pore_types"] = ["interparticle", "stiff (moldic or vuggy)", "crack-like"]
        _put(ly, "x", g.x, "m", "position of the CarboKitten columns along the section")
        _put(ly, "time", g.t, "Myr", "time at the top of each layer (CarboKitten output frame)")
        _put(ly, "sea_level", g.sea_level, "m", "sea level at the top of each layer")
        _put(ly, "timeline_elevation", g.z_top, "m",
             "present elevation of each time-line before compaction (Eq. 1)")
        _put(ly, "water_depth", g.wd, "m", "depositional water depth (Eq. 2)")
        _put(ly, "exposure_time", g.exposure, "Myr", "time spent in the meteoric zone (Eq. 3)")
        _put(ly, "thickness_depositional", g.H, "m", "preserved depositional thickness")
        _put(ly, "facies_fractions", g.p, "1", "shape (n_layers, n_facies, n_columns)")
        _put(ly, "texture_fractions", res.q, "1", "shape (n_layers, n_textures, n_columns) (Eq. 4)")
        _put(ly, "thickness_compacted", L["hc"], "m", "thickness after compaction (Eq. 5)")
        _put(ly, "burial_depth_mid", L["zmid"], "m", "mid-layer burial depth below the sea floor")
        _put(ly, "depth_top", L["d_top"], "m", "depth of the layer top below the sea surface")
        _put(ly, "depth_base", L["d_bot"], "m", "depth of the layer base below the sea surface")
        _put(ly, "porosity_compaction", L["phi_mech"], "1", "porosity after compaction only")
        _put(ly, "porosity_interparticle", L["phi_ip"], "1", "after diagenesis")
        _put(ly, "porosity_stiff", L["phi_stiff"], "1", "moldic or vuggy, after diagenesis")
        _put(ly, "porosity_crack", L["phi_crack"], "1", "crack-like, after diagenesis")
        _put(ly, "porosity", L["phi"], "1", "total porosity after diagenesis")
        for m in ("calcite", "dolomite", "clay"):
            _put(ly, f"fraction_{m}", L[m], "1", f"{m} fraction of the mineral matrix")
        _put(ly, "vp", L["vp"], "m/s", "P-wave velocity (Eq. 7)")
        _put(ly, "vs", L["vs"], "m/s", "S-wave velocity (Eq. 7)")
        _put(ly, "rho", L["rho"], "g/cm3", "bulk density (Eq. 7)")
        _put(ly, "impedance", L["rho"] * L["vp"], "(m/s)(g/cm3)", "acoustic impedance")

        e = f.create_group("earth_model")
        e.attrs["dz_m"] = float(em.dz)
        e.attrs["platform_top_depth_m"] = float(em.platform_top_depth)
        e.attrs["unit_codes"] = json.dumps(UNIT_NAMES)
        _put(e, "x", em.x, "m", "trace position")
        _put(e, "z", em.z, "m", "cell-centre depth below the sea surface")
        _put(e, "vp", em.vp.astype(np.float32), "m/s", "Backus-averaged P-wave velocity")
        _put(e, "vs", em.vs.astype(np.float32), "m/s", "Backus-averaged S-wave velocity")
        _put(e, "rho", em.rho.astype(np.float32), "g/cm3", "cell-averaged density")
        _put(e, "impedance", em.impedance.astype(np.float32), "(m/s)(g/cm3)", "acoustic impedance")
        _put(e, "reflectivity", reflectivity(em.impedance).astype(np.float32), "1",
             "normal-incidence reflection coefficient at the top of each cell (Eq. 8)")
        _put(e, "unit", em.unit.astype(np.int8), "", "unit code, see attribute unit_codes")
        _put(e, "layer", em.layer.astype(np.int32), "", "carbonate layer index, -1 elsewhere")
        for name, units, desc in (("age", "Myr", "age of the carbonate layer"),
                                  ("phi", "1", "total porosity"),
                                  ("texture", "", "index of the dominant texture"),
                                  ("facies", "", "index of the dominant facies")):
            if name in em.fields:
                _put(e, name if name != "phi" else "porosity", em.fields[name].astype(np.float32),
                     units, desc)

        sg = f.create_group("seismic")
        if res.l1 is not None:
            _time_section(sg.create_group("level1"), res.l1, seis, f0,
                          "1-D convolution in two-way time")
        if res.l2 is not None:
            l2 = sg.create_group("level2")
            _time_section(l2, res.l2, seis, f0, "point-spread-function image in two-way time")
            l2.attrs["max_illuminated_dip_deg"] = float(seis.get("psf_max_dip_deg", np.nan))
            if res.l2_depth is not None:
                _put(l2, "depth_image", res.l2_depth.astype(np.float32), "",
                     "point-spread-function image in depth, on the earth-model grid")
    return path
