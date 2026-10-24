"""Read CarboKitten.jl HDF5 output (v0.6 layout) into a 2-D section.

CarboKitten writes, per output group, the datasets ``production``,
``deposition`` and ``disintegration`` with Julia shape (facies, x, y, frame)
and ``sediment_thickness`` with shape (x, y, frame); a slice group keeps a
singleton axis, e.g. (facies, x, 1, frame) for ``slice=(:, j)``.  HDF5 stores
Julia arrays in column-major order, so h5py sees the axes reversed:
(frame, y, x, facies) and (frame, y, x).  Frame 0 holds the initial state;
frame j >= 1 accumulates the ``write_interval`` model steps that end at
``input/t[j * write_interval]``.

Surface elevation (m, positive up, including subsidence) is

    eta(t) = initial_topography + sediment_thickness(t) - subsidence_rate * (t - t0),

which is how CarboKitten's own validation scripts reconstruct it; water depth is
``sea_level - eta``.  CarboKitten v0.6.1 writes no ``bathymetry`` dataset, so
eta is always recomputed with this formula from the *selected* row of the
initial topography.  Files from older writers (and older stand-ins)
that do contain ``bathymetry`` are still read, and the two must agree.

Compatibility notes (COMPATIBILITY.md): #2 missing ``bathymetry`` dataset,
#3 ``slice`` attribute stored as bytes, #4 elevation versus water depth, and
#6 the initial-topography row used for a slice group.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import h5py
import numpy as np


@dataclass
class CKSection:
    """One dip section of CarboKitten output (all lengths in m, times in Myr)."""

    x: np.ndarray                 # (nx,) position along the section
    t: np.ndarray                 # (nw+1,) time of each output frame
    deposition: np.ndarray        # (nw+1, nf, nx) per frame and facies
    disintegration: np.ndarray    # (nw+1, nf, nx)
    bathymetry: np.ndarray        # (nw+1, nx) surface elevation
    sea_level: np.ndarray         # (nw+1,)
    eta0: np.ndarray              # (nx,) initial topography
    subsidence_rate: float        # m/Myr
    facies_names: list[str]
    standin: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def n_facies(self) -> int:
        return self.deposition.shape[1]


def _parse_slice(s: str):
    out = []
    for part in s.split(","):
        part = part.strip()
        if part == ":":
            out.append(slice(None))
        elif ":" in part:
            a, b = part.split(":")
            out.append(slice(int(a) - 1, int(b)))
        else:
            out.append(int(part) - 1)       # Julia is 1-based
    return tuple(out)


def _decode(v) -> str:
    """HDF5 string attributes arrive as bytes, str or numpy scalars."""
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, str):
        return v
    v = np.asarray(v).item()
    return v.decode() if isinstance(v, bytes) else str(v)


def _select(arr, sx, sy, row, axis):
    """Reduce an h5py-ordered (frame, Y, X[, facies]) array of a group to a
    (frame, n[, facies]) section.  ``Y``/``X`` are the group's own (possibly
    singleton) axes."""
    if axis == "x":
        r = 0 if arr.shape[1] == 1 else (row if row is not None else arr.shape[1] // 2)
        return arr[:, r]
    c = 0 if arr.shape[2] == 1 else (row if row is not None else arr.shape[2] // 2)
    return arr[:, :, c]


def read_section(path: str, group: str = "profile", row: int | None = None,
                 axis: str = "x") -> CKSection:
    """Read a dip section from a CarboKitten HDF5 file.

    ``group`` may be a slice group (one y-row, written with ``slice=(:, j)``) or
    a volume group; for a volume give ``row`` (0-based, within the group) and
    ``axis`` ("x" keeps the section along x at fixed y).
    """
    with h5py.File(path, "r") as f:
        inp = f["input"]
        a = inp.attrs
        x_all = np.asarray(inp["x"][()], float)
        y_all = np.asarray(inp["y"][()], float)
        t_all = np.asarray(inp["t"][()], float)
        sl_all = np.asarray(inp["sea_level"][()], float)
        eta0_all = np.asarray(inp["initial_topography"][()], float)  # (ny, nx)
        sub = float(np.asarray(a["subsidence_rate"]).item())       # m/Myr
        g = f[group]
        wi = int(np.asarray(g.attrs["write_interval"]).item())
        sx, sy = _parse_slice(_decode(g.attrs["slice"]))
        dep = np.asarray(g["deposition"][()], float)        # (nw+1, Y, X, nf)
        dis = np.asarray(g["disintegration"][()], float)
        thick = (np.asarray(g["sediment_thickness"][()], float)
                 if "sediment_thickness" in g else None)    # (nw+1, Y, X)
        bat_file = np.asarray(g["bathymetry"][()], float) if "bathymetry" in g else None
        nf = dep.shape[-1]
        names = []
        for i in range(1, nf + 1):
            key = f"facies{i}"
            nm = inp[key].attrs.get("name", f"facies{i}") if key in inp else f"facies{i}"
            names.append(_decode(nm))
        # stand-in files written before v4.0 (when CarboSeis was called SeisKitten) carry the
        # attribute under its old name
        standin = bool(a.get("carboseis_standin", a.get("seiskitten_standin", False)))
        meta = {k: (v.item() if hasattr(v, "item") else v) for k, v in a.items()}

    # initial topography restricted to the group's own (Y, X) window
    ys = sy if isinstance(sy, slice) else [sy]
    xs = sx if isinstance(sx, slice) else [sx]
    eta_grp = eta0_all[ys][:, xs]                           # (Y, X)
    x_grp = x_all[xs]
    y_grp = y_all[ys]

    dep = _select(dep, sx, sy, row, axis)
    dis = _select(dis, sx, sy, row, axis)
    eta0 = _select(eta_grp[None], sx, sy, row, axis)[0]
    x = x_grp if axis == "x" else y_grp
    nw1 = dep.shape[0]
    t = t_all[::wi][:nw1]
    sea = sl_all[::wi][:nw1]
    if thick is not None:
        thick = _select(thick, sx, sy, row, axis)
        bat = eta0[None, :] + thick - sub * (t[:, None] - t_all[0])
        if bat_file is not None:
            b2 = _select(bat_file, sx, sy, row, axis)
            if not np.allclose(b2, bat, atol=1e-6):
                raise ValueError(f"{path}: 'bathymetry' and the elevation recomputed from "
                                 f"'sediment_thickness' differ by {np.abs(b2 - bat).max():.3g} m")
    elif bat_file is not None:
        bat = _select(bat_file, sx, sy, row, axis)
    else:
        raise KeyError(f"{path}/{group}: neither 'sediment_thickness' nor 'bathymetry' present")
    return CKSection(x=np.asarray(x), t=t, deposition=np.moveaxis(dep, -1, 1),
                     disintegration=np.moveaxis(dis, -1, 1), bathymetry=bat,
                     sea_level=sea, eta0=np.asarray(eta0), subsidence_rate=sub,
                     facies_names=names, standin=standin, meta=meta)
