"""Stratigraphic grid from CarboKitten output (manuscript Sect. 3.1)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import CKSection


def stratigraphic_column(net: np.ndarray) -> np.ndarray:
    """Preserved thickness per frame, following CarboKitten's stack algorithm.

    ``net`` holds deposition minus disintegration along axis 0.  CarboKitten
    takes the cumulative sum, then sweeps backwards replacing each value by the
    minimum of itself and the (already updated) value above; the differences
    are the preserved thicknesses.  That sweep is a suffix minimum.
    """
    cum = np.cumsum(net, axis=0)
    suffix_min = np.minimum.accumulate(cum[::-1], axis=0)[::-1]
    out = np.diff(suffix_min, axis=0, prepend=np.zeros_like(suffix_min[:1]))
    return out


@dataclass
class StratGrid:
    """Layer-cake grid: layer k is CarboKitten output frame k (k = 0 is initial)."""

    x: np.ndarray            # (nx,)
    t: np.ndarray            # (nl,) time at the top of each layer, Myr
    h: np.ndarray            # (nl, nf, nx) preserved depositional thickness, m
    H: np.ndarray            # (nl, nx) total layer thickness, m
    p: np.ndarray            # (nl, nf, nx) facies fractions
    z_base: np.ndarray       # (nx,) basement elevation at the final time, m
    z_top: np.ndarray        # (nl, nx) elevation of each time-line at the final time
    wd: np.ndarray           # (nl, nx) water depth at deposition, m
    exposure: np.ndarray     # (nl, nx) exposure time within the meteoric zone, Myr
    sea_level: np.ndarray    # (nl,)
    facies_names: list
    standin: bool


def build_grid(sec: CKSection, meteoric_zone_m: float = 20.0, meteoric_profile: str = "step") -> StratGrid:
    net = sec.deposition - sec.disintegration                      # (nl, nf, nx)
    h = np.clip(stratigraphic_column(net), 0.0, None)
    H = h.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(H[:, None, :] > 0, h / H[:, None, :], 0.0)
    z_base = sec.eta0 - sec.subsidence_rate * (sec.t[-1] - sec.t[0])
    z_top = z_base[None, :] + np.cumsum(H, axis=0)                 # Eq. (1)
    wd = sec.sea_level[:, None] - sec.bathymetry
    tau = exposure_time(sec.t, sec.sea_level, sec.bathymetry, H, meteoric_zone_m, meteoric_profile)
    return StratGrid(x=sec.x, t=sec.t, h=h, H=H, p=p, z_base=z_base, z_top=z_top,
                     wd=wd, exposure=tau, sea_level=sec.sea_level,
                     facies_names=sec.facies_names, standin=sec.standin)


def exposure_time(t, sea_level, bathymetry, H, d_m, profile="step"):
    """Eq. (3): time each layer spends close to a subaerially exposed surface.

    Depth of layer k below the surface at frame j > k is approximated by the
    preserved thickness deposited in frames k+1..j.

    ``profile="step"``: exposure counts fully while the layer is within ``d_m`` of the
    surface and not at all below (the v0.1 rule).  ``profile="exponential"``: exposure
    at depth d is weighted by exp(-d / d_m), so that alteration is strongest directly
    beneath the exposure surface and fades gradually downward.
    """
    nl, nx = H.shape
    dt = np.diff(t, prepend=t[0])
    exposed = (sea_level[:, None] < bathymetry).astype(float) * dt[:, None]
    C = np.cumsum(H, axis=0)
    tau = np.zeros((nl, nx))
    if profile == "step":
        E = np.cumsum(exposed, axis=0)                             # (nl, nx)
        for ix in range(nx):
            jmax = np.searchsorted(C[:, ix], C[:, ix] + d_m, side="left") - 1
            jmax = np.clip(jmax, np.arange(nl), nl - 1)
            tau[:, ix] = E[jmax, ix] - E[:, ix]
        return tau
    if profile != "exponential":
        raise ValueError(f"unknown meteoric profile {profile!r}")
    k = np.arange(nl)
    for ix in range(nx):
        J = np.flatnonzero(exposed[:, ix] > 0)
        if J.size == 0:
            continue
        depth = C[J, ix][None, :] - C[:, ix][:, None]              # (nl, nJ)
        w = np.where(J[None, :] > k[:, None], np.exp(-np.maximum(depth, 0.0) / d_m), 0.0)
        tau[:, ix] = w @ exposed[J, ix]
    return tau


def flooding_surface_frames(t, sea_level, rise_fraction: float = 0.5):
    """Frames of flooding surfaces: after each sea-level minimum, the first frame
    where sea level has recovered ``rise_fraction`` of the way to the next maximum."""
    s = np.asarray(sea_level)
    mins = [i for i in range(1, len(s) - 1) if s[i] <= s[i - 1] and s[i] < s[i + 1]]
    frames = []
    for i in mins:
        nxt = i + 1
        while nxt < len(s) - 1 and not (s[nxt] >= s[nxt - 1] and s[nxt] > s[nxt + 1]):
            nxt += 1
        amp = s[nxt] - s[i]
        if amp < 2.0:                     # ignore sub-2 m wiggles
            continue
        target = s[i] + rise_fraction * amp
        j = i + int(np.argmax(s[i:nxt + 1] >= target))
        frames.append(j)
    return np.array(sorted(set(frames)), dtype=int)
