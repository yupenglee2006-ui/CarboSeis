"""Seismic simulation (manuscript Sect. 3.7).

Level 1: normal-incidence reflectivity and depth-to-time conversion with the model
velocities (Eq. 8), FFT convolution with a zero-phase wavelet in two-way time.
Level 2: point-spread-function (PSF) imaging in depth (Eq. 9): a wavenumber-domain
filter that keeps the wavelet band and illuminated dips up to theta_max,
applied in overlapping depth windows, then converted to time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve


def ricker(f0, dt, half_length=None):
    half_length = half_length or 1.5 / f0
    t = np.arange(-half_length, half_length + dt / 2, dt)
    a = (np.pi * f0 * t) ** 2
    return t, (1 - 2 * a) * np.exp(-a)


def ricker_spectrum(f, f0):
    return (f / f0) ** 2 * np.exp(1 - (f / f0) ** 2)          # peak 1 at f0


def ormsby(f, dt, half_length=0.1):
    f1, f2, f3, f4 = f
    t = np.arange(-half_length, half_length + dt / 2, dt)

    def term(fi):
        return (np.pi * fi) ** 2 * np.sinc(fi * t) ** 2
    w = (term(f4) - term(f3)) / (f4 - f3) - (term(f2) - term(f1)) / (f2 - f1)
    return t, w / w.max()


def wavelet(cfg, dt):
    if cfg.get("wavelet", "ricker") == "ormsby":
        return ormsby(cfg["ormsby_Hz"], dt)
    return ricker(cfg["dominant_frequency_Hz"], dt)


def reflectivity(Z):
    """R at the interface above each cell (first row zero)."""
    R = np.zeros_like(Z)
    R[1:] = (Z[1:] - Z[:-1]) / (Z[1:] + Z[:-1])
    return R


def twt_edges(vp, dz):
    """Two-way time (s) at the top edge of each cell, shape (nz, ntr)."""
    t = np.zeros_like(vp)
    t[1:] = 2.0 * np.cumsum(dz / vp[:-1], axis=0)
    return t


@dataclass
class TimeSection:
    t: np.ndarray            # (nt,) s
    x: np.ndarray            # (ntr,)
    data: np.ndarray         # (nt, ntr)


def level1(em, cfg, f0=None):
    """1-D convolution in two-way time; returns TimeSection."""
    cfg = dict(cfg)
    if f0 is not None:
        cfg["dominant_frequency_Hz"] = f0
    dt_f, dt = cfg["dt_fine_ms"] * 1e-3, cfg["dt_ms"] * 1e-3
    R = reflectivity(em.impedance)
    te = twt_edges(em.vp, em.dz)
    tmax = te[-1].max()
    nf = int(np.ceil(tmax / dt_f)) + 2
    fine = np.zeros((nf, R.shape[1]))
    pos = te / dt_f
    i0 = np.floor(pos).astype(int)
    w1 = pos - i0
    cols = np.broadcast_to(np.arange(R.shape[1]), R.shape)
    np.add.at(fine, (i0, cols), R * (1 - w1))
    np.add.at(fine, (np.minimum(i0 + 1, nf - 1), cols), R * w1)
    _, w = wavelet(cfg, dt_f)
    out = fftconvolve(fine, w[:, None], mode="same")
    step = int(round(dt / dt_f))
    data = out[::step]
    return TimeSection(t=np.arange(data.shape[0]) * dt, x=em.x, data=data)


def psf_filter(nz, nx, dz, dx, vref, cfg, f0=None):
    f0 = f0 or cfg["dominant_frequency_Hz"]
    kz = np.fft.fftfreq(nz, dz)[:, None]
    kx = np.fft.fftfreq(nx, dx)[None, :]
    k = np.hypot(kx, kz)
    f = 0.5 * k * vref
    W = ricker_spectrum(f, f0)
    dip = np.degrees(np.arctan2(np.abs(kx), np.abs(kz)))
    tmax = cfg["psf_max_dip_deg"]
    taper = np.clip((tmax + 5.0 - dip) / 10.0, 0.0, 1.0)
    H = W * taper
    norm = H[:, 0].sum() / nz                                # flat-layer impulse peak -> 1
    return H / norm


def level2_depth(em, cfg, f0=None):
    """PSF image in depth (nz, ntr), overlapping windows with sin^2 tapers."""
    R = reflectivity(em.impedance)
    nz, nx = R.shape
    L = int(round(cfg["psf_window_m"] / em.dz))
    hop = L // 2
    img = np.zeros_like(R)
    wsum = np.zeros(nz)
    npx = int(2 ** np.ceil(np.log2(2 * nx)))
    npz = int(2 ** np.ceil(np.log2(2 * L)))
    taper = np.sin(np.pi * (np.arange(L) + 0.5) / L) ** 2
    for start in range(-hop, nz, hop):
        lo, hi = max(start, 0), min(start + L, nz)
        if hi <= lo:
            continue
        tw = taper[lo - start:hi - start]
        seg = np.zeros((npz, npx))
        seg[: hi - lo, :nx] = R[lo:hi] * tw[:, None]
        vref = float(np.median(em.vp[lo:hi]))
        H = psf_filter(npz, npx, em.dz, em.x[1] - em.x[0], vref, cfg, f0)
        out = np.real(np.fft.ifft2(np.fft.fft2(seg) * H))
        # undo the circular shift of a zero-phase kernel: it is centred at 0 already
        img[lo:hi] += out[: hi - lo, :nx]
        wsum[lo:hi] += tw
    return img / np.maximum(wsum, 1e-9)[:, None]


def depth_to_time(em, field, dt, nearest=False):
    """Map a depth-domain section to two-way time using the model velocities."""
    te = twt_edges(em.vp, em.dz)
    tc = te + em.dz / em.vp                                  # time at cell centres
    nt = int(np.ceil(te[-1].max() / dt)) + 1
    t = np.arange(nt) * dt
    out = np.full((nt, field.shape[1]), np.nan)
    for j in range(field.shape[1]):
        if nearest:
            idx = np.clip(np.searchsorted(tc[:, j], t), 0, field.shape[0] - 1)
            v = field[idx, j]
            v = np.where(t <= tc[-1, j], v, np.nan)
        else:
            v = np.interp(t, tc[:, j], field[:, j], left=np.nan, right=np.nan)
        out[:, j] = v
    return TimeSection(t=t, x=em.x, data=out)


def level2(em, cfg, f0=None):
    img = level2_depth(em, cfg, f0)
    ts = depth_to_time(em, img, cfg["dt_ms"] * 1e-3)
    ts.data = np.nan_to_num(ts.data)
    return ts, img


def add_noise(ts, snr, cfg, seed=0):
    """Band-limited random noise (white noise convolved with the wavelet) scaled so that the
    RMS amplitude of the section divided by the RMS amplitude of the noise equals ``snr``.
    ``snr`` = 0 or None returns the section unchanged."""
    if not snr:
        return ts
    rng = np.random.default_rng(seed)
    n = rng.standard_normal(ts.data.shape)
    _, w = wavelet(cfg, cfg["dt_ms"] * 1e-3)
    n = fftconvolve(n, w[:, None], mode="same")
    n *= np.sqrt(np.mean(ts.data ** 2)) / (snr * np.sqrt(np.mean(n ** 2)))
    return TimeSection(t=ts.t, x=ts.x, data=ts.data + n)
