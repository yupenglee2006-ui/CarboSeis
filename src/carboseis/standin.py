"""Development stand-in for CarboKitten.jl.

A compact Python emulation of CarboKitten's ALCAP model (Bosscher-Schlager
production, Burgess cellular automaton, active-layer transport with
disintegration and lithification) on a thin periodic strip.  It writes HDF5 in
CarboKitten's layout so the full CarboSeis chain can be developed and tested
without Julia.

It is NOT CarboKitten: transport and the sediment buffer are simplified.  Files
it writes carry the attribute ``carboseis_standin = True`` and every figure made
from them is watermarked.  Paper figures must come from CarboKitten runs
(``ck_run.jl`` of the CarboSeis-CarboKitten bridge).
"""
from __future__ import annotations

import h5py
import numpy as np
from pathlib import Path

from .config import load_scenario


def sea_level_from_file(path, window_Ma, t_Myr):
    d = np.loadtxt(path, skiprows=1)
    m = (d[:, 0] >= window_Ma[0]) & (d[:, 0] <= window_Ma[1])
    d = d[m][np.argsort(-d[m][:, 0])]
    return np.interp(t_Myr, -d[:, 0], d[:, 1])


def dome(X, Y, Lx, Ly, top, rfrac, slope, floor):
    r = rfrac * min(Lx, Ly)
    dist = np.hypot(X - 0.5 * Lx, Y - 0.5 * Ly)
    return np.where(dist <= r, top, np.maximum(floor, top - slope * (dist - r)))


def bs92(w, gm, k, Ik, I0):
    return np.where(w >= 0, gm * np.tanh(I0 / Ik * np.exp(-k * np.maximum(w, 0.0))), 0.0)


def pelagic(w, gm, k, Ik, I0, dmax, n=16):
    top = np.clip(w, 0.0, dmax)
    s = (np.arange(n) + 0.5) / n
    z = top[..., None] * s
    return gm * np.tanh(I0 / Ik * np.exp(-k * z)).mean(axis=-1) * top


def _neighbour_counts(state, s):
    occ = (state == s).astype(np.int16)
    tot = np.zeros_like(occ)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dx or dy:
                tot += np.roll(np.roll(occ, dy, 0), dx, 1)
    return tot


def ca_step(state, n_species, viab, activ, step):
    counts = [_neighbour_counts(state, s + 1) for s in range(n_species)]
    new = np.zeros_like(state)
    for s in range(n_species):
        alive = (state == s + 1) & (counts[s] >= viab[s][0]) & (counts[s] <= viab[s][1])
        new[alive] = s + 1
    dead = state == 0
    for r in range(n_species):
        s = (r + step) % n_species                      # rotating priority
        born = dead & (new == 0) & (counts[s] >= activ[s][0]) & (counts[s] <= activ[s][1])
        new[born] = s + 1
    return new


def _upwind(A, eta, d, dx, tau):
    """Advect active-layer amounts A (nf, ny, nx) with velocity -d grad(eta) for time tau."""
    gx = (np.roll(eta, -1, 1) - np.roll(eta, 1, 1)) / (2 * dx)
    gy = (np.roll(eta, -1, 0) - np.roll(eta, 1, 0)) / (2 * dx)
    vmax = float(np.max(np.abs(d)) * max(np.abs(gx).max(), np.abs(gy).max(), 1e-12))
    nsub = max(1, int(np.ceil(vmax * tau / (0.8 * dx))))
    h = tau / nsub
    for _ in range(nsub):
        for f in range(A.shape[0]):
            vx, vy = -d[f] * gx, -d[f] * gy
            a = A[f]
            fx = np.where(vx > 0, vx * a, vx * np.roll(a, -1, 1))       # flux at i+1/2
            fy = np.where(vy > 0, vy * a, vy * np.roll(a, -1, 0))
            A[f] = a - h / dx * (fx - np.roll(fx, 1, 1) + fy - np.roll(fy, 1, 0))
    return np.clip(A, 0.0, None)


def simulate(ck: dict, *, t0, dt, steps, eta0, sea_level, subsidence, rd, t_lith,
             transfer, ca_interval, seed, dx, row, write_interval):
    rng = np.random.default_rng(seed)
    facies = ck["facies"]
    nf = len(facies)
    ny, nx = eta0.shape
    I0 = ck["insolation_W_m2"]
    active = [i for i, f in enumerate(facies) if f.get("active", False)]
    viab = [(4, 10)] * len(active)
    activ = [(6, 10)] * len(active)
    d = np.array([f["transport_coefficient_m_per_yr"] * 1e6 for f in facies])  # m/Myr per unit slope
    T = np.asarray(transfer, float)
    state = rng.integers(0, len(active) + 1, size=(ny, nx))
    for k in range(20):
        state = ca_step(state, len(active), viab, activ, k)
    eta = eta0.copy()
    comp = np.zeros((nf, ny, nx))
    comp[0] = 1.0
    cum = np.zeros((nf, ny, nx))          # net deposited per facies (caps disintegration)
    nw = steps // write_interval
    rec = {k: np.zeros((nw + 1, nf, nx)) for k in ("deposition", "disintegration", "production")}
    bath = np.zeros((nw + 1, nx))
    bath[0] = eta[row]
    tau_tr = t_lith / np.log(2.0)
    for n in range(1, steps + 1):
        t = t0 + n * dt
        if (n - 1) % ca_interval == 0:
            state = ca_step(state, len(active), viab, activ, n)
        eta -= subsidence * dt
        w = sea_level[n] - eta
        prod = np.zeros((nf, ny, nx))
        for i, f in enumerate(facies):
            if f["type"] == "benthic":
                p = bs92(w, f["maximum_growth_rate_m_per_Myr"], f["extinction_coefficient_per_m"],
                         f["saturation_intensity_W_m2"], I0)
                if i in active:
                    p = p * (state == active.index(i) + 1)
                prod[i] = p * dt
            elif f["type"] == "pelagic":
                prod[i] = pelagic(w, f["maximum_growth_rate_per_Myr"], f["extinction_coefficient_per_m"],
                                  f["saturation_intensity_W_m2"], I0, f["maximum_production_depth_m"]) * dt
        tot = prod.sum(0)
        cap = np.maximum(w, 0.0)
        scale = np.where(tot > cap, cap / np.maximum(tot, 1e-12), 1.0)
        prod *= scale
        dis_amt = np.where(w > 0, rd * dt, 0.0)
        dis = np.minimum(comp * dis_amt, cum)
        A = prod + np.einsum("ij,jyx->iyx", T, dis)
        A = _upwind(A, eta, d, dx, tau_tr)
        dep = A
        eta += dep.sum(0) - dis.sum(0)
        cum += dep - dis
        mem = 1.0
        comp = (comp * mem + dep) / (mem + dep.sum(0) + 1e-12)
        comp /= np.maximum(comp.sum(0), 1e-12)
        j = (n - 1) // write_interval + 1
        rec["deposition"][j] += dep[:, row]
        rec["disintegration"][j] += dis[:, row]
        rec["production"][j] += prod[:, row]
        if n % write_interval == 0:
            bath[j] = eta[row]
    return rec, bath, eta


def run_standin(scenario_path: str, out_path: str | None = None, overrides: dict | None = None) -> str:
    sc = load_scenario(scenario_path, overrides)
    ck, si = sc["carbokitten"], sc["standin"]
    out_path = out_path or si["output"]
    nx = ck["grid_size"][0]
    ny = si["ny"]
    dx = ck["phys_scale_m"]
    x = (np.arange(nx) + 0.5) * dx
    Ly_full = ck["grid_size"][1] * dx
    y = 0.5 * Ly_full + (np.arange(ny) - ny // 2) * dx
    X, Y = np.meshgrid(x, y)
    pr = ck["prerun"]
    eta_dome = dome(X, Y, nx * dx, Ly_full, pr["dome_top_m"], pr["dome_radius_fraction"],
                    pr["dome_flank_slope"], pr["dome_floor_m"])
    # pre-run (coarse) to create the initial topography, as in the CarboKitten validation case
    dt_p = 0.001
    steps_p = int(round(pr["steps"] * pr["dt_yr"] * 1e-6 / dt_p))
    tp = np.arange(steps_p + 1) * dt_p
    slp = pr["sea_level_base_m"] + pr["sea_level_amplitude_m"] * np.sin(2 * np.pi * tp / pr["sea_level_period_Myr"])
    _, _, eta_init = simulate(ck, t0=0.0, dt=dt_p, steps=steps_p, eta0=eta_dome, sea_level=slp,
                              subsidence=pr["subsidence_rate_m_per_Myr"], rd=pr["disintegration_rate_m_per_Myr"],
                              t_lith=pr["lithification_time_yr"] * 1e-6, transfer=pr["disintegration_transfer"],
                              ca_interval=1, seed=si["seed"] + 1000, dx=dx, row=ny // 2, write_interval=steps_p)
    dt = si["dt_yr"] * 1e-6
    total = ck["steps"] * ck["dt_yr"] * 1e-6
    steps = int(round(total / dt))
    t = ck["t0_Myr"] + np.arange(steps + 1) * dt
    stretch = ck.get("sea_level_time_stretch", 1.0)
    sl = sea_level_from_file(ck["sea_level_file"], ck["sea_level_window_Ma"], t[0] + (t - t[0]) / stretch)
    sl = sl.mean() + ck.get("sea_level_scale", 1.0) * (sl - sl.mean())
    wi = si["write_interval"]
    rec, bath, _ = simulate(ck, t0=ck["t0_Myr"], dt=dt, steps=steps, eta0=eta_init, sea_level=sl,
                            subsidence=ck["subsidence_rate_m_per_Myr"], rd=ck["disintegration_rate_m_per_Myr"],
                            t_lith=ck["lithification_time_yr"] * 1e-6, transfer=ck["disintegration_transfer"],
                            ca_interval=max(1, int(round(ck["ca_interval"] * ck["dt_yr"] / si["dt_yr"]))),
                            seed=si["seed"], dx=dx, row=ny // 2, write_interval=wi)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    # CarboKitten v0.6.1 layout: no bathymetry dataset; the cumulative sediment
    # thickness is written instead, and the slice attribute is a byte string.
    t_w = t[::wi][:bath.shape[0]]
    thick = bath - eta_init[ny // 2][None, :] + ck["subsidence_rate_m_per_Myr"] * (t_w - t[0])[:, None]
    with h5py.File(out_path, "w") as f:
        g = f.create_group("input")
        g["x"], g["y"], g["t"], g["sea_level"] = x, y, t, sl
        g["initial_topography"] = eta_init
        for k, v in {"subsidence_rate": ck["subsidence_rate_m_per_Myr"], "delta_t": dt,
                     "time_steps": steps, "n_facies": len(ck["facies"]), "tag": ck["tag"] + " (stand-in)",
                     "carboseis_standin": True}.items():
            g.attrs[k] = v
        for i, fc in enumerate(ck["facies"], 1):
            g.create_group(f"facies{i}").attrs["name"] = fc["name"]
        p = f.create_group("profile")
        p.attrs["slice"] = np.bytes_(f":,{ny // 2 + 1}")
        p.attrs["write_interval"] = wi
        for k, v in rec.items():
            p[k] = np.moveaxis(v, 1, -1)[:, None, :, :]          # (frame, 1, x, facies)
        p["sediment_thickness"] = thick[:, None, :]           # (frame, 1, x)
    return out_path
