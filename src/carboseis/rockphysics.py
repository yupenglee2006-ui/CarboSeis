"""Carbonate rock physics (manuscript Sect. 3.5).  Moduli in GPa, density in g/cm3,
velocities in m/s.

Tier A: Raymer-Hunt-Gardner velocity with mass-balance density.
Tier B: Xu-Payne-type model: Voigt-Reuss-Hill matrix, dry pores of three types
(interparticle, stiff, crack) inserted by differential effective medium (DEM)
with Berryman's spheroid P and Q factors, then Gassmann fluid substitution.
"""
from __future__ import annotations

import numpy as np


def vrh(fracs, K, mu, rho):
    """Voigt-Reuss-Hill average; fracs is a list of arrays summing to 1."""
    f = [np.asarray(x, float) for x in fracs]
    Kv = sum(fi * Ki for fi, Ki in zip(f, K))
    Kr = 1.0 / sum(fi / Ki for fi, Ki in zip(f, K))
    Gv = sum(fi * Gi for fi, Gi in zip(f, mu))
    Gr = 1.0 / sum(fi / Gi for fi, Gi in zip(f, mu))
    return 0.5 * (Kv + Kr), 0.5 * (Gv + Gr), sum(fi * ri for fi, ri in zip(f, rho))


def berryman_pq(Km, Gm, alpha, Ki=0.0, Gi=0.0):
    """P and Q factors for spheroidal inclusions of aspect ratio alpha < 1
    (Berryman, 1980; Mavko et al., Rock Physics Handbook)."""
    a = alpha
    theta = a / (1 - a * a) ** 1.5 * (np.arccos(a) - a * np.sqrt(1 - a * a))
    f = a * a / (1 - a * a) * (3 * theta - 2)
    R = 3 * Gm / (3 * Km + 4 * Gm)
    A = Gi / Gm - 1.0
    B = (Ki / Km - Gi / Gm) / 3.0
    F1 = 1 + A * (1.5 * (f + theta) - R * (1.5 * f + 2.5 * theta - 4.0 / 3.0))
    F2 = (1 + A * (1 + 1.5 * (f + theta) - 0.5 * R * (3 * f + 5 * theta)) + B * (3 - 4 * R)
          + 0.5 * A * (A + 3 * B) * (3 - 4 * R) * (f + theta - R * (f - theta + 2 * theta ** 2)))
    F3 = 1 + A * (1 - (f + 1.5 * theta) + R * (f + theta))
    F4 = 1 + 0.25 * A * (f + 3 * theta - R * (f - theta))
    F5 = A * (-f + R * (f + theta - 4.0 / 3.0)) + B * theta * (3 - 4 * R)
    F6 = 1 + A * (1 + f - R * (f + theta)) + B * (1 - theta) * (3 - 4 * R)
    F7 = 2 + 0.25 * A * (3 * f + 9 * theta - R * (3 * f + 5 * theta)) + B * theta * (3 - 4 * R)
    F8 = A * (1 - 2 * R + 0.5 * f * (R - 1) + 0.5 * theta * (5 * R - 3)) + B * (1 - theta) * (3 - 4 * R)
    F9 = A * ((R - 1) * f - R * theta) + B * theta * (3 - 4 * R)
    Tiijj = 3 * F1 / F2
    Tijij = 2 / F3 + 1 / F4 + (F4 * F5 + F6 * F7 - F8 * F9) / (F2 * F4)
    return Tiijj / 3.0, Tijij / 5.0


def sphere_pq(Km, Gm, Ki=0.0, Gi=0.0):
    zeta = Gm / 6.0 * (9 * Km + 8 * Gm) / (Km + 2 * Gm)
    return (Km + 4.0 / 3.0 * Gm) / (Ki + 4.0 / 3.0 * Gm), (Gm + zeta) / (Gi + zeta)


def _dry_pq_factory(alpha):
    """For empty pores (Ki = Gi = 0) every F_i is linear in R = 3G/(3K+4G);
    precompute the coefficients so P(R), Q(R) are cheap inside DEM."""
    Rs = np.array([0.0, 1.0])
    K1, G1 = 1.0, 0.0
    # evaluate F_i at R = 0 and R = 1 via berryman_pq's algebra
    a = alpha
    th = a / (1 - a * a) ** 1.5 * (np.arccos(a) - a * np.sqrt(1 - a * a))
    f = a * a / (1 - a * a) * (3 * th - 2)
    A, B = -1.0, 0.0

    def F(R):
        F1 = 1 + A * (1.5 * (f + th) - R * (1.5 * f + 2.5 * th - 4.0 / 3.0))
        F2 = (1 + A * (1 + 1.5 * (f + th) - 0.5 * R * (3 * f + 5 * th)) + B * (3 - 4 * R)
              + 0.5 * A * (A + 3 * B) * (3 - 4 * R) * (f + th - R * (f - th + 2 * th ** 2)))
        F3 = 1 + A * (1 - (f + 1.5 * th) + R * (f + th))
        F4 = 1 + 0.25 * A * (f + 3 * th - R * (f - th))
        F5 = A * (-f + R * (f + th - 4.0 / 3.0)) + B * th * (3 - 4 * R)
        F6 = 1 + A * (1 + f - R * (f + th)) + B * (1 - th) * (3 - 4 * R)
        F7 = 2 + 0.25 * A * (3 * f + 9 * th - R * (3 * f + 5 * th)) + B * th * (3 - 4 * R)
        F8 = A * (1 - 2 * R + 0.5 * f * (R - 1) + 0.5 * th * (5 * R - 3)) + B * (1 - th) * (3 - 4 * R)
        F9 = A * ((R - 1) * f - R * th) + B * th * (3 - 4 * R)
        return np.array([F1, F2, F3, F4, F5, F6, F7, F8, F9])

    c0 = F(0.0)
    c1 = F(1.0) - c0                       # F2 has an R^2 term: fit it exactly below
    c2 = 0.5 * (F(2.0) - 2 * F(1.0) + c0)
    c1 = F(1.0) - c0 - c2

    def pq(R):
        Fv = [c0[i] + c1[i] * R + c2[i] * R * R for i in range(9)]
        F1, F2, F3, F4, F5, F6, F7, F8, F9 = Fv
        Tiijj = 3 * F1 / F2
        Tijij = 2 / F3 + 1 / F4 + (F4 * F5 + F6 * F7 - F8 * F9) / (F2 * F4)
        return Tiijj / 3.0, Tijij / 5.0
    return pq


def dem_dry(Km, Gm, phis, alphas, nsteps=20):
    """Dry-frame moduli by DEM (RK4): pores of several types (porosities ``phis``,
    aspect ratios ``alphas``) added in fixed proportions to the matrix."""
    Km = np.asarray(Km, float)
    Gm = np.asarray(Gm, float)
    phis = [np.broadcast_to(np.asarray(p, float), Km.shape) for p in phis]
    phit = sum(phis)
    wts = [np.where(phit > 0, p / np.where(phit > 0, phit, 1.0), 0.0) for p in phis]
    pqs = [_dry_pq_factory(a) for a in alphas]

    def rhs(K, G, y):
        R = 3 * G / (3 * K + 4 * G)
        dK = np.zeros_like(K)
        dG = np.zeros_like(G)
        for w, pq in zip(wts, pqs):
            P, Q = pq(R)
            dK -= w * K * P
            dG -= w * G * Q
        return dK / (1 - y), dG / (1 - y)

    K, G = Km.copy(), Gm.copy()
    h = 1.0 / nsteps
    for i in range(nsteps):
        y = i * h * phit
        k1 = rhs(K, G, y)
        k2 = rhs(K + 0.5 * h * phit * k1[0], G + 0.5 * h * phit * k1[1], y + 0.5 * h * phit)
        k3 = rhs(K + 0.5 * h * phit * k2[0], G + 0.5 * h * phit * k2[1], y + 0.5 * h * phit)
        k4 = rhs(K + h * phit * k3[0], G + h * phit * k3[1], y + h * phit)
        K = np.maximum(K + h * phit / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), 1e-6)
        G = np.maximum(G + h * phit / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]), 1e-6)
    return K, G


def gassmann(Kdry, Km, Kfl, phi):
    """Eq. (6): Gassmann fluid substitution."""
    phi = np.maximum(phi, 1e-9)
    return Kdry + (1 - Kdry / Km) ** 2 / (phi / Kfl + (1 - phi) / Km - Kdry / Km ** 2)


def wood(sat, K, rho):
    return 1.0 / sum(s / k for s, k in zip(sat, K)), sum(s * r for s, r in zip(sat, rho))


def velocities(K, G, rho):
    """Eq. (7): GPa and g/cm3 -> m/s."""
    return np.sqrt((K + 4.0 / 3.0 * G) / rho) * 1e3, np.sqrt(G / rho) * 1e3


def rhg(phi, vp_matrix, vp_fluid, rho_matrix, rho_fluid):
    """Raymer-Hunt-Gardner velocity (m/s) with the 0.37-0.47 harmonic transition."""
    phi = np.asarray(phi, float)
    rho = (1 - phi) * rho_matrix + phi * rho_fluid
    v_low = (1 - phi) ** 2 * vp_matrix + phi * vp_fluid

    def wood_v(p, r):
        return np.sqrt(1.0 / (r * (p / (rho_fluid * vp_fluid ** 2) + (1 - p) / (rho_matrix * vp_matrix ** 2))))
    v37 = (1 - 0.37) ** 2 * vp_matrix + 0.37 * vp_fluid
    v47 = wood_v(0.47, (1 - 0.47) * rho_matrix + 0.47 * rho_fluid)
    v_mid = 1.0 / ((0.47 - phi) / (0.10 * v37) + (phi - 0.37) / (0.10 * v47))
    v = np.where(phi < 0.37, v_low, np.where(phi > 0.47, wood_v(phi, rho), v_mid))
    return v, rho


def hashin_shtrikman(K1, G1, K2, G2, f1):
    """Upper bound when material 1 is the stiffer one."""
    f2 = 1 - f1
    K = K1 + f2 / (1 / (K2 - K1) + f1 / (K1 + 4 / 3 * G1))
    G = G1 + f2 / (1 / (G2 - G1) + 2 * f1 * (K1 + 2 * G1) / (5 * G1 * (K1 + 4 / 3 * G1)))
    return K, G


def elastic_properties(div: dict, cfg: dict, fluid_sat_gas=None):
    """Layer-wise Vp, Vs (m/s) and rho (g/cm3) from diagenesis output."""
    cal, dol, cla = (np.asarray(cfg[k], float) for k in ("calcite", "dolomite", "clay"))
    Km, Gm, rhom = vrh([div["calcite"], div["dolomite"], div["clay"]],
                       [cal[0], dol[0], cla[0]], [cal[1], dol[1], cla[1]], [cal[2], dol[2], cla[2]])
    phi = div["phi"]
    brine, gas = cfg["brine"], cfg["gas"]
    if fluid_sat_gas is None:
        Kfl, rhofl = np.full_like(phi, brine[0]), np.full_like(phi, brine[2])
    else:
        sg = fluid_sat_gas
        Kfl, rhofl = wood([sg, 1 - sg], [gas[0], brine[0]], [gas[2], brine[2]])
    rho = (1 - phi) * rhom + phi * rhofl
    if cfg.get("model", "xu_payne") == "rhg":
        vpm = np.sqrt((Km + 4 / 3 * Gm) / rhom) * 1e3
        vpf = np.sqrt(Kfl / rhofl) * 1e3
        vp, rho = rhg(phi, vpm, vpf, rhom, rhofl)
        vs = vp / 1.9
        return vp, vs, rho
    Kd, Gd = dem_dry(Km, Gm, [div["phi_ip"], div["phi_stiff"], div["phi_crack"]],
                     cfg["aspect_ratios"], cfg.get("dem_steps", 20))
    Ks = gassmann(Kd, Km, Kfl, phi)
    vp, vs = velocities(Ks, Gd, rho)
    return vp, vs, rho
