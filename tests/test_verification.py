"""Verification tests V1-V6 (manuscript Sect. 5, Tables 6 and 7)."""
import numpy as np
import pytest

from carboseis import rockphysics as rp
from carboseis.earthmodel import EarthModel
from carboseis.grid import stratigraphic_column
from carboseis.properties import compact
from carboseis.seismic import level1, level2_depth, depth_to_time

SEIS = {"wavelet": "ricker", "dominant_frequency_Hz": 30.0, "dt_ms": 2.0, "dt_fine_ms": 0.25,
        "psf_max_dip_deg": 45.0, "psf_window_m": 400.0}


def layered_model(vp_profile, rho_profile, ntr=64, dx=25.0, dz=1.0):
    nz = len(vp_profile)
    vp = np.repeat(np.asarray(vp_profile, float)[:, None], ntr, 1)
    rho = np.repeat(np.asarray(rho_profile, float)[:, None], ntr, 1)
    return EarthModel(x=np.arange(ntr) * dx, z=(np.arange(nz) + 0.5) * dz, dz=dz, vp=vp, vs=vp / 1.9,
                      rho=rho, unit=np.zeros((nz, ntr), int), layer=-np.ones((nz, ntr), int), fields={},
                      layer_top_depth=np.zeros((1, ntr)), layer_base_depth=np.zeros((1, ntr)),
                      platform_top_depth=0.0)


def test_v1_stack_algorithm_matches_carbokitten():
    rng = np.random.default_rng(0)
    net = rng.normal(0.2, 1.0, size=(200, 3, 5))
    out = stratigraphic_column(net)
    a = np.cumsum(net, axis=0)                     # literal port of CarboKitten's loop
    for i in range(a.shape[0] - 1, 0, -1):
        a[i - 1] = np.minimum(a[i - 1], a[i])
        a[i] -= a[i - 1]
    assert np.allclose(out, a)
    assert np.allclose(out.sum(0), net.sum(0))


def test_v2_compaction_conserves_solid():
    rng = np.random.default_rng(1)
    H = rng.uniform(0, 2, size=(300, 7))
    q = rng.dirichlet(np.ones(6), size=(300, 7)).transpose(0, 2, 1)
    cfg = {"phi0": [0.40, 0.45, 0.55, 0.65, 0.70, 0.65], "c_per_km": [0.1, 0.25, 0.4, 0.5, 0.55, 0.6]}
    hc, phi, zmid, s = compact(H, q, cfg, np.full(7, 2000.0))
    rel = np.abs((hc * (1 - phi)).sum(0) - s.sum(0)) / s.sum(0)
    assert rel.max() < 1e-9
    assert np.all(hc <= H + 1e-12)


def test_v3_spike_alignment():
    vp = np.full(600, 3000.0)
    rho = np.full(600, 2.2)
    rho[300:] = 2.5                                 # single interface at 300 m
    em = layered_model(vp, rho, ntr=4)
    ts = level1(em, SEIS)
    t_true = 2 * 300.0 / 3000.0
    k = np.argmax(np.abs(ts.data[:, 0]))
    assert abs(ts.t[k] - t_true) < 1e-9


def tuning_curve(thicknesses, f0=30.0, v_shale=3000.0, v_carb=4500.0):
    amps = []
    for th in thicknesses:
        vp = np.full(1200, v_shale)
        rho = np.full(1200, 2.3)
        vp[400:400 + int(th)] = v_carb
        rho[400:400 + int(th)] = 2.6
        ts = level1(layered_model(vp, rho, ntr=2), dict(SEIS, dominant_frequency_Hz=f0))
        amps.append(np.abs(ts.data[:, 0]).max())
    return np.array(amps)


def test_v4_wedge_tuning_near_quarter_wavelength():
    th = np.arange(5, 90, 1)
    a = tuning_curve(th)
    lam4 = 4500.0 / 30.0 / 4.0
    t_tune = th[np.argmax(a)]
    # Ricker tuning thickness ~ lambda_peak / 4 to within ~30 %
    assert 0.6 * lam4 < t_tune < 1.3 * lam4


def test_v5_rock_physics_limits():
    K, G = 76.8, 32.0
    P, Q = rp.berryman_pq(K, G, 0.9999)
    Ps, Qs = rp.sphere_pq(K, G)
    assert abs(P - Ps) < 1e-6 and abs(Q - Qs) < 1e-6
    for phi in (0.05, 0.15, 0.3):
        Kd, Gd = rp.dem_dry(np.array(K), np.array(G), [np.array(phi), np.array(0.0), np.array(0.0)], [0.15, 0.8, 0.02])
        Kh, Gh = rp.hashin_shtrikman(K, G, 1e-9, 1e-9, 1 - phi)
        assert Kd <= Kh + 1e-9 and Gd <= Gh + 1e-9
    Kd, Gd = rp.dem_dry(np.array(K), np.array(G), [np.array(0.0)] * 3, [0.15, 0.8, 0.02])
    assert abs(Kd - K) < 1e-9 and abs(Gd - G) < 1e-9
    # Gassmann: dry rock with Kdry = Km stays Km; fluid always stiffens
    assert abs(rp.gassmann(K, K, 2.8, 0.2) - K) < 1e-9
    assert rp.gassmann(20.0, K, 2.8, 0.2) > 20.0


def test_v6_psf_equals_convolution_for_flat_layers():
    vp = np.full(1600, 3500.0)
    rho = np.full(1600, 2.3)
    rho[500:] = 2.5
    rho[900:] = 2.4
    em = layered_model(vp, rho, ntr=64)
    l1 = level1(em, SEIS)
    img = level2_depth(em, SEIS)
    l2 = depth_to_time(em, img, SEIS["dt_ms"] * 1e-3)
    n = min(l1.data.shape[0], l2.data.shape[0]) - 5
    a, b = l1.data[50:n, 32], np.nan_to_num(l2.data[50:n, 32])
    assert np.corrcoef(a, b)[0, 1] > 0.99
    assert abs(np.abs(a).max() - np.abs(b).max()) / np.abs(a).max() < 0.05
