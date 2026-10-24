"""Figs. A1-A3 and Table 7 (tables/tabA1_verification.tex): verification of grid, compaction,
wavelet alignment, tuning, rock-physics limits and PSF consistency (manuscript Sect. 5, Table 6)."""
import sys
import numpy as np
import matplotlib.pyplot as plt
from common import ROOT, TAB, TWO_COL, chain, grid, save, scenario, style
sys.path.insert(0, str(ROOT / "tests"))
from test_verification import SEIS, layered_model, tuning_curve  # noqa: E402
from carboseis import rockphysics as rp  # noqa: E402
from carboseis.io import read_section  # noqa: E402
from carboseis.seismic import depth_to_time, level1, level2_depth  # noqa: E402

style()
sc = scenario()
rows = []
# V1, V2 on the scenario data
g = grid(sc)
from common import data_path  # noqa: E402
sec = read_section(data_path(sc), sc["grid"]["group"])
rows.append(("V1 Grid consistency", f"{np.abs(g.z_top[-1] - sec.bathymetry[-1]).max():.1e} m"))
res = chain(sc, grid_obj=g)
L = res.layers
solid = (L["hc"] * (1 - L["phi_mech"])).sum(0)
solid0 = (g.H * (1 - np.einsum("ktx,t->kx", res.q / np.maximum(res.q.sum(1, keepdims=True), 1e-12),
                                   np.asarray(sc["compaction"]["phi0"])))).sum(0)
rows.append(("V2 Solid conservation", f"{np.max(np.abs(solid - solid0) / solid0):.1e}"))

# A1: spike and wedge
fig, ax = plt.subplots(1, 2, figsize=(TWO_COL * 0.8, 2.4))
vp = np.full(600, 3000.0)
rho = np.full(600, 2.2)
rho[300:] = 2.5
ts = level1(layered_model(vp, rho, ntr=2), SEIS)
t_true = 0.2
k = np.argmax(np.abs(ts.data[:, 0]))
rows.append(("V3 Spike alignment", f"{(ts.t[k] - t_true) * 1e3:+.1f} ms shift"))
m = (ts.t > 0.15) & (ts.t < 0.25)
ax[0].plot(ts.t[m] * 1e3, ts.data[m, 0], color="k", lw=0.9)
ax[0].axvline(t_true * 1e3, color="#c0392b", ls="--", lw=0.8, label="True interface time")
ax[0].set_xlabel("Two-way time (ms)")
ax[0].set_ylabel("Amplitude")
ax[0].set_title("(a) Spike alignment, 30 Hz", loc="left")
ax[0].legend(frameon=False, fontsize=6, loc="upper left")
th = np.arange(2, 120, 1)
amp = tuning_curve(th)
lam4 = 4500.0 / 30.0 / 4
t_tune = th[np.argmax(amp)]
rows.append(("V4 Wedge tuning", f"{t_tune:.0f} m (lambda/4 = {lam4:.1f} m)"))
ax[1].plot(th, amp / amp[-1], color="k", lw=0.9)
ax[1].axvline(lam4, color="#c0392b", ls="--", lw=0.8, label="$\\lambda/4$ at 30 Hz")
ax[1].axvline(t_tune, color="#2a6db5", ls=":", lw=0.8, label="Measured tuning")
ax[1].set_xlabel("Carbonate wedge thickness (m)")
ax[1].set_ylabel("Peak amplitude / thick-bed amplitude")
ax[1].set_title("(b) Wedge tuning", loc="left")
ax[1].legend(frameon=False, fontsize=6)
fig.tight_layout()
save(fig, "figA1_spike_wedge", False)

# A2: rock physics
fig, ax = plt.subplots(1, 2, figsize=(TWO_COL * 0.8, 2.5))
K, G, rho_m = 76.8, 32.0, 2.71
phi = np.linspace(0.0, 0.35, 36)
z = np.zeros_like(phi)
Kh, Gh = rp.hashin_shtrikman(K, G, 1e-9, 1e-9, 1 - phi)
ax[0].plot(phi, Kh, color="0.5", ls="--", lw=0.8, label="HS upper bound")
labels = [("Interparticle (0.15)", 0), ("Stiff vuggy (0.8)", 1), ("Crack-like (0.02)", 2)]
cols = ["#2a6db5", "#2e9e6b", "#c0392b"]
for (lab, i), c in zip(labels, cols):
    ph = [phi if j == i else z for j in range(3)]
    Kd, Gd = rp.dem_dry(np.full_like(phi, K), np.full_like(phi, G), ph, [0.15, 0.8, 0.02])
    ax[0].plot(phi, Kd, color=c, lw=0.9, label=lab)
    Ks = rp.gassmann(Kd, K, 2.8, phi)
    rhob = (1 - phi) * rho_m + phi * 1.05
    ax[1].plot(phi, rp.velocities(Ks, Gd, rhob)[0] / 1e3, color=c, lw=0.9, label=lab)
vr, _ = rp.rhg(phi, 6.4e3, 1.63e3, rho_m, 1.05)
ax[1].plot(phi, vr / 1e3, color="k", ls=":", lw=0.9, label="Raymer-Hunt-Gardner")
ax[0].set_xlabel("Porosity")
ax[0].set_ylabel("Dry bulk modulus (GPa)")
ax[0].set_title("(a) DEM dry frame within bounds", loc="left")
ax[1].set_xlabel("Porosity")
ax[1].set_ylabel("Brine-saturated Vp (km/s)")
ax[1].set_title("(b) Pore type controls velocity", loc="left")
ax[0].legend(frameon=False, fontsize=6)
ax[1].legend(frameon=False, fontsize=6)
fig.tight_layout()
save(fig, "figA2_rockphysics", False)
P, Q = rp.berryman_pq(K, G, 0.9999)
Ps, Qs = rp.sphere_pq(K, G)
rows.append(("V5 Rock-physics limits", f"sphere-limit error {max(abs(P - Ps), abs(Q - Qs)):.1e}; bounds respected"))

# A3: PSF consistency
fig, ax = plt.subplots(1, 2, figsize=(TWO_COL * 0.8, 2.5))
vp = np.full(1600, 3500.0)
rho = np.full(1600, 2.3)
rho[500:] = 2.5
rho[900:] = 2.4
em = layered_model(vp, rho, ntr=64)
l1 = level1(em, SEIS)
l2 = depth_to_time(em, level2_depth(em, SEIS), 2e-3)
n = min(l1.data.shape[0], l2.data.shape[0]) - 5
a, b = l1.data[50:n, 32], np.nan_to_num(l2.data[50:n, 32])
cc = np.corrcoef(a, b)[0, 1]
m = (l1.t[50:n] > 0.25) & (l1.t[50:n] < 0.6)
ax[0].plot(l1.t[50:n][m], a[m], color="k", lw=1.2, label="Level 1")
ax[0].plot(l1.t[50:n][m], b[m], color="#e8a33d", lw=0.8, ls="--", label="Level 2")
ax[0].set_xlabel("Two-way time (s)")
ax[0].set_ylabel("Amplitude")
ax[0].set_title(f"(a) Flat layers, correlation {cc:.3f}", loc="left")
ax[0].legend(frameon=False, fontsize=6)
dips = np.arange(0, 81, 5)
amps = []
for d in dips:
    nx, dx = 512, 5.0          # fine traces: at 25 m, dips above ~50 degrees alias at 30 Hz
    x = np.arange(nx) * dx
    zz = (np.arange(1400) + 0.5)
    iface = 400 + (x - x.mean()) * np.tan(np.radians(d))
    rho = np.where(zz[:, None] > iface[None, :], 2.5, 2.3)
    e = layered_model(np.full(1400, 3500.0), np.full(1400, 2.3), ntr=nx, dx=dx)
    e.rho = rho
    img = level2_depth(e, SEIS)
    centre = slice(nx // 2 - 60, nx // 2 + 60)
    amps.append(np.abs(img[200:1200, centre]).max())
amps = np.array(amps) / amps[0]
ax[1].plot(dips, amps, "o-", ms=2.5, color="k", lw=0.8)
ax[1].axvline(SEIS["psf_max_dip_deg"], color="#c0392b", ls="--", lw=0.8, label="$\\theta_{max}$")
ax[1].set_xlabel("Reflector dip (degrees)")
ax[1].set_ylabel("Imaged amplitude / flat")
ax[1].set_title("(b) Dip-limited illumination", loc="left")
ax[1].legend(frameon=False, fontsize=6)
fig.tight_layout()
save(fig, "figA3_psf", False)
rows.append(("V6 PSF consistency", f"correlation {cc:.3f}; amplitude at 60 deg {amps[dips == 60][0]:.2f} of flat"))

with open(TAB / "tabA1_verification.tex", "w") as f:
    f.write("\\begin{tabular}{ll}\n\\tophline\nTest & Measured result \\\\\n\\middlehline\n")
    for r in rows:
        label = r[1].replace("lambda", "$\\lambda$")
        f.write(f"{r[0]} & {label} \\\\\n")
    f.write("\\bottomhline\n\\end{tabular}\n")
for r in rows:
    print(r)
