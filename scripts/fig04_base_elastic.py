"""Fig. 4: base case in depth - Vp, acoustic impedance, reflectivity."""
import numpy as np
import matplotlib.pyplot as plt
from common import TWO_COL, chain, depth_window, save, scenario, style
from carboseis.seismic import reflectivity

style()
sc = scenario()
res = chain(sc)
em = res.em
z0, z1 = depth_window(res)
m = (em.z >= z0) & (em.z <= z1)
ext = [em.x[0] / 1e3, em.x[-1] / 1e3, z1, z0]
Z = em.impedance * 1e3 / 1e6           # g/cm3 * m/s -> 10^6 kg m^-2 s^-1
R = reflectivity(em.impedance)
fig, ax = plt.subplots(3, 1, figsize=(TWO_COL, 6.2), sharex=True)
panels = [(em.vp[m] / 1e3, "viridis", None, "Vp (km/s)", "(a) P-wave velocity"),
          (Z[m], "magma", None, "Impedance (10$^6$ kg m$^{-2}$ s$^{-1}$)", "(b) Acoustic impedance"),
          (R[m], "RdBu", 0.03, "Reflection coefficient", "(c) Normal-incidence reflectivity")]
for a, (d, cmap, clip, lab, title) in zip(ax, panels):
    kw = dict(vmin=-clip, vmax=clip) if clip else {}
    im = a.imshow(d, aspect="auto", extent=ext, cmap=cmap, interpolation="nearest", **kw)
    a.set_title(title, loc="left")
    a.set_ylabel("Depth (m)")
    fig.colorbar(im, ax=a, fraction=0.02, pad=0.01, label=lab)
ax[-1].set_xlabel("Distance (km)")
fig.tight_layout()
save(fig, "fig04_base_elastic", sc["_use_standin"])
