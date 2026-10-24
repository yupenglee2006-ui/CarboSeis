"""Fig. 2: forward-model forcing of the example application - sea level with flooding surfaces, initial and final topography,
production curves of the CarboKitten factories."""
import numpy as np
import matplotlib.pyplot as plt
from common import FACIES_COLORS, TWO_COL, grid, save, scenario, style
from carboseis.grid import flooding_surface_frames
from carboseis.standin import bs92

style()
sc = scenario()
g = grid(sc)
fs = flooding_surface_frames(g.t, g.sea_level)
fig, ax = plt.subplots(1, 3, figsize=(TWO_COL, 2.4), gridspec_kw={"width_ratios": [1.3, 1.6, 1.0]})
a = ax[0]
a.plot(-g.t, g.sea_level, color="#2a6db5", lw=0.8)
a.plot(-g.t[fs], g.sea_level[fs], "v", ms=3.5, color="#c0392b", label="Flooding surfaces")
a.invert_xaxis()                                    
a.set_xlabel("Age (Ma)")
a.set_ylabel("Sea level (m)")
a.set_title("(a) Sea level, validation curve", loc="left")
a.legend(frameon=False, loc="lower left")
b = ax[1]
x = g.x / 1e3
eta0 = g.z_base + sc["carbokitten"]["subsidence_rate_m_per_Myr"] * (g.t[-1] - g.t[0])
b.plot(x, eta0, color="0.4", lw=0.8, label="Initial topography")
b.plot(x, g.z_base, color="0.1", lw=0.8, ls="--", label="Initial topography after subsidence")
b.plot(x, g.z_top[-1], color="#e8a33d", lw=1.0, label="Final carbonate top (uncompacted)")
b.set_xlabel("Distance (km)")
b.set_ylabel("Elevation (m)")
b.set_title("(b) Section geometry", loc="left")
b.legend(frameon=False, fontsize=6, loc="center", bbox_to_anchor=(0.5, 0.42))
c = ax[2]
w = np.linspace(0, 60, 300)
I0 = sc["carbokitten"]["insolation_W_m2"]
for f, col in zip(sc["carbokitten"]["facies"], FACIES_COLORS):
    if f["type"] == "benthic":
        c.plot(bs92(w, f["maximum_growth_rate_m_per_Myr"], f["extinction_coefficient_per_m"],
                    f["saturation_intensity_W_m2"], I0), w, color=col, label=f["name"])
c.invert_yaxis()
c.set_xlabel("Production (m/Myr)")
c.set_ylabel("Water depth (m)")
c.set_title("(c) Benthic production", loc="left")
c.legend(frameon=False)
fig.tight_layout()
save(fig, "fig02_scenario", sc["_use_standin"])
