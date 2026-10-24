"""Fig. 5: base case synthetic seismic in two-way time - Level 1 and Level 2."""
import matplotlib.pyplot as plt
from common import TWO_COL, chain, platform_time_window, save, scenario, seismic_panel, style

style()
sc = scenario()
res = chain(sc)
t0, t1 = platform_time_window(res)
f0 = sc["seismic"]["dominant_frequency_Hz"]
fig, ax = plt.subplots(2, 1, figsize=(TWO_COL, 4.6), sharex=True)
im, clip = seismic_panel(ax[0], res.l1, t0, t1)
ax[0].set_title(f"(a) Level 1: 1-D convolution, {f0:.0f} Hz Ricker", loc="left")
seismic_panel(ax[1], res.l2, t0, t1, clip)
ax[1].set_title(f"(b) Level 2: PSF imaging, maximum illuminated dip "
                f"{sc['seismic']['psf_max_dip_deg']:.0f} degrees", loc="left")
ax[1].set_xlabel("Distance (km)")
for a in ax:
    fig.colorbar(im, ax=a, fraction=0.02, pad=0.01, label="Amplitude")
fig.tight_layout()
save(fig, "fig05_base_seismic", sc["_use_standin"])
