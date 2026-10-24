"""Fig. 6: E1 synthetic sections (Level 2) for property scenarios DC, D0, D1, D2,
with the flooding-surface time-lines."""
import numpy as np
import matplotlib.pyplot as plt
from common import TWO_COL, platform_time_window, save, seismic_panel, style
from e1_common import LABELS, run_all
from carboseis.grid import flooding_surface_frames
from carboseis.pipeline import depth_to_twt

style()
runs = run_all()
sc0, r0 = runs["D2"]
fs = flooding_surface_frames(r0.grid.t, r0.grid.sea_level)
t0, t1 = platform_time_window(r0, pad=0.03)
fig, axs = plt.subplots(4, 2, figsize=(TWO_COL, 6.9), sharex="col",
                        gridspec_kw={"width_ratios": [1, 0.018]})
ax = axs[:, 0]
for b in axs[1:, 1]:
    b.axis("off")
clip = None
for a, (name, (sc, res)) in zip(ax, runs.items()):
    im, c = seismic_panel(a, res.l2, t0, t1, clip)
    clip = clip or c
    tw = depth_to_twt(res.em, res.em.layer_top_depth[fs])
    for row in tw:
        a.plot(res.em.x / 1e3, row, color="k", lw=0.35, alpha=0.6)
    a.set_ylim(t1, t0)
    a.set_title(f"({'abcd'[list(runs).index(name)]}) {LABELS[name]}", loc="left")
    if a is ax[0]:
        fig.colorbar(im, cax=axs[0, 1], label="Amplitude")
ax[-1].set_xlabel("Distance (km)")
for a in ax[1:]:
    a.set_ylabel("Two-way time (s)")
fig.tight_layout()
save(fig, "fig06_e1_sections", sc0["_use_standin"])
