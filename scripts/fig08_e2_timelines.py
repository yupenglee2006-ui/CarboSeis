"""Fig. 8: Level 2 synthetic section with the model's true time-lines coloured by age."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from common import TWO_COL, chain, grid, platform_time_window, save, scenario, seismic_panel, style
from carboseis.pipeline import depth_to_twt

style()
sc = scenario()
g = grid(sc)
res = chain(sc, grid_obj=g, f0=30.0)
t0, t1 = platform_time_window(res, pad=0.03)
fig, ax = plt.subplots(2, 1, figsize=(TWO_COL, 5.0), sharex=True)
for a, show in zip(ax, (False, True)):
    im, clip = seismic_panel(a, res.l2, t0, t1)
    if show:
        step = max(1, int(round(0.25 / np.median(np.diff(g.t)))))     # every 250 kyr
        ks = np.arange(1, len(g.t), step)
        tw = depth_to_twt(res.em, res.em.layer_top_depth[ks])
        x = res.em.x / 1e3
        segs = [np.column_stack([x, row]) for row in tw]
        lc = LineCollection(segs, cmap="viridis", linewidths=0.9)
        lc.set_array(-g.t[ks])                                        # age in Ma
        lc.set_cmap("viridis_r")
        a.add_collection(lc)
        fig.colorbar(lc, ax=a, fraction=0.02, pad=0.01, label="Time-line age (Ma)")
    else:
        fig.colorbar(im, ax=a, fraction=0.02, pad=0.01, label="Amplitude")
    a.set_ylim(t1, t0)
ax[0].set_title("(a) Level 2 synthetic seismic, 30 Hz", loc="left")
ax[1].set_title("(b) Same section with true time-lines every 250 kyr", loc="left")
ax[1].set_xlabel("Distance (km)")
fig.tight_layout()
save(fig, "fig08_e2_timelines", sc["_use_standin"])
