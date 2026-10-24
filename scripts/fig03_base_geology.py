"""Fig. 3: base case in depth - dominant facies, dominant texture, porosity."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from common import (FACIES_COLORS, TEXTURE_COLORS, TEXTURE_LABELS, TWO_COL, chain, depth_window,
                    save, scenario, style)

style()
sc = scenario()
res = chain(sc)
em = res.em
z0, z1 = depth_window(res)
m = (em.z >= z0) & (em.z <= z1)
ext = [em.x[0] / 1e3, em.x[-1] / 1e3, z1, z0]
fig, axs = plt.subplots(3, 2, figsize=(TWO_COL, 6.4), gridspec_kw={"width_ratios": [1, 0.018]})
ax = axs[:, 0]
for a in axs[:2, 1]:
    a.axis("off")
names = res.grid.facies_names
for a, key, cols, labels, title in [
        (ax[0], "facies", FACIES_COLORS, names, "(a) Dominant CarboKitten facies"),
        (ax[1], "texture", TEXTURE_COLORS, TEXTURE_LABELS, "(b) Dominant texture")]:
    f = em.fields[key][m]
    a.set_facecolor("0.93")
    a.imshow(np.ma.masked_invalid(np.where(f >= 0, f, np.nan)), aspect="auto", extent=ext,
             cmap=ListedColormap(cols), vmin=-0.5, vmax=len(cols) - 0.5, interpolation="nearest")
    a.set_title(title, loc="left")
    a.set_ylabel("Depth (m)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cols]
    a.legend(handles, labels, ncol=len(cols), frameon=False, fontsize=6, loc="lower right",
             bbox_to_anchor=(1.0, 0.98), handlelength=1.0, columnspacing=0.8)
phi = em.fields["phi"][m]
a = ax[2]
a.set_facecolor("0.93")
im = a.imshow(np.ma.masked_invalid(phi), aspect="auto", extent=ext, cmap="viridis", vmin=0.0, vmax=0.3)
a.set_title("(c) Porosity after compaction and diagenesis", loc="left")
a.set_ylabel("Depth (m)")
a.set_xlabel("Distance (km)")
fig.colorbar(im, cax=axs[2, 1], label="Porosity")
for b in ax[:2]:
    b.set_xticklabels([])
fig.tight_layout()
save(fig, "fig03_base_geology", sc["_use_standin"])
