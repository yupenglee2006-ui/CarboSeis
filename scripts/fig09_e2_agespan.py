"""Fig. 9: age span of tracked reflections versus frequency, imaging level and position."""
import csv
import numpy as np
import matplotlib.pyplot as plt
from common import TAB, TWO_COL, save, style
from e2_common import FREQS, ZONES, run_all

style()
sc, out, layer_dt = run_all()
fig, ax = plt.subplots(1, 4, figsize=(TWO_COL, 2.6), sharey=True)
rows = []
for a, (zc, zname) in zip(ax, ZONES.items()):
    for k, lev in enumerate(("Level 1", "Level 2")):
        data, pos = [], []
        for i, f0 in enumerate(FREQS):
            st = out[(f0, lev)]
            v = st[st[:, 2] == zc, 0] if st.size else np.array([])
            data.append(v if v.size else np.array([np.nan]))
            pos.append(i * 3 + k)
            fl = st[st[:, 2] == zc, 4] if st.size else np.array([])
            rows.append([zname, lev, f0, v.size, np.nanmedian(v) if v.size else np.nan,
                         np.nanmedian(st[st[:, 2] == zc, 1]) if v.size else np.nan,
                         np.nanmedian(fl) if fl.size else np.nan,
                         np.nanmedian(v / fl) if v.size else np.nan])
            if fl.size and np.isfinite(fl).any():
                a.plot([pos[-1] - 0.4, pos[-1] + 0.4], [np.nanmedian(fl)] * 2, color="k", lw=1.6,
                       solid_capstyle="butt", zorder=5)
        bp = a.boxplot(data, positions=pos, widths=0.8, showfliers=False, patch_artist=True,
                       medianprops=dict(color="k"))
        for b in bp["boxes"]:
            b.set_facecolor("#9fd3c7" if k == 0 else "#f2c9a0")
    a.set_xticks([i * 3 + 0.5 for i in range(len(FREQS))], [f"{f:.0f} Hz" for f in FREQS])
    a.set_title(f"({'abcd'[zc - 1]}) {zname.capitalize()}", loc="left")
    a.set_yscale("log")
ax[0].set_ylabel("Age span of reflection (kyr)")
ax[0].legend([plt.Rectangle((0, 0), 1, 1, fc="#9fd3c7"), plt.Rectangle((0, 0), 1, 1, fc="#f2c9a0"),
              plt.Line2D([], [], color="k", lw=1.6)],
             ["Level 1", "Level 2", "Age range of one sample"], frameon=False, loc="upper right", fontsize=6)
fig.tight_layout()
save(fig, "fig09_e2_agespan", sc["_use_standin"])
with open(TAB / "e2_agespan.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["zone", "level", "frequency_Hz", "n_reflections", "median_age_span_kyr",
                "median_fraction_within_one_interval", "median_age_per_sample_kyr",
                "median_span_in_samples"])
    for r in rows:
        w.writerow([r[0], r[1], r[2], r[3], f"{r[4]:.1f}", f"{r[5]:.2f}", f"{r[6]:.1f}", f"{r[7]:.2f}"])
print(open(TAB / "e2_agespan.csv").read())
