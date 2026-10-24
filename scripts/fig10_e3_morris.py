"""Fig. 10: Morris screening, mu* versus sigma, for two seismic statistics."""
import matplotlib.pyplot as plt
import numpy as np
from common import TWO_COL, save, style
from e3_common import load

style()
r = load()
fig, ax = plt.subplots(1, 2, figsize=(TWO_COL * 0.85, 2.9))
groups = ["#2a6db5"] * 5 + ["#2e9e6b"] * 4 + ["#c0392b"] * 2 + ["#7a7a7a"]
for a, s in zip(ax, (0, 2)):
    sd = np.sqrt(np.nanvar(r["seeds"][:, s], ddof=1))
    a.axvspan(0, sd, color="0.85", lw=0)
    for j, (m, sg) in enumerate(zip(r["mu"][:, s], r["sigma"][:, s])):
        a.plot(m, sg, "o", ms=4, color=groups[j])
        a.annotate(str(j + 1), (m, sg), textcoords="offset points", xytext=(3, 2), fontsize=6)
    a.set_xlabel("$\\mu^*$ (mean |change| per Morris step)")
    a.set_ylabel("$\\sigma$ (interaction / nonlinearity)")
    a.set_title(f"({'ab'[s // 2]}) {r['stats'][s]}", loc="left")
handles = [plt.Line2D([], [], marker="o", ls="", color=c) for c in ("#2a6db5", "#2e9e6b", "#c0392b", "#7a7a7a")]
ax[1].legend(handles + [plt.Rectangle((0, 0), 1, 1, fc="0.85", ec="0.6")],
             ["Sedimentological", "Diagenesis/compaction", "Rock physics/fluid", "Acquisition",
              "CA-seed standard deviation"], frameon=False, fontsize=6, loc="upper left")
fig.tight_layout()
if r["partial"]:
    fig.text(0.5, 0.5, f"PARTIAL E3 - {r['n_traj']} of {r['r_planned']} trajectories", ha="center",
             va="center", rotation=20, fontsize=14, color="red", alpha=0.25, transform=fig.transFigure)
save(fig, "fig10_e3_morris", r["engine"] == "standin")
