"""Fig. 11: detectability index D (parameter-induced variance / CA-seed variance)."""
import csv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from common import TAB, TWO_COL, save, style
from e3_common import load

style()
r = load()
L = np.log10(np.clip(r["D"], 1e-3, 1e4))
fig, ax = plt.subplots(figsize=(TWO_COL * 0.85, 4.8))
im = ax.imshow(L, cmap="PuOr_r", norm=TwoSlopeNorm(0.0, -3, 4), aspect="auto")
for j in range(L.shape[0]):
    for s in range(L.shape[1]):
        rob = bool(r["robust"][j, s])
        ax.text(s, j, f"{r['D'][j, s]:.2g}" + ("*" if rob else ""), ha="center", va="center",
                fontsize=6.0, fontweight="bold" if rob else "normal",
                color="k" if abs(L[j, s]) < 2 else "w")
ax.set_yticks(range(len(r["params"])), [f"{i + 1}. {p}" for i, p in enumerate(r["params"])], fontsize=8)
ax.set_xticks(range(len(r["stats"])), r["stats"], rotation=35, ha="right", fontsize=8)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, label="log$_{10}$ D   (D > 1: detectable)")
cb.set_label("log$_{10}$ D   (D > 1: detectable)", fontsize=8)
cb.ax.tick_params(labelsize=7.5)
# split over two or three lines: the single line was wider than the panel
title = "Detectability index (* robust)\n" + \
        f"{r['n_traj']} Morris trajectories, {r['n_seeds']} seed replicates"
if r["n_failed"]:
    title += f"\n{r['n_failed']} of {r['n_evals']} evaluations excluded"
ax.set_title(title, loc="left", fontsize=8)
fig.tight_layout()
if r["partial"]:
    fig.text(0.5, 0.5, f"PARTIAL E3 - {r['n_traj']} of {r['r_planned']} trajectories", ha="center",
             va="center", rotation=20, fontsize=14, color="red", alpha=0.25, transform=fig.transFigure)
save(fig, "fig11_e3_detectability", r["engine"] == "standin")
with open(TAB / "e3_detectability.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["parameter"] + r["stats"] + ["n_statistics_detectable", "n_elementary_effects_min"])
    for p, row, n in zip(r["params"], r["D"], r["n_eff"]):
        w.writerow([p] + [f"{v:.3g}" for v in row] + [int((row > 1).sum()), int(n.min())])
with open(TAB / "e3_detectability_ci.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["parameter", "statistic", "D", "D_bootstrap_p05", "D_bootstrap_p95", "robust"])
    for j, p in enumerate(r["params"]):
        for s_, st in enumerate(r["stats"]):
            w.writerow([p, st, f"{r['D'][j, s_]:.3g}", f"{r['D_lo'][j, s_]:.3g}", f"{r['D_hi'][j, s_]:.3g}",
                        int(r["robust"][j, s_])])
dm = r["domain"]
print(f"robust detections (bootstrap 5th percentile > 1): {int(r['robust'].any(axis=1).sum())} parameters")
print(f"failed forward combinations: {dm['n_failed']} of {dm['n_combos']}; in the corner transport >= "
      f"{dm['transport_min']} and growth >= {dm['growth_min']}: {dm['corner_failed']} of {dm['corner']}; "
      f"elsewhere {dm['rest_failed']} of {dm['rest']}")
print(f"failed evaluations: {r['n_failed']} of {r['n_evals']}; seed replicates used: {r['n_seeds']}")
print("parameters detectable for at least one statistic:", int((r["D"] > 1).any(axis=1).sum()), "of", len(r["params"]))
