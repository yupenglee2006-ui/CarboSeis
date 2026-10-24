"""Fig. 7: E1 reflection strength and lateral continuity along flooding surfaces."""
import csv
import numpy as np
import matplotlib.pyplot as plt
from common import ONE_COL, TAB, save, style
from e1_common import run_all
from carboseis.analysis import e1_metrics, e1_reference_horizons, sample_at, zones
from carboseis.pipeline import depth_to_twt
from carboseis.grid import flooding_surface_frames

style()
runs = run_all()
rows = []
for name, (sc, res) in runs.items():
    fs = flooding_surface_frames(res.grid.t, res.grid.sea_level)
    cls = zones(res.em)
    m = e1_metrics(res.l2, res.em, fs, cls)
    ref = e1_reference_horizons(res.l2, res.em, fs, cls)
    fin = lambda a: a[np.isfinite(a)]
    top = depth_to_twt(res.em, res.em.layer_top_depth.min(axis=0)[None, :])
    rms_top = m["rms"] / float(np.nanmedian(sample_at(res.l2, top)[0, cls == 1]))
    rows.append((name, fin(m["strength"]), fin(m["continuity"]), fin(ref["strength"]), fin(ref["continuity"]), rms_top))
fig, ax = plt.subplots(1, 2, figsize=(ONE_COL * 1.6, 2.3))
names = [r[0] for r in rows]
for a, k, lab in [(ax[0], 1, "Reflection strength\n(amplitude / platform RMS)"),
                  (ax[1], 2, "Lateral continuity\n(fraction of interior traces)")]:
    pos = np.arange(1, len(rows) + 1)
    ref = [r[k + 2] for r in rows]
    vals = [r[k] for r in rows]
    a.boxplot(ref, positions=pos + 0.22, widths=0.3, showfliers=False, patch_artist=True,
              boxprops=dict(facecolor="0.88", edgecolor="0.6"), medianprops=dict(color="0.4"),
              whiskerprops=dict(color="0.6"), capprops=dict(color="0.6"))
    a.boxplot(vals, positions=pos - 0.12, widths=0.3, showfliers=False,
              medianprops=dict(color="#c0392b"))
    for i, v in zip(pos, vals):
        a.plot(np.full(len(v), i - 0.12) + np.linspace(-0.08, 0.08, len(v)), v, ".", ms=3, color="0.3")
    a.set_xticks(pos, names)
    a.set_ylabel(lab)
ax[0].axhline(1.0, color="0.6", lw=0.6, ls="--")
ax[1].legend([plt.Rectangle((0, 0), 1, 1, fc="w", ec="k"), plt.Rectangle((0, 0), 1, 1, fc="0.88", ec="0.6")],
             ["Flooding surfaces", "Reference horizons"], frameon=False, fontsize=6, loc="upper right")
fig.tight_layout()
save(fig, "fig07_e1_metrics", runs["D2"][0]["_use_standin"])
with open(TAB / "e1_metrics.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scenario", "median_strength", "median_continuity", "n_flooding_surfaces",
                "reference_median_strength", "reference_median_continuity", "n_reference_horizons",
                "strength_percentile_vs_reference", "continuity_percentile_vs_reference",
                "interior_rms_over_top_amplitude"])
    for name, s, c, rs, rc, rt in rows:
        ms, mc = np.nanmedian(s), np.nanmedian(c)
        w.writerow([name, f"{ms:.3f}", f"{mc:.3f}", len(s), f"{np.nanmedian(rs):.3f}",
                    f"{np.nanmedian(rc):.3f}", len(rs), f"{np.mean(rs < ms):.2f}", f"{np.mean(rc < mc):.2f}", f"{rt:.3f}"])
print(open(TAB / "e1_metrics.csv").read())
