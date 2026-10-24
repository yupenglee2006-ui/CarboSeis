"""Fig. 1: the CarboSeis workflow, with the user inputs read from the scenario file (blue) and
the quantities saved at each step (brown); see manuscript Sect. 2 and Table 1."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from common import save

# colours of the v3 figure, plus one for user input
STEP_FILL, STEP_EDGE = "#dcefe9", "#0f6e56"
CK_FILL, CK_EDGE = "#f4e1d2", "#a0522d"
USER_FILL, USER_EDGE, USER_TEXT = "#e3ecf7", "#2b5797", "#2b5797"
OUT_FILL, OUT_EDGE = "#efefef", "#555555"
SAVED_TEXT, ARROW = "#6b4226", "#444444"

BW, BH, GAP, X0 = 1.24, 0.64, 0.20, 0.10
W = 2 * X0 + 5 * BW + 4 * GAP + 0.14   # figure width (in), incl. the turn on the right
H = 2.78                                  # figure height (in); axes in inch units
Y1, Y2 = 1.82, 0.62              # bottom of the upper and lower row of boxes
FS_TITLE, FS_BODY, FS_TAG = 7.0, 6.5, 6.0


def col_x(i):
    return X0 + i * (BW + GAP)


def box(ax, i, y, title, body, fill, edge, tag_in=None, tag_saved=None):
    x = col_x(i)
    ax.add_patch(FancyBboxPatch((x, y), BW, BH, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=fill, ec=edge, lw=1.0))
    ax.text(x + BW / 2, y + BH - 0.11, title, ha="center", va="center",
            fontsize=FS_TITLE, fontweight="bold")
    ax.text(x + BW / 2, y + BH / 2 - 0.08, body, ha="center", va="center",
            fontsize=FS_BODY, linespacing=1.15)
    if tag_in:
        ax.text(x + BW / 2, y + BH + 0.04, tag_in, ha="center", va="bottom",
                fontsize=FS_TAG, style="italic", color=USER_TEXT, linespacing=1.1)
    if tag_saved:
        ax.text(x + BW / 2, y - 0.05, tag_saved, ha="center", va="top",
                fontsize=FS_TAG, style="italic", color=SAVED_TEXT)


def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7,
                                 color=ARROW, lw=0.9, shrinkA=0, shrinkB=0))


fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

# upper row, left to right
box(ax, 0, Y1, "Scenario file", "forward-model and\nformation parameters",
    USER_FILL, USER_EDGE, tag_in="supplied by the user")
box(ax, 1, Y1, "CarboKitten.jl", "deposition, sea level,\nsediment height",
    CK_FILL, CK_EDGE, tag_in="sea level, subsidence,\nproducers, transport",
    tag_saved="HDF5 (new or existing run)")
box(ax, 2, Y1, "1  Stratigraphic grid", "time-lines, water\ndepth, exposure time",
    STEP_FILL, STEP_EDGE, tag_in=r"meteoric zone $d_\mathrm{m}$",
    tag_saved=r"saved: $z_k$, $w_k$, $\tau_\mathrm{e}$")
box(ax, 3, Y1, "2  Texture", "producer fractions\n" + r"$\rightarrow$ Dunham textures",
    STEP_FILL, STEP_EDGE, tag_in=r"$M_{f,\tau}(w)$, wave bases",
    tag_saved=r"saved: $q_{k,\tau}$")
box(ax, 4, Y1, "3  Compaction", "solid-conserving\nAthy trends",
    STEP_FILL, STEP_EDGE, tag_in=r"$\phi_{0,\tau}$, $c_\tau$, $H_\mathrm{ob}$",
    tag_saved=r"saved: $h$, $\phi$")

# lower row, right to left
box(ax, 4, Y2, "4  Diagenesis", "meteoric, burial,\noptional rules",
    STEP_FILL, STEP_EDGE,
    tag_in=r"$\tau_\mathrm{crit}$, $\Delta\phi$, $z_\mathrm{ch}$, $c_\mathrm{ch}$," + "\npore-type partition",
    tag_saved="saved: $\\phi$ by pore type")
box(ax, 3, Y2, "5  Rock physics", "Xu–Payne DEM +\nGassmann",
    STEP_FILL, STEP_EDGE, tag_in="mineral and fluid moduli,\npore aspect ratios",
    tag_saved=r"saved: $V_\mathrm{P}$, $V_\mathrm{S}$, $\rho$")
box(ax, 2, Y2, "6  Earth model", "water, overburden,\nfill, basement",
    STEP_FILL, STEP_EDGE, tag_in="water depth, velocity\ntrends, basement",
    tag_saved=r"saved: $Z$, $R$")
box(ax, 1, Y2, "7  Seismic", "1-D convolution,\nPSF imaging (TWT)",
    STEP_FILL, STEP_EDGE, tag_in=r"wavelet, $\Delta t$, $\Delta x$," + "\n" + r"$\theta_\mathrm{max}$, S/N",
    tag_saved="saved: seismic")
box(ax, 0, Y2, "Outputs", "seismic sections and\nsaved cubes (HDF5)",
    OUT_FILL, OUT_EDGE)

# arrows: upper row left to right
yc1, yc2 = Y1 + BH / 2, Y2 + BH / 2
for i in range(4):
    arrow(ax, (col_x(i) + BW, yc1), (col_x(i + 1), yc1))
# turn from step 3 down to step 4 along the right margin
xr = col_x(4) + BW
ax.plot([xr, xr + 0.12, xr + 0.12], [yc1, yc1, yc2], color=ARROW, lw=0.9,
        solid_joinstyle="round")
arrow(ax, (xr + 0.12, yc2), (xr, yc2))
# lower row right to left
for i in range(4, 0, -1):
    arrow(ax, (col_x(i), yc2), (col_x(i - 1) + BW, yc2))

# legend
ax.text(W / 2 - 0.08, 0.12, "blue: parameters read from the scenario file", fontsize=FS_TAG,
        style="italic", color=USER_TEXT, va="center", ha="right")
ax.text(W / 2 + 0.08, 0.12, "brown: quantities saved for inspection", fontsize=FS_TAG,
        style="italic", color=SAVED_TEXT, va="center", ha="left")

save(fig, "fig01_workflow", False)
