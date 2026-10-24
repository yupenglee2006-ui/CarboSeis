"""Shared helpers for the figure and table scripts.

Data source: if the CarboKitten output named in the scenario exists, it is used.
Otherwise the Python stand-in is used (generated on first use) and every figure
is watermarked "STAND-IN DATA".  Force the stand-in with CARBOSEIS_STANDIN=1.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import subprocess
import sys
import tomllib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carboseis.config import load_scenario  # noqa: E402
from carboseis.grid import build_grid  # noqa: E402
from carboseis.io import read_section  # noqa: E402
from carboseis.pipeline import run_chain  # noqa: E402
from carboseis.standin import run_standin  # noqa: E402

SCENARIO = ROOT / "scenarios" / "luconia_base.toml"
FIG = ROOT / "figures"
TAB = ROOT / "tables"
CACHE = ROOT / "build" / "cache"
for d in (FIG, TAB, CACHE):
    d.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
ONE_COL, TWO_COL = 84 * MM, 174 * MM
TEXTURE_LABELS = ["Boundstone", "Grainstone", "Packstone", "Wackestone", "Mudstone", "Argillaceous"]
TEXTURE_COLORS = ["#1f6f8b", "#e6b33d", "#c97b2d", "#8c6d4f", "#7a7a7a", "#4f5d3a"]
FACIES_COLORS = ["#2a6db5", "#e8a33d", "#2e9e6b", "#d46a9f"]


def style():
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                         "axes.linewidth": 0.6, "figure.dpi": 150, "savefig.dpi": 300,
                         "font.family": "DejaVu Sans"})


def scenario(overrides=None):
    sc = load_scenario(SCENARIO, overrides)
    real = Path(sc["carbokitten"]["output"])
    use_standin = os.environ.get("CARBOSEIS_STANDIN") == "1" or not real.exists()
    sc["_use_standin"] = use_standin
    if use_standin and not Path(sc["standin"]["output"]).exists():
        run_standin(str(SCENARIO))
    return sc


def data_path(sc):
    return sc["standin"]["output"] if sc["_use_standin"] else sc["carbokitten"]["output"]


def grid(sc):
    return build_grid(read_section(data_path(sc), sc["grid"]["group"]), sc["grid"]["meteoric_zone_m"],
                      sc["grid"].get("meteoric_profile", "step"))


def ck_bridge():
    """Folder of the CarboSeis-CarboKitten bridge (ck_run.jl + Project.toml).

    Looked up in this order: environment variable CARBOSEIS_CK_BRIDGE, the folder `ck-bridge` of this
    repository (where it ships since v5.0), and a folder `carboseis-ck-bridge` next to this repository
    (the layout before v5.0)."""
    cands = [os.environ.get("CARBOSEIS_CK_BRIDGE"), ROOT / "ck-bridge",
             ROOT.parent / "carboseis-ck-bridge"]
    for c in cands:
        if c and (Path(c) / "ck_run.jl").is_file():
            return Path(c).resolve()
    raise SystemExit("CarboSeis-CarboKitten bridge not found: the folder 'ck-bridge' is missing from "
                     "this repository, or set CARBOSEIS_CK_BRIDGE (see README).")


# ---------------------------------------------------------------- Julia safeguards (v4.0.1)
# Julia does not guarantee identical random-number streams across minor versions, so CarboKitten
# runs with the same seed may differ between, e.g., Julia 1.12 and 1.13.  All forward runs of one
# study must therefore come from the Julia minor version for which the bridge was resolved.

START_MARKER = "carboseis-ck-bridge: starting CarboKitten run"      # printed by ck_run.jl >= 0.1.0.dev3
ENV_ERROR_MARKERS = ("CARBOSEIS-ENVIRONMENT-ERROR", "Julia launcher", "juliaup",
                     "Package CarboKitten not found", "ArgumentError: Package",
                     "is not recognized as an internal or external command", "No such file or directory")
JULIA = os.environ.get("JULIA", "julia")
VERSION_ATTR = "carboseis_julia_version"                             # written by ck_run.jl >= 0.1.0.dev3


def bridge_julia_version(bridge) -> str:
    """Julia version the bridge environment was resolved with (``julia_version`` in Manifest.toml)."""
    with open(Path(bridge) / "Manifest.toml", "rb") as f:
        return str(tomllib.load(f)["julia_version"])


def same_minor(a: str, b: str) -> bool:
    return str(a).split(".")[:2] == str(b).split(".")[:2]


def julia_env(bridge) -> dict:
    """Environment for Julia subprocesses.  With juliaup, the channel of the bridge's Julia version
    is selected unless JULIAUP_CHANNEL is already set, so a newer default Julia can stay installed."""
    env = dict(os.environ)
    env.setdefault("JULIAUP_CHANNEL", bridge_julia_version(bridge))
    return env


def is_env_error(text: str) -> bool:
    """True if a Julia log or failure reason shows that Julia or the bridge environment failed,
    rather than CarboKitten itself."""
    return any(m in text for m in ENV_ERROR_MARKERS)


def classify_failure(log_text: str) -> str:
    """'environment' if CarboKitten never started (or the log shows an environment error),
    'model' if the run started and then stopped."""
    if START_MARKER not in log_text or is_env_error(log_text):
        return "environment"
    return "model"


def probe_julia(bridge) -> str:
    """Start Julia once in the bridge environment and return its version.  Stops with a clear message
    if Julia cannot start or has another minor version than the bridge (override for exploratory
    work only: CARBOSEIS_ALLOW_ANY_JULIA=1)."""
    want = bridge_julia_version(bridge)
    cmd = [JULIA, f"--project={bridge}", "--startup-file=no", "-e", "print(VERSION)"]
    hint = (f"With juliaup, install it once with `juliaup add {want}`; CarboSeis then selects it through "
            f"JULIAUP_CHANNEL (currently {julia_env(bridge)['JULIAUP_CHANNEL']}), and your default Julia "
            "can stay as it is.")
    try:
        p = subprocess.run(cmd, env=julia_env(bridge), capture_output=True, text=True, timeout=900)
    except FileNotFoundError:
        raise SystemExit(f"Julia not found ('{JULIA}'). The bridge needs Julia {want}. {hint}")
    out = (p.stdout or "").strip().splitlines()
    if p.returncode != 0 or not out:
        raise SystemExit("Julia did not start in the bridge environment:\n  " + " ".join(cmd) + "\n"
                         + ((p.stderr or "") + (p.stdout or "")).strip()[-1500:] + "\n" + hint)
    got = out[-1].strip()
    if not same_minor(got, want):
        msg = (f"Julia {got} answered, but the bridge was resolved for Julia {want}. Runs from different "
               f"Julia minor versions must not be mixed. {hint}")
        if os.environ.get("CARBOSEIS_ALLOW_ANY_JULIA") != "1":
            raise SystemExit(msg)
        print("WARNING: " + msg, flush=True)
    return got


def run_provenance(path) -> dict:
    """Versions recorded by the bridge in a CarboKitten output file (empty values if not recorded)."""
    import h5py
    try:
        with h5py.File(path, "r") as f:
            a = f["input"].attrs
            get = lambda k: (a[k].decode() if isinstance(a[k], bytes) else str(a[k])) if k in a else None
            return {"julia": get(VERSION_ATTR), "carbokitten": get("carboseis_carbokitten_version"),
                    "bridge": get("carboseis_bridge_version")}
    except (OSError, KeyError):
        return {"julia": None, "carbokitten": None, "bridge": None}


def audit_versions(paths, julia_version):
    """Split cached runs into those without recorded Julia version and those from another minor version."""
    unversioned, other = [], []
    for p in paths:
        v = run_provenance(p)["julia"]
        if v is None:
            unversioned.append(Path(p).name)
        elif not same_minor(v, julia_version):
            other.append((Path(p).name, v))
    return unversioned, other


def clear_failure(failures_json, tag):
    """Remove a run from failures.json once it has completed (stale entries made counts wrong)."""
    path = Path(failures_json)
    if not path.exists():
        return
    fails = json.loads(path.read_text())
    if tag in fails:
        del fails[tag]
        path.write_text(json.dumps(fails, indent=1, default=str))


def _code_version():
    """Hash of the carboseis sources, so that cached chains are rebuilt after a code change."""
    h = hashlib.sha1()
    for f in sorted((ROOT / "src" / "carboseis").glob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def _key(sc, extra):
    src = data_path(sc)
    h = hashlib.sha1(repr((sorted((k, repr(v)) for k, v in sc.items() if not k.startswith("_")),
                           os.path.getmtime(src), os.path.getsize(src), _code_version(),
                           extra)).encode()).hexdigest()[:16]
    return CACHE / f"chain_{h}.pkl"


def chain(sc, grid_obj=None, **kw):
    """run_chain with an on-disk cache."""
    path = _key(sc, sorted(kw.items()))
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    res = run_chain(sc, grid=grid_obj or grid(sc), **kw)
    lean = copy.copy(res)
    lean.l2_depth = None            # the depth-domain PSF image is large and not used by the scripts
    with open(path, "wb") as f:
        pickle.dump(lean, f, protocol=pickle.HIGHEST_PROTOCOL)
    return res


def save(fig, name, standin):
    if standin:
        fig.text(0.5, 0.5, "STAND-IN DATA - rerun with CarboKitten", ha="center", va="center",
                 rotation=20, fontsize=16, color="red", alpha=0.18, transform=fig.transFigure)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/{name}.png")


def seismic_panel(ax, ts, t0, t1, clip=None, label=True):
    m = (ts.t >= t0) & (ts.t <= t1)
    d = ts.data[m]
    clip = clip or np.percentile(np.abs(d), 99)
    im = ax.imshow(d, aspect="auto", cmap="RdBu", vmin=-clip, vmax=clip, interpolation="bilinear",
                   extent=[ts.x[0] / 1e3, ts.x[-1] / 1e3, t1, t0])
    if label:
        ax.set_ylabel("Two-way time (s)")
    return im, clip


def platform_time_window(res, pad=0.06):
    from carboseis.pipeline import depth_to_twt
    em = res.em
    top = depth_to_twt(em, em.layer_top_depth.min(axis=0)[None, :])[0]
    base = depth_to_twt(em, em.layer_base_depth[0][None, :])[0]
    return float(top.min() - pad), float(base.max() + 0.03)


def depth_window(res, pad=60.0):
    em = res.em
    return float(em.layer_top_depth.min() - pad), float(em.layer_base_depth.max() + 40.0)
