"""Example E3 (manuscript Sect. 6.5): Morris screening of 12 parameters plus
cellular-automaton seed replicates.

Usage: python scripts/e3_run.py [--r 10] [--seeds 10] [--engine standin|carbokitten]
                                [--retry-failed] [--qc-max-layer 25] [--accept-unversioned]

Sedimentological parameters need new forward runs (CarboKitten through the bridge's ck_run.jl,
or the stand-in); property and acquisition parameters reuse stored runs.  Forward runs
are cached in build/e3 by a hash of their CarboKitten parameters, so an interrupted
experiment resumes where it stopped and only the CarboSeis chain is re-evaluated.

Forward runs that do not complete, or that fail the deposit QC are recorded
in build/e3/failures.json and enter the results as NaN; the Morris estimator skips the
elementary effects that touch them.  A failed run is not retried unless --retry-failed
is given.  Julia's console output goes to build/e3/logs/<run>.log.

Safeguards (v4.0.1).  Before any run, Julia is started once in the bridge environment; E3 stops
if it cannot start or if its minor version differs from the bridge's (Julia does not guarantee the
same random streams across minor versions).  With juliaup the right channel is selected through
JULIAUP_CHANNEL.  Cached runs must carry the Julia version recorded by the bridge (>= 0.1.0.dev3)
and come from the same minor version; runs without it are refused unless --accept-unversioned.
A run that fails before CarboKitten starts (launcher, missing package, wrong Julia) stops E3 instead
of being recorded as a model failure; failure flags left by such errors in older versions are
retried automatically, and failures.json drops runs that later completed.

Seed replicates are evaluated at the centre of the parameter box (u = 0.5 for every
parameter), which is where the Morris trajectories are centred.
Results are written incrementally to build/e3/results.npz; progress lines, with an estimate of
the remaining time, are also appended to build/e3/status.txt.  Cached runs are named by a hash of
their CarboKitten parameters without file paths, so they survive moving or copying the folder;
runs cached by v3.1-v3.4 under path-dependent names are renamed automatically.
"""
import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

from common import (ROOT, SCENARIO, audit_versions, ck_bridge, classify_failure, clear_failure,
                    is_env_error, julia_env, JULIA, probe_julia, run_provenance, same_minor, scenario)
from carboseis.analysis import STAT_NAMES, morris_trajectories, summary_statistics
from carboseis.config import set_dotted
from carboseis.grid import build_grid
from carboseis.io import read_section
from carboseis.pipeline import run_chain
from carboseis.standin import run_standin

# (name, dotted key or special, low, high, needs forward run)
PARAMS = [
    ("Sea-level amplitude scale", "carbokitten.sea_level_scale", 0.5, 1.5, True),
    ("Sea-level time stretch", "carbokitten.sea_level_time_stretch", 0.8, 1.2, True),
    ("Subsidence rate (m/Myr)", "carbokitten.subsidence_rate_m_per_Myr", 120.0, 180.0, True),
    ("Euphotic max. growth (m/Myr)", "carbokitten.facies.0.maximum_growth_rate_m_per_Myr", 1200.0, 2400.0, True),
    ("Transport coefficient (m/yr)", "@transport", 1.0, 4.0, True),
    ("Meteoric tau_crit (Myr)", "diagenesis.tau_crit_Myr", 0.005, 0.08, False),
    ("Moldic porosity gain", "diagenesis.dphi_mold", 0.0, 0.12, False),
    ("Burial cementation c_ch (1/km)", "diagenesis.c_ch_per_km", 0.1, 0.5, False),
    ("Compaction coefficient scale", "@compaction", 0.5, 1.5, False),
    ("Interparticle aspect ratio", "rockphysics.aspect_ratios.0", 0.08, 0.25, False),
    ("Gas column (m)", "rockphysics.gas_column_m", 0.0, 200.0, False),
    ("Dominant frequency (Hz)", "seismic.dominant_frequency_Hz", 20.0, 45.0, False),
]
OUT = ROOT / "build" / "e3"
LOGS = OUT / "logs"
FAILURES = OUT / "failures.json"
for _d in (OUT, LOGS):
    _d.mkdir(parents=True, exist_ok=True)
QC_MAX_LAYER_M = 25.0      # = sediment_buffer_size x depositional_resolution of the scenario


class ForwardRunFailed(RuntimeError):
    pass


def apply(sc, u):
    sc = copy.deepcopy(sc)
    for (name, key, lo, hi, _), ui in zip(PARAMS, u):
        v = lo + ui * (hi - lo)
        if key == "@transport":
            for f in sc["carbokitten"]["facies"]:
                f["transport_coefficient_m_per_yr"] = v
        elif key == "@compaction":
            sc["compaction"]["c_per_km"] = [c * v for c in sc["compaction"]["c_per_km"]]
        else:
            set_dotted(sc, key, v)
    return sc


def _record_failure(tag, ck, reason):
    fails = json.loads(FAILURES.read_text()) if FAILURES.exists() else {}
    fails[tag] = {"reason": reason,
                  "parameters": {k: ck[k] for k in ("sea_level_scale", "sea_level_time_stretch",
                                                    "subsidence_rate_m_per_Myr", "ca_random_seed")},
                  "facies": [{k: v for k, v in f.items()
                              if k in ("name", "maximum_growth_rate_m_per_Myr",
                                       "transport_coefficient_m_per_yr")} for f in ck["facies"]]}
    FAILURES.write_text(json.dumps(fails, indent=1, default=str))


def qc_forward(path, group, max_layer_m):
    """Reject runs with single-interval deposits far larger than carbonate production can supply,
    which indicate that the explicit transport scheme left its stable range (bridge NOTES.md B4)."""
    sec = read_section(path, group)
    net = sec.deposition.sum(axis=1)[1:]
    worst = float(np.nanmax(net)) if net.size else 0.0
    if not np.isfinite(sec.deposition).all():
        return "non-finite deposition"
    if worst > max_layer_m:
        return f"deposit of {worst:.1f} m in one output interval (limit {max_layer_m} m)"
    return None


def run_ck(sc, seed_offset=0):
    ck = copy.deepcopy(sc["carbokitten"])
    ck["ca_random_seed"] = ck["ca_random_seed"] + seed_offset
    return ck


def run_tag(ck):
    """Hash of the CarboKitten parameters that change the result.  File locations are left out,
    so cached runs stay valid when the repository folder is moved, copied or renamed."""
    c = copy.deepcopy(ck)
    c.pop("output", None)
    if isinstance(c.get("prerun"), dict):
        c["prerun"].pop("output", None)
    if "sea_level_file" in c:
        c["sea_level_file"] = Path(str(c["sea_level_file"])).name
    return hashlib.sha1(json.dumps(c, sort_keys=True, default=str).encode()).hexdigest()[:12]


def legacy_tag(ck):
    """Tag used up to v3.4, which included absolute paths (valid only in the same folder)."""
    return hashlib.sha1(json.dumps(ck, sort_keys=True, default=str).encode()).hexdigest()[:12]


def adopt_legacy(ck, engine):
    """Rename a run cached under its v3.1-v3.4 name to the path-independent name."""
    new, old = OUT / f"{engine}_{run_tag(ck)}.h5", OUT / f"{engine}_{legacy_tag(ck)}.h5"
    moved = False
    for suf in (".h5", ".failed"):
        a, b = old.with_suffix(suf), new.with_suffix(suf)
        if a.exists() and not b.exists():
            os.replace(a, b)
            moved = True
    return moved


def complete_run(path, group="profile"):
    """True if a cached HDF5 file is a finished run.  A run interrupted by Ctrl+C leaves a file
    that either does not open or whose last output frame was never written (all zeros)."""
    try:
        with h5py.File(path, "r") as f:
            last = f[group]["deposition"][-1]
            return bool(np.isfinite(last).all() and np.abs(last).sum() > 0)
    except Exception:
        return False


PROGRESS = {"todo": 0, "done": 0, "times": []}
JULIA_VERSION = {"value": None}             # set by main() after probing Julia in the bridge environment


class EnvironmentStop(SystemExit):
    """Julia or the bridge environment failed: stop E3 without recording a model failure."""


def fmt_time(sec):
    if sec is None or not np.isfinite(sec):
        return "unknown"
    h, m = divmod(int(round(sec / 60.0)), 60)
    return f"{h} h {m:02d} min" if h else f"{m} min"


def eta():
    t = PROGRESS["times"]
    left = max(PROGRESS["todo"] - PROGRESS["done"], 0)
    return fmt_time(np.mean(t) * left) if t else ("none" if left == 0 else "unknown until the first run ends")


def say(msg):
    """Print a progress line and append it, time-stamped, to build/e3/status.txt."""
    print(msg, flush=True)
    with open(OUT / "status.txt", "a", encoding="utf-8") as fh:
        fh.write(time.strftime("%Y-%m-%d %H:%M:%S  ") + msg + "\n")


def forward(sc, engine, seed_offset=0, retry_failed=False, qc_max_layer=QC_MAX_LAYER_M):
    ck = run_ck(sc, seed_offset)
    adopt_legacy(ck, engine)
    tag = run_tag(ck)
    path = OUT / f"{engine}_{tag}.h5"
    flag = path.with_suffix(".failed")
    if flag.exists() and is_env_error(flag.read_text(errors="replace")):
        say(f"  {flag.name}: earlier attempt never started CarboKitten (environment error); running it again")
        flag.unlink()
        clear_failure(FAILURES, tag)
    if flag.exists() and not retry_failed:
        raise ForwardRunFailed(flag.read_text())
    if path.exists() and not complete_run(path, sc["grid"]["group"]):
        say(f"  {path.name} is incomplete (interrupted run); running it again")
        path.unlink()
    if not path.exists():
        t_start = time.time()
        if engine == "standin":
            run_standin(str(SCENARIO), str(path), overrides={
                **{f"carbokitten.{k}": v for k, v in ck.items() if k not in ("facies", "prerun")},
                "carbokitten.facies": ck["facies"], "standin.seed": sc["standin"]["seed"] + seed_offset})
        else:
            tmp = path.with_name(path.stem + ".partial.h5")      # renamed only when the run succeeds
            tmp.unlink(missing_ok=True)
            sets = ["--set", f"output={tmp}", "--set", f"ca_random_seed={ck['ca_random_seed']}"]
            for key in ("sea_level_scale", "sea_level_time_stretch", "subsidence_rate_m_per_Myr"):
                sets += ["--set", f"{key}={ck[key]}"]
            for i, f in enumerate(ck["facies"], 1):
                sets += ["--set", f"facies.{i}.transport_coefficient_m_per_yr={f['transport_coefficient_m_per_yr']}"]
                if "maximum_growth_rate_m_per_Myr" in f:
                    sets += ["--set", f"facies.{i}.maximum_growth_rate_m_per_Myr={f['maximum_growth_rate_m_per_Myr']}"]
            log = LOGS / f"{engine}_{tag}.log"
            with open(log, "w") as fh:
                bridge = ck_bridge()
                ret = subprocess.run([JULIA, "--project=" + str(bridge), str(bridge / "ck_run.jl"),
                                      str(SCENARIO)] + sets, stdout=fh, stderr=subprocess.STDOUT,
                                     env=julia_env(bridge))
            if ret.returncode != 0:
                text = log.read_text(errors="replace")
                tail = text.strip().splitlines()
                err = next((ln for ln in tail if ln.startswith("ERROR") or "ENVIRONMENT-ERROR" in ln),
                           tail[-1] if tail else "unknown error")
                tmp.unlink(missing_ok=True)                   # never cache a partial file
                if classify_failure(text) == "environment":
                    say(f"  E3 STOPPED: Julia or the bridge environment failed before CarboKitten started "
                        f"({err[:160]}; log: {log.name}). Nothing was recorded as a model failure. Fix the "
                        "environment and run the same command again; finished runs are kept.")
                    raise EnvironmentStop(2)
                flag.write_text(f"run did not complete: {err} (log: {log.name})")
                _record_failure(tag, ck, flag.read_text())
                PROGRESS["done"] += 1
                say(f"  forward run {PROGRESS['done']}/{PROGRESS['todo']} FAILED after "
                    f"{time.time() - t_start:.0f} s ({err[:80]}); about {eta()} left")
                raise ForwardRunFailed(flag.read_text())
            prov = run_provenance(tmp)
            if prov["julia"] is None or not same_minor(prov["julia"], JULIA_VERSION["value"] or prov["julia"]):
                say(f"  E3 STOPPED: {tmp.name} does not record the expected Julia version (found {prov['julia']}, "
                    f"expected {JULIA_VERSION['value']}). Is the bridge older than 0.1.0.dev3? (log: {log.name})")
                raise EnvironmentStop(2)
            os.replace(tmp, path)
        PROGRESS["done"] += 1
        PROGRESS["times"].append(time.time() - t_start)
        say(f"  forward run {PROGRESS['done']}/{PROGRESS['todo']} finished in "
            f"{PROGRESS['times'][-1]:.0f} s; about {eta()} left")
    reason = qc_forward(str(path), sc["grid"]["group"], qc_max_layer)
    if reason:
        flag.write_text(f"QC: {reason}")
        _record_failure(tag, ck, flag.read_text())
        raise ForwardRunFailed(flag.read_text())
    flag.unlink(missing_ok=True)
    clear_failure(FAILURES, tag)
    return str(path)


def evaluate(sc, engine, seed_offset=0, **kw):
    """Seven seismic statistics of one parameter set, or NaN if its forward run failed."""
    try:
        path = forward(sc, engine, seed_offset, **kw)
    except ForwardRunFailed as e:
        print(f"  forward run failed: {e}", flush=True)
        return np.full(len(STAT_NAMES), np.nan)
    g = build_grid(read_section(path, sc["grid"]["group"]), sc["grid"]["meteoric_zone_m"],
                   sc["grid"].get("meteoric_profile", "step"))
    res = run_chain(sc, grid=g, levels=(1,))
    return summary_statistics(res.l1, res.em)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--engine", default=None)
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--qc-max-layer", type=float, default=QC_MAX_LAYER_M)
    ap.add_argument("--accept-unversioned", action="store_true",
                    help="keep cached runs that do not record their Julia version (made before v4.0.1)")
    a = ap.parse_args()
    kw = dict(retry_failed=a.retry_failed, qc_max_layer=a.qc_max_layer)
    base = scenario()
    engine = a.engine or ("standin" if base["_use_standin"] else "carbokitten")
    if engine == "carbokitten":
        bridge = ck_bridge()
        JULIA_VERSION["value"] = probe_julia(bridge)
        say(f"Julia {JULIA_VERSION['value']} in the bridge environment {bridge} "
            f"(JULIAUP_CHANNEL={julia_env(bridge)['JULIAUP_CHANNEL']})")
    k = len(PARAMS)
    trajs, delta = morris_trajectories(k, a.r, seed=42)
    u_centre = np.array([0.5] * k)

    # plan: every distinct forward run, adopting runs cached under their old names
    plan = [(u_centre, s + 1) for s in range(a.seeds)] + [(x, 0) for X in trajs for x in X]
    runs, moved = {}, 0
    for u, off in plan:
        ck = run_ck(apply(base, u), off)
        moved += adopt_legacy(ck, engine)
        runs[run_tag(ck)] = ck
    cached = [OUT / f"{engine}_{t}.h5" for t in runs if complete_run(OUT / f"{engine}_{t}.h5", base["grid"]["group"])]
    have = len(cached)
    if engine == "carbokitten" and cached:
        unversioned, other = audit_versions(cached, JULIA_VERSION["value"])
        if other:
            raise SystemExit(f"E3 STOPPED: {len(other)} cached runs come from another Julia minor version "
                             f"(e.g. {other[0][0]}: Julia {other[0][1]}; now {JULIA_VERSION['value']}). Delete "
                             "them (or the whole build/e3 folder) and run again.")
        if unversioned and not a.accept_unversioned:
            raise SystemExit(f"E3 STOPPED: {len(unversioned)} cached runs do not record their Julia version "
                             "(made before v4.0.1), so they cannot be checked against the current Julia. "
                             "Delete build/e3 to start afresh, or pass --accept-unversioned if you know they "
                             f"were made with Julia {JULIA_VERSION['value'].rsplit('.', 1)[0]}.")
    flags = [OUT / f"{engine}_{t}.failed" for t in runs]
    bad = sum(f.exists() and not is_env_error(f.read_text(errors="replace")) for f in flags) \
        if not a.retry_failed else 0
    PROGRESS["todo"] = len(runs) - have - bad
    say(f"E3 plan: {len(plan)} chain evaluations ({a.seeds} seed replicates + {a.r} trajectories x {k + 1} "
        f"points) needing {len(runs)} distinct {engine} runs: {have} already cached"
        + (f" ({moved} renamed from older names)" if moved else "") + f", {bad} failed earlier, "
        f"{PROGRESS['todo']} to run. Figures can be made only after the final 'E3 finished' message.")

    meta = dict(engine=engine, delta=delta, julia_version=str(JULIA_VERSION["value"] or ""), params=np.array([p[0] for p in PARAMS]),
                stats=np.array(STAT_NAMES), lo=np.array([p[2] for p in PARAMS]),
                hi=np.array([p[3] for p in PARAMS]), forward=np.array([p[4] for p in PARAMS]),
                r_planned=a.r, seeds_planned=a.seeds)
    Y, seeds = [], []
    t0 = time.time()
    for s in range(a.seeds):
        seeds.append(evaluate(apply(base, u_centre), engine, seed_offset=s + 1, **kw))
        say(f"seed {s + 1}/{a.seeds}  {time.time() - t0:.0f} s")
        np.savez(OUT / "results.npz", seeds=np.array(seeds), **meta)
    say(f"Seed replicates finished ({a.seeds}/{a.seeds}) after {fmt_time(time.time() - t0)}. "
        f"Starting the Morris screening: {a.r} trajectories x {k + 1} points; "
        f"{PROGRESS['todo'] - PROGRESS['done']} forward runs still to do, about {eta()}.")
    for i, X in enumerate(trajs):
        Y.append(np.array([evaluate(apply(base, x), engine, **kw) for x in X]))
        n_fail = int(np.isnan(Y[-1]).all(axis=1).sum())
        np.savez(OUT / "results.npz", seeds=np.array(seeds), trajs=np.array(trajs[:len(Y)]),
                 Y=np.array(Y), **meta)
        say(f"trajectory {i + 1}/{a.r} finished  {fmt_time(time.time() - t0)} elapsed, failed points {n_fail}, "
            f"forward runs left {PROGRESS['todo'] - PROGRESS['done']}, about {eta()} left")
    n_seed_fail = int(np.isnan(np.array(seeds)).all(axis=1).sum())
    n_traj_fail = int(sum(np.isnan(y).all(axis=1).sum() for y in Y))
    say(f"E3 finished after {fmt_time(time.time() - t0)}: {n_traj_fail} of {sum(len(y) for y in Y)} "
        f"trajectory evaluations and {n_seed_fail} of {len(seeds)} seed replicates failed "
        f"(see {FAILURES.name}). Next: make figures-e3")


if __name__ == "__main__":
    main()
