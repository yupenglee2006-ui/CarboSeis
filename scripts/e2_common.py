"""E2 runs: dominant frequency x imaging level, event tracking and age span by zone."""
import numpy as np
from common import chain, grid, scenario
from carboseis import analysis as A

FREQS = [15.0, 30.0, 60.0]
ZONES = {1: "interior", 2: "margin", 3: "slope", 4: "basin"}
BASIN_TOL_M = 20.0


def run_all():
    sc = scenario()
    g = grid(sc)
    layer_dt = float(np.median(np.diff(g.t)))
    out = {}
    for f0 in FREQS:
        res = chain(sc, grid_obj=g, f0=f0)
        cls = A.zones(res.em, basin_tol_m=BASIN_TOL_M)
        for lev, ts in (("Level 1", res.l1), ("Level 2", res.l2)):
            ages = A.age_section(ts, res.em)
            win = A.platform_window(ts, res.em, cls > 0, start_s=0.004)
            pk, tr = A.pick_peaks(ts, win, rel_threshold=0.3)
            ev = A.track_events(pk, max_jump=3, min_traces=8) + A.track_events(tr, max_jump=3, min_traces=8)
            out[(f0, lev)] = A.event_age_stats(ev, ages, layer_dt, cls)
    return sc, out, layer_dt
