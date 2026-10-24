"""E1 property models (manuscript Sect. 6.3, Table 10)."""
from common import chain, grid, scenario

SCENARIOS = {
    "DC": ({}, "constant"),
    "D0": ({"diagenesis.meteoric": False, "diagenesis.burial_cementation": False}, "process"),
    "D1": ({"diagenesis.meteoric": True, "diagenesis.burial_cementation": False}, "process"),
    "D2": ({"diagenesis.meteoric": True, "diagenesis.burial_cementation": True}, "process"),
}
LABELS = {"DC": "DC: constant impedance per facies", "D0": "D0: texture + compaction",
          "D1": "D1: D0 + meteoric exposure", "D2": "D2: D1 + burial cementation (full)"}


def run_all():
    out = {}
    g = None
    for name, (ov, mode) in SCENARIOS.items():
        sc = scenario(ov)
        g = g or grid(sc)
        out[name] = (sc, chain(sc, grid_obj=g, property_mode=mode))
    return out
