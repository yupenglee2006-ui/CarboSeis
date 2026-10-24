"""Integration: stand-in output in CarboKitten's HDF5 layout runs through the whole chain."""
from pathlib import Path

import numpy as np

from carboseis.config import load_scenario
from carboseis.grid import build_grid
from carboseis.io import read_section
from carboseis.pipeline import run_chain
from carboseis.standin import run_standin

ROOT = Path(__file__).resolve().parents[1]
SCEN = ROOT / "scenarios" / "luconia_base.toml"


def test_standin_chain(tmp_path):
    if not (ROOT / "data" / "external" / "Morley_2021.txt").exists():
        import subprocess, sys
        subprocess.run([sys.executable, str(ROOT / "scripts" / "fetch_inputs.py")], check=True)
    ov = {"carbokitten.steps": 1250, "carbokitten.prerun.steps": 1000}
    out = run_standin(str(SCEN), str(tmp_path / "s.h5"), overrides=ov)
    sec = read_section(out, "profile")
    g = build_grid(sec)
    assert np.abs(g.z_top[-1] - sec.bathymetry[-1]).max() < 1e-6       # V1 on real file layout
    sc = load_scenario(SCEN, ov)
    res = run_chain(sc, grid=g, levels=(1,))
    assert np.isfinite(res.l1.data).all()
    th = res.layers["hc"] > 0.01
    assert 3000 < np.median(res.layers["vp"][th]) < 6500
