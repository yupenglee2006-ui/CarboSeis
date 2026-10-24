"""Reader tests for the CarboKitten v0.6.1 HDF5 layout (COMPATIBILITY.md #2-#4, #6)."""
from pathlib import Path

import h5py
import numpy as np
import pytest

from carboseis.grid import build_grid
from carboseis.io import read_section
from carboseis.standin import run_standin

ROOT = Path(__file__).resolve().parents[1]
SCEN = ROOT / "scenarios" / "luconia_base.toml"
FAST = {"carbokitten.steps": 1250, "carbokitten.prerun.steps": 1000}


@pytest.fixture(scope="module")
def standin_file(tmp_path_factory):
    if not (ROOT / "data" / "external" / "Morley_2021.txt").exists():
        import subprocess, sys
        subprocess.run([sys.executable, str(ROOT / "scripts" / "fetch_inputs.py")], check=True)
    return run_standin(str(SCEN), str(tmp_path_factory.mktemp("io") / "s.h5"), overrides=FAST)


def test_v061_layout_has_no_bathymetry(standin_file):
    with h5py.File(standin_file) as f:
        assert "bathymetry" not in f["profile"]
        assert "sediment_thickness" in f["profile"]
        assert isinstance(f["profile"].attrs["slice"], bytes)


def test_elevation_uses_the_profile_row(standin_file):
    """Regression for COMPATIBILITY.md #6: the initial topography of the slice row,
    not row 0 of the full grid, must enter the recomputed elevation."""
    with h5py.File(standin_file) as f:
        eta_all = f["input/initial_topography"][()]
        j = int(f["profile"].attrs["slice"].decode().split(",")[1]) - 1
    assert np.ptp(eta_all[:, eta_all.shape[1] // 2]) > 1.0     # topography varies along y
    sec = read_section(standin_file, "profile")
    assert np.allclose(sec.eta0, eta_all[j])
    assert np.allclose(sec.bathymetry[0], eta_all[j])            # frame 0 = initial surface
    g = build_grid(sec)
    assert np.abs(g.z_top[-1] - sec.bathymetry[-1]).max() < 1e-6  # V1


def test_legacy_bathymetry_dataset_gives_same_section(standin_file, tmp_path):
    """A file that also carries an explicit bathymetry dataset must agree exactly."""
    sec = read_section(standin_file, "profile")
    legacy = tmp_path / "legacy.h5"
    with h5py.File(standin_file) as src, h5py.File(legacy, "w") as dst:
        src.copy("input", dst)
        src.copy("profile", dst)
        dst["profile"]["bathymetry"] = sec.bathymetry[:, None, :]
        del dst["profile"]["sediment_thickness"]
        dst["profile"].attrs["slice"] = src["profile"].attrs["slice"].decode()   # str form
    sec2 = read_section(str(legacy), "profile")
    assert np.allclose(sec2.bathymetry, sec.bathymetry)
    assert np.allclose(sec2.deposition, sec.deposition)


def test_inconsistent_bathymetry_is_rejected(standin_file, tmp_path):
    bad = tmp_path / "bad.h5"
    sec = read_section(standin_file, "profile")
    with h5py.File(standin_file) as src, h5py.File(bad, "w") as dst:
        src.copy("input", dst)
        src.copy("profile", dst)
        dst["profile"]["bathymetry"] = sec.bathymetry[:, None, :] - 50.0
    with pytest.raises(ValueError):
        read_section(str(bad), "profile")
