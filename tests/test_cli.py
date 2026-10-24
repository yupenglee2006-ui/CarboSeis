"""The user workflow of manuscript Sect. 2: one command builds the seismic model from a scenario
file and a CarboKitten run, writes every intermediate quantity to HDF5, and petrophysical or
seismic parameters can be changed with --set without a new forward run."""
from pathlib import Path

import h5py
import numpy as np
import pytest

from carboseis import __version__
from carboseis.cli import main, parse_value
from carboseis.config import load_scenario
from carboseis.grid import build_grid
from carboseis.io import read_section
from carboseis.pipeline import run_chain
from carboseis.standin import run_standin

ROOT = Path(__file__).resolve().parents[1]
SCEN = ROOT / "scenarios" / "luconia_base.toml"
FAST = {"carbokitten.steps": 1250, "carbokitten.prerun.steps": 1000}


@pytest.fixture(scope="module")
def ck_file(tmp_path_factory):
    if not (ROOT / "data" / "external" / "Morley_2021.txt").exists():
        import subprocess, sys
        subprocess.run([sys.executable, str(ROOT / "scripts" / "fetch_inputs.py")], check=True)
    return run_standin(str(SCEN), str(tmp_path_factory.mktemp("cli") / "ck.h5"), overrides=FAST)


def test_parse_value():
    assert parse_value("45") == 45 and parse_value("0.15") == 0.15
    assert parse_value("false") is False and parse_value("[0.1, 0.8, 0.02]") == [0.1, 0.8, 0.02]
    assert parse_value("rhg") == "rhg" and parse_value('"ormsby"') == "ormsby"


def test_run_writes_all_saved_quantities(ck_file, tmp_path):
    out = tmp_path / "res.h5"
    main(["run", str(SCEN), "--input", ck_file, "--levels", "1", "--out", str(out)])
    with h5py.File(out) as f:
        assert f.attrs["carboseis_version"] == __version__
        assert "file_text" in f["scenario"].attrs and "effective_json" in f["scenario"].attrs
        for name in ("timeline_elevation", "water_depth", "exposure_time", "texture_fractions",
                     "porosity_interparticle", "porosity_stiff", "porosity_crack", "vp", "vs",
                     "rho", "impedance"):
            assert name in f["layers"], name
        for name in ("vp", "vs", "rho", "impedance", "reflectivity", "age"):
            assert name in f["earth_model"], name
        d = f["seismic/level1/data"][()]
        assert np.isfinite(d).all() and np.abs(d).max() > 0
        assert "level2" not in f["seismic"]
        nl, nx = f["layers/vp"].shape
        assert f["layers/texture_fractions"].shape == (nl, 6, nx)


def test_set_changes_properties_without_new_forward_run(ck_file, tmp_path):
    a, b = tmp_path / "a.h5", tmp_path / "b.h5"
    main(["run", str(SCEN), "--input", ck_file, "--levels", "1", "--out", str(a)])
    main(["run", str(SCEN), "--input", ck_file, "--levels", "1", "--out", str(b),
          "--set", "rockphysics.aspect_ratios.0=0.08", "--set", "seismic.dominant_frequency_Hz=45"])
    with h5py.File(a) as fa, h5py.File(b) as fb:
        assert np.array_equal(fa["layers/porosity"][()], fb["layers/porosity"][()])   # same history
        assert not np.allclose(fa["layers/vp"][()], fb["layers/vp"][()])            # softer pores
        assert fb["seismic/level1"].attrs["dominant_frequency_Hz"] == 45.0


def test_noise_off_by_default_and_reproducible(ck_file):
    sc = load_scenario(SCEN, FAST)
    g = build_grid(read_section(ck_file, "profile"))
    clean = run_chain(sc, grid=g, levels=(1,))
    sc_n = load_scenario(SCEN, {**FAST, "seismic.noise_snr": 4.0, "seismic.noise_seed": 7})
    n1 = run_chain(sc_n, grid=g, levels=(1,))
    n2 = run_chain(sc_n, grid=g, levels=(1,))
    assert np.array_equal(n1.l1.data, n2.l1.data)
    resid = n1.l1.data - clean.l1.data
    snr = np.sqrt(np.mean(clean.l1.data ** 2)) / np.sqrt(np.mean(resid ** 2))
    assert abs(snr - 4.0) < 0.05


def test_standin_attribute_of_older_files_is_read(ck_file, tmp_path):
    old = tmp_path / "old.h5"
    with h5py.File(ck_file) as src, h5py.File(old, "w") as dst:
        src.copy("input", dst)
        src.copy("profile", dst)
        del dst["input"].attrs["carboseis_standin"]
        dst["input"].attrs["seiskitten_standin"] = True        # name used before v4.0
    assert read_section(str(old), "profile").standin
