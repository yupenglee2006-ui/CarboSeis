"""Safeguards of v4.0.1: one Julia minor version for all CarboKitten runs, recorded versions, and
environment failures (e.g. the juliaup launcher) kept apart from model failures."""
import json
import os
import stat
import sys
from pathlib import Path

import h5py
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import common  # noqa: E402

LAUNCHER = ("run did not complete: Error: The Julia launcher failed to figure out which juliaup "
            "channel to use. (log: carbokitten_da63fc4c54da.log)")


def test_failure_classification():
    assert common.is_env_error(LAUNCHER)
    assert common.classify_failure("Error: The Julia launcher failed to figure out ...") == "environment"
    assert common.classify_failure("ERROR: LoadError: ArgumentError: Package CarboKitten not found") == "environment"
    started = "carboseis-ck-bridge 0.1.0.dev3: Julia 1.12.7\n" + common.START_MARKER + "\n"
    assert common.classify_failure(started + "ERROR: LoadError: DimensionMismatch: ...") == "model"
    assert common.classify_failure(started + "CARBOSEIS-ENVIRONMENT-ERROR: ...") == "environment"


def test_version_helpers(tmp_path, monkeypatch):
    (tmp_path / "Manifest.toml").write_text('julia_version = "1.12.7"\nmanifest_format = "2.0"\n')
    assert common.bridge_julia_version(tmp_path) == "1.12.7"
    assert common.same_minor("1.12.7", "1.12.2") and not common.same_minor("1.13.1", "1.12.7")
    monkeypatch.delenv("JULIAUP_CHANNEL", raising=False)
    assert common.julia_env(tmp_path)["JULIAUP_CHANNEL"] == "1.12.7"
    monkeypatch.setenv("JULIAUP_CHANNEL", "1.12.5")
    assert common.julia_env(tmp_path)["JULIAUP_CHANNEL"] == "1.12.5"      # an explicit choice is kept


@pytest.mark.skipif(os.name == "nt", reason="uses a shell script as a stand-in for julia")
def test_probe_julia(tmp_path, monkeypatch):
    (tmp_path / "Manifest.toml").write_text('julia_version = "1.12.7"\n')
    fake = tmp_path / "julia"

    def answer(version, code=0):
        fake.write_text(f"#!/bin/sh\nprintf '{version}'\nexit {code}\n")
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC)

    monkeypatch.setattr(common, "JULIA", str(fake))
    monkeypatch.delenv("CARBOSEIS_ALLOW_ANY_JULIA", raising=False)
    answer("1.12.7")
    assert common.probe_julia(tmp_path) == "1.12.7"
    answer("1.13.1")
    with pytest.raises(SystemExit):
        common.probe_julia(tmp_path)                      # other minor version: refused
    monkeypatch.setenv("CARBOSEIS_ALLOW_ANY_JULIA", "1")
    assert common.probe_julia(tmp_path) == "1.13.1"       # explicit override for exploratory work
    monkeypatch.delenv("CARBOSEIS_ALLOW_ANY_JULIA")
    answer("Error: The Julia launcher failed to figure out which juliaup channel to use.", code=1)
    with pytest.raises(SystemExit):
        common.probe_julia(tmp_path)                      # launcher failure stops before any run


def _run_file(path, version=None):
    with h5py.File(path, "w") as f:
        g = f.create_group("input")
        if version:
            g.attrs["carboseis_julia_version"] = version
            g.attrs["carboseis_carbokitten_version"] = "0.6.1"
    return path


def test_audit_and_provenance(tmp_path):
    a = _run_file(tmp_path / "a.h5", "1.12.7")
    b = _run_file(tmp_path / "b.h5", "1.13.1")
    c = _run_file(tmp_path / "c.h5")
    assert common.run_provenance(a) == {"julia": "1.12.7", "carbokitten": "0.6.1", "bridge": None}
    unversioned, other = common.audit_versions([a, b, c], "1.12.7")
    assert unversioned == ["c.h5"] and other == [("b.h5", "1.13.1")]


def test_clear_failure(tmp_path):
    fj = tmp_path / "failures.json"
    fj.write_text(json.dumps({"aaa": {"reason": LAUNCHER}, "bbb": {"reason": "QC: deposit"}}))
    common.clear_failure(fj, "aaa")
    assert list(json.loads(fj.read_text())) == ["bbb"]
    common.clear_failure(tmp_path / "missing.json", "x")     # no file: nothing to do
