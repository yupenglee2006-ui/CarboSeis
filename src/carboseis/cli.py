"""Command line of CarboSeis.

    carboseis run SCENARIO.toml [--input CARBOKITTEN.h5 | --standin] [--set KEY=VALUE ...]
                                [--levels 1 2] [--frequency HZ] [--constant-impedance]
                                [--out RESULT.h5]
    carboseis standin SCENARIO.toml

``run`` builds the seismic model of the formation described by the scenario file from a
CarboKitten run and writes every intermediate quantity and the synthetic seismic sections to
one HDF5 file (layout in :mod:`carboseis.output`).  The CarboKitten output is the file named
in ``[carbokitten] output`` unless ``--input`` points to another run.  ``--set`` changes single
parameters without editing the scenario file, for example
``--set rockphysics.gas_column_m=100`` or ``--set seismic.dominant_frequency_Hz=45``.
Petrophysical, earth-model and seismic parameters take effect at once; parameters of the
``[carbokitten]`` table describe the forward model and need a new CarboKitten run with the
CarboSeis-CarboKitten bridge.

``standin`` writes the development stand-in dataset (a simplified Python emulation of
CarboKitten, for testing only).
"""
from __future__ import annotations

import argparse
import sys
import time
import tomllib
from pathlib import Path

from .config import load_scenario
from .output import write_result
from .pipeline import run_chain
from .standin import run_standin


def parse_value(text: str):
    """Read a command-line value as a TOML value (number, boolean, array or quoted string);
    anything else is kept as a plain string."""
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text


def parse_sets(items) -> dict:
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = parse_value(v.strip())
    return out


def cmd_run(a) -> Path:
    overrides = parse_sets(a.set)
    ck_keys = [k for k in overrides if k.startswith("carbokitten.")]
    if ck_keys:
        print("note: " + ", ".join(ck_keys) + " describe the forward model; they take effect only "
              "after a new CarboKitten run with the CarboSeis-CarboKitten bridge.", file=sys.stderr)
    scen = Path(a.scenario)
    sc = load_scenario(scen, overrides)
    if a.input:
        sc["carbokitten"]["output"] = str(Path(a.input).resolve())
    sc["_use_standin"] = bool(a.standin)
    if a.standin and not Path(sc["standin"]["output"]).exists():
        run_standin(str(scen))
    src = Path(sc["standin"]["output"] if a.standin else sc["carbokitten"]["output"])
    if not src.exists():
        raise SystemExit(f"CarboKitten output not found: {src}\n"
                         "Run CarboKitten with the CarboSeis-CarboKitten bridge (make carbokitten), "
                         "pass an existing run with --input, or use --standin for a test.")
    mode = "constant" if a.constant_impedance else "process"
    t0 = time.time()
    res = run_chain(sc, f0=a.frequency, levels=tuple(a.levels), property_mode=mode)
    out = Path(a.out) if a.out else Path.cwd() / f"{scen.stem}_carboseis.h5"
    write_result(out, res, sc, scenario_text=scen.read_text(encoding="utf-8"), f0=a.frequency,
                 property_mode=mode)
    nl, nx = res.layers["hc"].shape
    print(f"CarboSeis: {nl} layers x {nx} columns from {src.name}; earth model "
          f"{res.em.vp.shape[0]} x {res.em.vp.shape[1]} cells; levels {sorted(a.levels)}; "
          f"{time.time() - t0:.0f} s")
    if res.grid.standin:
        print("warning: STAND-IN data, not CarboKitten output - for testing only", file=sys.stderr)
    print(f"wrote {out}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="carboseis", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("standin", help="generate the development stand-in dataset")
    s.add_argument("scenario")
    r = sub.add_parser("run", help="build the seismic model and write it to HDF5")
    r.add_argument("scenario", help="scenario file (TOML)")
    src = r.add_mutually_exclusive_group()
    src.add_argument("--input", help="CarboKitten HDF5 output to use instead of [carbokitten] output")
    src.add_argument("--standin", action="store_true", help="use the development stand-in data")
    r.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="override one scenario parameter (dotted key), may be repeated")
    r.add_argument("--levels", type=int, nargs="+", choices=(1, 2), default=[1, 2],
                   help="imaging levels: 1 = 1-D convolution, 2 = point-spread function")
    r.add_argument("--frequency", type=float, default=None,
                   help="dominant frequency (Hz), overrides [seismic] dominant_frequency_Hz")
    r.add_argument("--constant-impedance", action="store_true",
                   help="one impedance per facies ([constant_impedance]) instead of the full chain")
    r.add_argument("--out", help="output HDF5 file (default: <scenario>_carboseis.h5 here)")
    a = ap.parse_args(argv)
    if a.cmd == "standin":
        print(run_standin(a.scenario))
        return
    cmd_run(a)


if __name__ == "__main__":
    main()
