"""Write a compact copy of a CarboKitten output file that holds only what CarboSeis reads.

    python scripts/pack_output.py data/carbokitten/luconia_base.h5 [-o OUT.h5] [--keep-production]

Keeps the ``input`` group and the section group named in the scenario (``profile``) with
``deposition``, ``disintegration`` and ``sediment_thickness``; drops the 3-D volume group and
``production``.  Datasets are re-compressed (shuffle + gzip 9).  Values are copied bit for bit,
and the script checks that CarboSeis reads the same section from both files.
"""
import argparse
from pathlib import Path

import h5py
import numpy as np

from common import scenario
from carboseis.io import read_section


def copy_attrs(src, dst):
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def copy_group(src, dst, keep=None):
    copy_attrs(src, dst)
    for k, obj in src.items():
        if keep is not None and k not in keep:
            continue
        if isinstance(obj, h5py.Group):
            copy_group(obj, dst.create_group(k))
        else:
            data = obj[()]
            kw = dict(compression="gzip", compression_opts=9, shuffle=True) if np.ndim(data) > 0 and np.size(data) > 64 else {}
            d = dst.create_dataset(k, data=data, **kw)
            copy_attrs(obj, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("-o", "--out")
    ap.add_argument("--keep-production", action="store_true")
    a = ap.parse_args()
    group = scenario()["grid"]["group"]
    src = Path(a.file)
    out = Path(a.out) if a.out else src.with_name(src.stem + f"_{group}.h5")
    keep = {"deposition", "disintegration", "sediment_thickness", "bathymetry"}
    if a.keep_production:
        keep.add("production")
    with h5py.File(src, "r") as fs, h5py.File(out, "w") as fd:
        copy_attrs(fs, fd)
        copy_group(fs["input"], fd.create_group("input"))
        copy_group(fs[group], fd.create_group(group), keep=keep)
    s0, s1 = read_section(str(src), group), read_section(str(out), group)
    for name in ("deposition", "disintegration", "bathymetry", "sea_level", "t", "x", "eta0"):
        assert np.array_equal(getattr(s0, name), getattr(s1, name)), name
    print(f"wrote {out}  ({src.stat().st_size / 2**20:.1f} MB -> {out.stat().st_size / 2**20:.1f} MB); "
          f"CarboSeis reads an identical section")


if __name__ == "__main__":
    main()
