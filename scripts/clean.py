"""Remove generated files on any operating system (Makefile targets clean, clean-cache, clean-runs).

    python scripts/clean.py outputs   # build/, figures/*.png|pdf, tables/*.tex|csv   (make clean)
    python scripts/clean.py cache     # build/cache                                    (make clean-cache)
    python scripts/clean.py runs      # data/carbokitten (base case, pre-run) and build/ (E3 runs, caches)
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("what", choices=["outputs", "cache", "runs"])
a = ap.parse_args()
if a.what == "cache":
    targets = [ROOT / "build" / "cache"]
elif a.what == "runs":
    targets = [ROOT / "data" / "carbokitten", ROOT / "build"]
else:
    targets = [ROOT / "build"]
    for folder, patterns in (("figures", ("*.png", "*.pdf")), ("tables", ("*.tex", "*.csv"))):
        for pat in patterns:
            targets += sorted((ROOT / folder).glob(pat))
n = 0
for t in targets:
    if t.is_dir():
        shutil.rmtree(t)
        n += 1
    elif t.exists():
        t.unlink()
        n += 1
print(f"removed {n} item(s): " + ", ".join(str(t.relative_to(ROOT)) for t in targets if not t.exists())[:300])
