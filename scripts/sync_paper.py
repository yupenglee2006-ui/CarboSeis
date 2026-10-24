"""Copy the generated figures and tables into the manuscript folder (make paper).

The manuscript is a separate folder beside this repository.  It holds its own copies of
`figures/*.pdf` and `tables/*.tex`, so that it builds without the code folder; this script
refreshes those copies after `make figures`, `make tables` or `make figures-e3`.

    python scripts/sync_paper.py                        # ../CarboSeis_manuscript_v5.0
    python scripts/sync_paper.py --paper ../other/folder

The target can also be set with the CARBOSEIS_PAPER environment variable.  Only PDF figures and
LaTeX table snippets are copied; the manuscript's own sources (manuscript.tex, references.bib,
copernicus.*) are never touched.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--paper", default=os.environ.get("CARBOSEIS_PAPER", str(ROOT.parent / "CarboSeis_manuscript_v5.0")))
a = ap.parse_args()

paper = Path(a.paper).resolve()
if not paper.is_dir():
    raise SystemExit(f"manuscript folder not found: {paper}\n"
                     f"pass --paper or set CARBOSEIS_PAPER")

n = 0
for sub, pattern in (("figures", "*.pdf"), ("tables", "*.tex")):
    src_dir, dst_dir = ROOT / sub, paper / sub
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(src_dir.glob(pattern)):
        shutil.copy2(src, dst_dir / src.name)
        n += 1

# report files the manuscript still holds that the code no longer produces
stale = []
for sub, pattern in (("figures", "*.pdf"), ("tables", "*.tex")):
    produced = {p.name for p in (ROOT / sub).glob(pattern)}
    for p in sorted((paper / sub).glob(pattern)):
        if p.name not in produced:
            stale.append(f"{sub}/{p.name}")

print(f"copied {n} file(s) to {paper}")
if stale:
    print("in the manuscript but not produced by the code (kept): " + ", ".join(stale))
