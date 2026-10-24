"""Download external inputs that are not redistributed in this repository.

Morley (2021) sea-level curve, as used in the CarboKitten validation case, pinned to commit
ec4cf5f of MindTheGap-ERC/CarboKitten-research-paper. Checked against its SHA-256 hash.
"""
import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "data/external/Morley_2021.txt": (
        "https://raw.githubusercontent.com/MindTheGap-ERC/CarboKitten-research-paper/"
        "ec4cf5f567598e84924e9ecad4ea0b089b17690f/data/Morley_2021.txt",
        "944ec8def0c35176b8e2721fe049ca3d13aac027c98ed575fd12ce6c4db62834"),
}

for rel, (url, sha) in FILES.items():
    dst = ROOT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        urllib.request.urlretrieve(url, dst)
    digest = hashlib.sha256(dst.read_bytes()).hexdigest()
    if digest != sha:
        raise SystemExit(f"checksum mismatch for {rel}")
    print(f"ok  {rel}")
