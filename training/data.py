"""UJI Pen Characters v2 loader.

Dataset: UJI Pen Characters (Version 2), UCI Machine Learning Repository,
https://archive.ics.uci.edu/dataset/177 (DOI 10.24432/C5FG8S), CC BY 4.0.
Creators: F. Prat, M. Castro, D. Llorens, A. Marzal, J. Vilar.
"""

from __future__ import annotations

import io
import pickle
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fontgen.strokes import CHAR_INDEX, clip_glyph, resample_glyph

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_FILE = DATA_DIR / "ujipenchars2.txt"
CACHE_FILE = DATA_DIR / "uji2_processed.pkl"
URL = "https://archive.ics.uci.edu/static/public/177/uji+pen+characters+version+2.zip"

# Writers from the official "trn" pool held out for early stopping / tuning.
VAL_WRITERS = ["UJI_W05", "UJI_W10", "UPV_W23", "UPV_W34", "UPV_W47", "UPV_W56"]


@dataclass
class Sample:
    char: str
    writer: str  # e.g. "UPV_W23"
    rep: int  # session 1 or 2
    pool: str  # "trn" or "tst" (official UJI split)
    strokes: list  # list of (n, 2) float arrays, mm, resampled

    @property
    def split(self) -> str:
        if self.pool == "tst":
            return "test"
        return "val" if self.writer in VAL_WRITERS else "train"


def download(force: bool = False) -> Path:
    if RAW_FILE.exists() and not force:
        return RAW_FILE
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL) as r:
        blob = r.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for name in z.namelist():
            (DATA_DIR / Path(name).name).write_bytes(z.read(name))
    return RAW_FILE


def parse_raw(path: Path = RAW_FILE):
    """Yield (char, session_id, strokes_in_mm) from the raw UNIPEN-like file."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("WORD"):
            i += 1
            continue
        parts = line.split()
        char, sid = parts[1], parts[2]
        n = int(lines[i + 1].split()[1])
        scale = 152.0 if "_UPV_" in sid else 100.0  # ink units per mm
        strokes = []
        for k in range(n):
            toks = lines[i + 2 + k].split("#", 1)[1].split()
            xy = np.asarray(toks, dtype=np.float64).reshape(-1, 2) / scale
            strokes.append(xy)
        yield char, sid, strokes
        i += 2 + n


def load(rebuild: bool = False) -> list[Sample]:
    if CACHE_FILE.exists() and not rebuild:
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    download()
    samples = []
    for char, sid, strokes in parse_raw():
        if char not in CHAR_INDEX:
            continue  # Spanish letters and non-ASCII symbols are not used
        pool, site, w, rep = sid.split("_")[0], sid.split("_")[1], *sid.split("_")[2].split("-")
        glyph = clip_glyph(resample_glyph(strokes))
        if not glyph:
            continue
        samples.append(Sample(char, f"{site}_{w}", int(rep), pool, glyph))
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(samples, f)
    return samples


if __name__ == "__main__":
    s = load(rebuild=True)
    from collections import Counter

    print(len(s), "samples")
    print(Counter(x.split for x in s))
    print("writers per split", {k: len({x.writer for x in s if x.split == k}) for k in ("train", "val", "test")})
    lens = np.array([sum(len(t) for t in x.strokes) for x in s])
    print("points: mean %.1f p50 %d p95 %d max %d" % (lens.mean(), np.median(lens), np.percentile(lens, 95), lens.max()))
    allpts = np.concatenate([np.concatenate(x.strokes) for x in s])
    print("x range", np.percentile(allpts[:, 0], [0.5, 99.5]), "y range", np.percentile(allpts[:, 1], [0.5, 99.5]))
