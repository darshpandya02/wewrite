"""Per-character mean box position (mm) from the training writers, used to place photo glyphs."""

import json

import numpy as np

from fontgen.vectorize import STATS_FILE
from training.data import load


def main():
    samples = [s for s in load() if s.split == "train"]
    stats = {}
    for c in sorted({s.char for s in samples}):
        b = [np.concatenate(s.strokes) for s in samples if s.char == c]
        stats[c] = {
            "top": float(np.mean([p[:, 1].min() for p in b])),
            "bottom": float(np.mean([p[:, 1].max() for p in b])),
            "cx": float(np.mean([(p[:, 0].min() + p[:, 0].max()) / 2 for p in b])),
        }
    STATS_FILE.write_text(json.dumps(stats, indent=1))
    print("wrote", STATS_FILE, len(stats))


if __name__ == "__main__":
    main()
