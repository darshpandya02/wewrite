"""Write public/demo.json: the prompt characters of a few held-out (test) writers."""

import json

import numpy as np

from training.data import ROOT, load
from training.sample_grid import PROMPT

WRITERS = ["UJI_W12", "UPV_W21", "UPV_W36", "UPV_W48"]


def main():
    samples = load()
    demo = {}
    for w in WRITERS:
        d = {x.char: [np.round(s, 2).tolist() for s in x.strokes] for x in samples
             if x.writer == w and x.rep == 1 and x.char in PROMPT}
        demo[f"Held-out writer {w}"] = d
    (ROOT / "public" / "demo.json").write_text(json.dumps(demo, separators=(",", ":")))
    print("wrote demo.json", list(demo))


if __name__ == "__main__":
    main()
