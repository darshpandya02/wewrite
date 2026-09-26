"""End-to-end check of a deployed (or local) WeWrite site in a real browser.

Draws the ten prompt characters on the canvases with mouse events (strokes
taken from a held-out writer of the dataset, so the drawing is real
handwriting), generates, waits for the font preview, downloads the .ttf and
verifies it with fontTools. Repeats the generate/build cycle to measure
latency as seen from the browser.

    uv run --with playwright --python 3.12 python scripts/verify_site.py https://example.vercel.app
"""

from __future__ import annotations

import io
import json
import statistics
import sys
import time
from pathlib import Path

from fontTools.ttLib import TTFont
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
EXPECTED = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
BOX_W, BOX_H = 13.6, 20.4


def draw_char(page, idx, strokes):
    box = page.locator(f"#box-{idx}").bounding_box()
    sx, sy = box["width"] / BOX_W, box["height"] / BOX_H
    for s in strokes:
        x0, y0 = s[0]
        page.mouse.move(box["x"] + x0 * sx, box["y"] + y0 * sy)
        page.mouse.down()
        for x, y in s[1:]:
            page.mouse.move(box["x"] + x * sx, box["y"] + y * sy, steps=2)
        page.mouse.up()


def make_photo(glyphs, labels, path, scale=14):
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (int(len(labels) * 16 * scale + 60), int(22 * scale)), (236, 231, 219))
    d = ImageDraw.Draw(im)
    for i, ch in enumerate(labels):
        x0 = 30 + i * 16 * scale
        for s in glyphs[ch]:
            pts = [(x0 + x * scale, y * scale) for x, y in s]
            if len(pts) == 1:
                x, y = pts[0]
                d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(25, 30, 60))
            else:
                d.line(pts, fill=(25, 30, 60), width=int(0.6 * scale), joint="curve")
    im.save(path, quality=88)


def main(url: str, runs: int = 5, out: str | None = None):
    demo = json.loads((ROOT / "public" / "demo.json").read_text())
    writer = list(demo)[1]
    report = {"url": url, "writer": writer}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(accept_downloads=True, viewport={"width": 1280, "height": 1600})
        t0 = time.perf_counter()
        resp = page.goto(url, wait_until="networkidle")
        report["page_status"] = resp.status
        report["page_load_ms"] = (time.perf_counter() - t0) * 1000
        labels = [page.input_value(f"#label-{i}") for i in range(10)]
        for i, ch in enumerate(labels):
            draw_char(page, i, demo[writer][ch])
        drawn = page.evaluate("window.__wewrite.boxes.map(b => b.strokes.length)")
        report["strokes_drawn_per_box"] = drawn
        assert all(n > 0 for n in drawn), drawn

        timings = []
        for r in range(runs):
            page.fill("#seed", str(r))
            page.click("#generate")
            page.wait_for_function("document.querySelector('#status').textContent.startsWith('Done')", timeout=60000)
            timings.append(page.evaluate("({...window.__wewrite.timings})"))
        report["timings_ms"] = timings
        report["latency_summary_ms"] = {
            k: {"first": timings[0][k], "median_warm": statistics.median(t[k] for t in timings[1:]) if runs > 1 else None}
            for k in ("generate", "font", "total")
        }
        report["glyph_cells"] = page.locator("#glyphs .glyph").count()
        page.fill("#text", "Hello from WeWrite 2026")
        family = page.evaluate("getComputedStyle(document.querySelector('#preview')).fontFamily")
        report["preview_font_family"] = family
        report["preview_font_loaded"] = page.evaluate(
            "document.fonts.check('40px ' + document.querySelector('#preview').dataset.font)")
        # Photo path: render the same writer's strokes as ink on paper, upload it, check the boxes fill.
        photo = ROOT / "results" / "verify_photo.jpg"
        make_photo(demo[writer], labels, photo)
        page.click("#photo-panel summary")
        page.set_input_files("#photo", str(photo))
        page.fill("#photo-labels", "".join(labels))
        t1 = time.perf_counter()
        page.click("#vectorize")
        page.wait_for_function("document.querySelector('#status').textContent.startsWith('Traced')", timeout=60000)
        report["photo_vectorize_ms"] = (time.perf_counter() - t1) * 1000
        report["photo_status"] = page.inner_text("#status")
        report["photo_strokes_per_box"] = page.evaluate("window.__wewrite.boxes.map(b => b.strokes.length)")
        page.click("#generate")
        page.wait_for_function("document.querySelector('#status').textContent.startsWith('Done')", timeout=60000)
        report["photo_generate_ok"] = True
        page.fill("#text", "Hello from WeWrite 2026")
        shot = ROOT / "results" / "live_screenshot.png"
        page.screenshot(path=str(shot), full_page=True)
        with page.expect_download() as dl:
            page.click("#download")
        path = dl.value.path()
        data = Path(path).read_bytes()
        report["download_name"] = dl.value.suggested_filename
        browser.close()

    font = TTFont(io.BytesIO(data))
    cmap = font.getBestCmap()
    have = {chr(u) for u in cmap}
    glyf = font["glyf"]
    empty = [c for c in EXPECTED if c in have and glyf[cmap[ord(c)]].numberOfContours == 0]
    report["font"] = {
        "bytes": len(data), "num_glyphs": len(font.getGlyphOrder()), "family": font["name"].getDebugName(1),
        "missing_alnum": "".join(c for c in EXPECTED if c not in have), "empty_alnum": "".join(empty),
        "has_punctuation": "".join(c for c in ".,;:?!'\"()%-@$<>" if c in have),
    }
    report["ok"] = (report["page_status"] == 200 and not report["font"]["missing_alnum"]
                    and not report["font"]["empty_alnum"] and report["preview_font_loaded"])
    text = json.dumps(report, indent=1)
    print(text)
    if out:
        Path(out).write_text(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    args = sys.argv[1:]
    sys.exit(main(args[0], out=args[1] if len(args) > 1 else None))
