import io
import json

from django.test import Client
from fontTools.ttLib import TTFont

from tests.test_infer import DEMO

client = Client()


def samples():
    d = next(iter(DEMO.values()))
    return [{"char": c, "strokes": s} for c, s in d.items()]


def post(url, body):
    return client.post(url, data=json.dumps(body), content_type="application/json")


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["ok"]


def test_style_then_generate_then_font():
    r = post("/api/style", {"samples": samples()})
    assert r.status_code == 200
    style = r.json()["style"]
    r = post("/api/generate", {"style": style, "chars": "Hello", "seed": 1})
    assert r.status_code == 200
    glyphs = r.json()["glyphs"]
    assert set(glyphs) == set("Helo")
    r = post("/api/font", {"glyphs": glyphs, "family": "Api Test"})
    assert r.status_code == 200 and r["Content-Type"] == "font/ttf"
    font = TTFont(io.BytesIO(r.content))
    assert set("Helo") <= {chr(u) for u in font.getBestCmap()}
    assert json.loads(r["X-Font-Info"])["missing"] == ""


def test_generate_full_set_from_samples():
    r = post("/api/generate", {"samples": samples(), "temperature": 0.15})
    assert r.status_code == 200
    assert len(r.json()["glyphs"]) == 78


def test_bad_requests():
    assert post("/api/style", {"samples": []}).status_code == 400
    assert post("/api/style", {"samples": [{"char": "é", "strokes": [[[1, 1]]]}]}).status_code == 400
    assert post("/api/generate", {"style": [0.1, 0.2]}).status_code == 400
    assert client.post("/api/style", data="nope", content_type="application/json").status_code == 400
    assert client.get("/api/generate").status_code == 405
    big = {"samples": [{"char": "a", "strokes": [[[1e6, 1]]]}]}
    assert post("/api/style", big).status_code == 400
