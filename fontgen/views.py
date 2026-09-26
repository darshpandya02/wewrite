"""JSON API. Stateless: the style vector and glyph strokes travel with each request."""

from __future__ import annotations

import json
import time
from functools import wraps

from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from fontgen import service
from fontgen.fontbuild import build_font, inspect_font
from fontgen.infer import load_model
from fontgen.strokes import CHARSET
from fontgen.vectorize import VectorizeError, vectorize


def api(methods=("POST",)):
    def deco(fn):
        @csrf_exempt
        @wraps(fn)
        def wrapper(request):
            if request.method == "OPTIONS":
                return HttpResponse(status=204)
            if request.method not in methods:
                return JsonResponse({"error": f"use {'/'.join(methods)}"}, status=405)
            t0 = time.perf_counter()
            try:
                resp = fn(request)
            except (service.BadRequest, VectorizeError) as e:
                return JsonResponse({"error": str(e)}, status=400)
            except json.JSONDecodeError:
                return JsonResponse({"error": "body must be JSON"}, status=400)
            resp["Server-Timing"] = f"app;dur={(time.perf_counter() - t0) * 1000:.1f}"
            resp["Cache-Control"] = "no-store"
            return resp

        return wrapper

    return deco


def _body(request) -> dict:
    data = json.loads(request.body or b"{}")
    if not isinstance(data, dict):
        raise service.BadRequest("body must be a JSON object")
    return data


def _num(data, key, default, lo, hi):
    try:
        v = float(data.get(key, default))
    except (TypeError, ValueError) as e:
        raise service.BadRequest(f"{key} must be a number") from e
    return min(max(v, lo), hi)


@api(methods=("GET",))
def health(request):
    m = load_model()
    return JsonResponse({"ok": True, "charset": CHARSET, "style_dim": m.cfg["style_dim"],
                         "dec_hidden": m.cfg["dec_hidden"], "n_mix": m.cfg["n_mix"]})


@api()
def style(request):
    data = _body(request)
    glyphs, chars = service.parse_samples(data.get("samples"))
    t0 = time.perf_counter()
    z = service.encode(glyphs, chars)
    return JsonResponse({"style": [round(float(v), 5) for v in z], "n_samples": len(chars),
                         "model_ms": round((time.perf_counter() - t0) * 1000, 1)})


@api()
def generate(request):
    data = _body(request)
    user = None
    if "samples" in data:
        glyphs, chars = service.parse_samples(data["samples"])
        z = service.encode(glyphs, chars)
        if data.get("calibrate", True):
            user = (glyphs, chars)
    else:
        z = service.parse_style(data.get("style"))
    chars = data.get("chars") or CHARSET
    if not isinstance(chars, str) or any(c not in CHARSET for c in chars):
        raise service.BadRequest("chars must be a string of supported characters")
    chars = "".join(dict.fromkeys(chars))
    temperature = _num(data, "temperature", service.DEFAULT_TEMPERATURE, 0.01, 1.5)
    seed = int(_num(data, "seed", 0, 0, 2**31 - 1))
    t0 = time.perf_counter()
    glyphs = service.generate(z, chars, temperature=temperature, seed=seed, user=user)
    return JsonResponse({"glyphs": service.glyphs_to_json(glyphs), "style": [round(float(v), 5) for v in z],
                         "model_ms": round((time.perf_counter() - t0) * 1000, 1)})


@api()
def font(request):
    data = _body(request)
    glyphs = service.parse_glyphs(data.get("glyphs"))
    width = _num(data, "width", 0.6, 0.15, 2.0)
    family = str(data.get("family") or "WeWrite Hand")
    blob = build_font(glyphs, family=family, width_mm=width)
    info = inspect_font(blob)
    missing = [c for c in glyphs if c not in info["chars"]]
    resp = HttpResponse(blob, content_type="font/ttf")
    fname = info["family"].replace(" ", "") + ".ttf"
    resp["Content-Disposition"] = f'attachment; filename="{fname}"'
    resp["X-Font-Info"] = json.dumps({**info, "missing": "".join(missing)})
    resp["Access-Control-Expose-Headers"] = "X-Font-Info, Server-Timing"
    return resp


@api()
def vectorize_view(request):
    upload = request.FILES.get("image")
    if upload is None:
        raise service.BadRequest("attach an image file as 'image'")
    if upload.size > 8 * 1024 * 1024:
        raise service.BadRequest("image too large (8 MB max)")
    labels = request.POST.get("labels", "")
    result = vectorize(upload.read(), labels)
    return JsonResponse(result)
