"use strict";

// Box frame shared with the model: 13.6 x 20.4 mm, guides at 7.5 mm and 12.7 mm from the top.
const BOX_W = 13.6, BOX_H = 20.4, XHEIGHT_Y = 7.5, BASELINE_Y = 12.7;
const PX_PER_MM = 10;
const PROMPT = ["a", "d", "g", "h", "k", "s", "t", "y", "B", "M"];
const CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:?!'\"()%-@$<>";

const $ = (id) => document.getElementById(id);
const state = { boxes: [], glyphs: null, fontBlob: null, fontUrl: null, fontSeq: 0, timings: {} };
window.__wewrite = state;

function drawGuides(ctx) {
  ctx.save();
  ctx.clearRect(0, 0, BOX_W * PX_PER_MM, BOX_H * PX_PER_MM);
  ctx.strokeStyle = "#dfe4f2";
  ctx.lineWidth = 1;
  for (const y of [XHEIGHT_Y, BASELINE_Y]) {
    ctx.beginPath();
    ctx.moveTo(0, y * PX_PER_MM);
    ctx.lineTo(BOX_W * PX_PER_MM, y * PX_PER_MM);
    ctx.stroke();
  }
  ctx.restore();
}

function redraw(box) {
  const ctx = box.ctx;
  drawGuides(ctx);
  ctx.strokeStyle = "#1d2433";
  ctx.fillStyle = "#1d2433";
  ctx.lineWidth = 0.6 * PX_PER_MM;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (const s of box.strokes) {
    ctx.beginPath();
    s.forEach(([x, y], i) => (i ? ctx.lineTo(x * PX_PER_MM, y * PX_PER_MM) : ctx.moveTo(x * PX_PER_MM, y * PX_PER_MM)));
    if (s.length === 1) {
      ctx.arc(s[0][0] * PX_PER_MM, s[0][1] * PX_PER_MM, 0.3 * PX_PER_MM, 0, 2 * Math.PI);
      ctx.fill();
    } else ctx.stroke();
  }
}

function makeBox(i, ch) {
  const wrap = document.createElement("div");
  wrap.className = "box";
  const label = document.createElement("input");
  label.value = ch;
  label.maxLength = 1;
  label.id = `label-${i}`;
  label.setAttribute("aria-label", `character for box ${i + 1}`);
  const canvas = document.createElement("canvas");
  canvas.id = `box-${i}`;
  canvas.width = BOX_W * PX_PER_MM;
  canvas.height = BOX_H * PX_PER_MM;
  const clear = document.createElement("button");
  clear.textContent = "clear";
  clear.className = "ghost";
  wrap.append(label, canvas, clear);
  const box = { label, canvas, ctx: canvas.getContext("2d"), strokes: [], current: null };
  const toMm = (e) => {
    const r = canvas.getBoundingClientRect();
    return [+((e.clientX - r.left) / r.width * BOX_W).toFixed(3), +((e.clientY - r.top) / r.height * BOX_H).toFixed(3)];
  };
  canvas.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    canvas.setPointerCapture(e.pointerId);
    box.current = [toMm(e)];
    box.strokes.push(box.current);
    redraw(box);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!box.current) return;
    const evts = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
    for (const ev of evts.length ? evts : [e]) box.current.push(toMm(ev));
    redraw(box);
  });
  const end = () => { box.current = null; };
  canvas.addEventListener("pointerup", end);
  canvas.addEventListener("pointercancel", end);
  clear.addEventListener("click", () => { box.strokes = []; redraw(box); });
  redraw(box);
  return [wrap, box];
}

function initBoxes() {
  const host = $("boxes");
  PROMPT.forEach((ch, i) => {
    const [el, box] = makeBox(i, ch);
    host.append(el);
    state.boxes.push(box);
  });
}

function samplesFromBoxes() {
  const out = [];
  for (const b of state.boxes) {
    const ch = b.label.value;
    if (b.strokes.length && ch && CHARSET.includes(ch)) out.push({ char: ch, strokes: b.strokes });
  }
  return out;
}

function setStatus(msg, isError = false) {
  const s = $("status");
  s.textContent = msg;
  s.className = isError ? "error" : "";
}

async function postJSON(url, body) {
  const t0 = performance.now();
  const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const ms = performance.now() - t0;
  if (!res.ok) {
    let msg = `${res.status}`;
    try { msg = (await res.json()).error || msg; } catch (_) { /* not JSON */ }
    throw new Error(msg);
  }
  return [res, ms];
}

function glyphSVG(strokes, widthMm) {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${BOX_W} ${BOX_H}`);
  for (const y of [XHEIGHT_Y, BASELINE_Y]) {
    const l = document.createElementNS(ns, "line");
    Object.entries({ x1: 0, x2: BOX_W, y1: y, y2: y, stroke: "#e6e9f2", "stroke-width": 0.12 }).forEach(([k, v]) => l.setAttribute(k, v));
    svg.append(l);
  }
  for (const s of strokes) {
    const p = document.createElementNS(ns, "polyline");
    const pts = s.length === 1 ? [s[0], [s[0][0] + 0.01, s[0][1]]] : s;
    p.setAttribute("points", pts.map(([x, y]) => `${x},${y}`).join(" "));
    Object.entries({ fill: "none", stroke: "#1d2433", "stroke-width": widthMm, "stroke-linecap": "round", "stroke-linejoin": "round" }).forEach(([k, v]) => p.setAttribute(k, v));
    svg.append(p);
  }
  return svg;
}

function renderGlyphs(mine) {
  const host = $("glyphs");
  host.innerHTML = "";
  const w = +$("width").value;
  for (const ch of CHARSET) {
    const g = state.glyphs[ch];
    if (!g) continue;
    const cell = document.createElement("div");
    cell.className = "glyph" + (mine.has(ch) ? " mine" : "");
    cell.dataset.char = ch;
    const tag = document.createElement("span");
    tag.textContent = ch;
    cell.append(glyphSVG(g, w), tag);
    host.append(cell);
  }
}

function showTimings() {
  const t = state.timings;
  const parts = [];
  if (t.generate != null) parts.push(`generate ${t.generate.toFixed(0)} ms (model ${t.model_ms} ms on the server)`);
  if (t.font != null) parts.push(`font build ${t.font.toFixed(0)} ms`);
  if (t.total != null) parts.push(`total ${t.total.toFixed(0)} ms`);
  $("timing").textContent = parts.join("  |  ");
}

async function buildFont() {
  if (!state.glyphs) return;
  const t0 = performance.now();
  const [res, ms] = await postJSON("/api/font", {
    glyphs: state.glyphs, family: $("family").value || "My Hand", width: +$("width").value,
  });
  const blob = await res.blob();
  state.timings.font = ms;
  const info = JSON.parse(res.headers.get("X-Font-Info") || "{}");
  state.fontInfo = info;
  const buf = await blob.arrayBuffer();
  const name = `WeWriteUser${++state.fontSeq}`;
  const face = new FontFace(name, buf);
  await face.load();
  document.fonts.add(face);
  $("preview").style.fontFamily = `"${name}", cursive`;
  $("preview").dataset.font = name;
  updatePreview();
  if (state.fontUrl) URL.revokeObjectURL(state.fontUrl);
  state.fontBlob = blob;
  state.fontUrl = URL.createObjectURL(blob);
  const dl = $("download");
  dl.href = state.fontUrl;
  dl.download = `${(info.family || "WeWrite").replace(/\s+/g, "")}.ttf`;
  dl.classList.remove("disabled");
  $("font-info").textContent = `${info.family}: ${info.chars ? info.chars.length : 0} glyphs, ${(info.bytes / 1024).toFixed(1)} KB TrueType`;
  return performance.now() - t0;
}

async function generate() {
  const samples = samplesFromBoxes();
  if (!samples.length) {
    setStatus("Write at least one character first (five or more gives a better match).", true);
    return;
  }
  $("generate").disabled = true;
  setStatus(`Generating from ${samples.length} sample${samples.length > 1 ? "s" : ""}...`);
  const t0 = performance.now();
  try {
    const [res, ms] = await postJSON("/api/generate", {
      samples, temperature: +$("temperature").value, seed: +$("seed").value || 0,
    });
    const data = await res.json();
    state.timings = { generate: ms, model_ms: data.model_ms };
    const glyphs = data.glyphs;
    const mine = new Set();
    if ($("use-mine").checked) for (const s of samples) { glyphs[s.char] = s.strokes; mine.add(s.char); }
    state.glyphs = glyphs;
    state.mine = mine;
    renderGlyphs(mine);
    setStatus("Building the font...");
    await buildFont();
    state.timings.total = performance.now() - t0;
    showTimings();
    setStatus("Done. Type below to try it, or download the .ttf and install it.");
  } catch (e) {
    setStatus(`Something went wrong: ${e.message}`, true);
  } finally {
    $("generate").disabled = false;
  }
}

function updatePreview() {
  const p = $("preview");
  if (!p.dataset.font) return;
  p.textContent = $("text").value || " ";
  p.style.fontSize = `${$("size").value}px`;
}

function initSpeech() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  const btn = $("mic");
  if (!SR) {
    btn.disabled = true;
    $("mic-note").textContent = "Dictation needs the Web Speech API, which this browser does not provide. Try Chrome, Edge or Safari.";
    return;
  }
  const rec = new SR();
  rec.lang = navigator.language || "en-US";
  rec.interimResults = true;
  rec.continuous = false;
  let base = "";
  rec.onresult = (e) => {
    let text = "";
    for (let i = 0; i < e.results.length; i++) text += e.results[i][0].transcript;
    $("text").value = (base ? base + " " : "") + text;
    updatePreview();
  };
  rec.onend = () => btn.classList.remove("listening");
  rec.onerror = (e) => { btn.classList.remove("listening"); $("mic-note").textContent = `Speech recognition error: ${e.error}`; };
  btn.addEventListener("click", () => {
    if (btn.classList.contains("listening")) { rec.stop(); return; }
    base = $("text").value.trim();
    btn.classList.add("listening");
    rec.start();
  });
}

async function initDemo() {
  try {
    const res = await fetch("/demo.json");
    const demo = await res.json();
    const sel = $("demo-writer");
    for (const w of Object.keys(demo)) {
      const o = document.createElement("option");
      o.value = w;
      o.textContent = w;
      sel.append(o);
    }
    sel.addEventListener("change", () => {
      const d = demo[sel.value];
      if (!d) return;
      state.boxes.forEach((b, i) => {
        const ch = PROMPT[i];
        b.label.value = ch;
        b.strokes = (d[ch] || []).map((s) => s.map((p) => p.slice()));
        redraw(b);
      });
    });
  } catch (_) { /* demo list is optional */ }
}

async function usePhoto() {
  const file = $("photo").files[0];
  const labels = $("photo-labels").value.replace(/\s+/g, "");
  if (!file || !labels) { setStatus("Choose a photo and type its characters.", true); return; }
  const fd = new FormData();
  fd.append("image", file);
  fd.append("labels", labels);
  setStatus("Reading the photo...");
  const res = await fetch("/api/vectorize", { method: "POST", body: fd });
  const data = await res.json();
  if (!res.ok) { setStatus(data.error || "Could not read the photo", true); return; }
  data.samples.forEach((s, i) => {
    if (i >= state.boxes.length) return;
    const b = state.boxes[i];
    b.label.value = s.char;
    b.strokes = s.strokes;
    redraw(b);
  });
  for (let i = data.samples.length; i < state.boxes.length; i++) { state.boxes[i].strokes = []; redraw(state.boxes[i]); }
  setStatus(`Traced ${data.samples.length} characters from the photo. Check them, then generate.`);
}

function init() {
  initBoxes();
  initSpeech();
  initDemo();
  $("generate").addEventListener("click", generate);
  $("clear-all").addEventListener("click", () => state.boxes.forEach((b) => { b.strokes = []; redraw(b); }));
  $("vectorize").addEventListener("click", () => usePhoto().catch((e) => setStatus(e.message, true)));
  $("text").addEventListener("input", updatePreview);
  $("size").addEventListener("input", updatePreview);
  $("temperature").addEventListener("input", () => { $("temperature-out").textContent = (+$("temperature").value).toFixed(2); });
  $("width").addEventListener("input", () => { $("width-out").textContent = `${(+$("width").value).toFixed(2)} mm`; });
  let pending = null;
  const rebuild = () => {
    if (!state.glyphs) return;
    clearTimeout(pending);
    pending = setTimeout(async () => {
      try { renderGlyphs(state.mine || new Set()); await buildFont(); showTimings(); } catch (e) { setStatus(e.message, true); }
    }, 300);
  };
  $("width").addEventListener("change", rebuild);
  $("family").addEventListener("change", rebuild);
}

init();
