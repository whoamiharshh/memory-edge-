"use strict";
// Phone as a sensor. Text is set with textContent only (never innerHTML).
// Two sources:
//  - MICROPHONE (recommended): the browser delivers raw audio at 44.1/48 kHz, fast enough to hear bearing-defect
//    impacts and shaft orders. Echo cancellation, noise suppression and auto-gain are switched OFF (they would
//    reshape the very signal we measure). 1 s of 16-bit PCM is sent per request to /api/ingest/audio.
//  - MOTION sensor: browsers cap it at ~60 Hz (W3C Generic Sensor / DeviceMotion), so it only sees shaft-rate
//    problems such as imbalance. Windows of 256 samples, sampling rate measured from event timestamps.
const $ = (id) => document.getElementById(id);
const WIN = 256, HOP = 128, KEY = "mm_operator_token";
$("tok").value = sessionStorage.getItem(KEY) || "";
let buf = [], times = [], sinceSend = 0, sending = false, started = false;
let audio = null;

function rpm() {
  const v = parseFloat($("rpm").value);
  return v > 0 ? v : null;
}

async function api(path, body) {
  const token = $("tok").value.trim();
  sessionStorage.setItem(KEY, token);
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json", "X-Operator-Token": token },
    body: JSON.stringify(body) });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof data.detail === "string" ? data.detail : "HTTP " + r.status);
  return data;
}

function show(r) {
  const last = r.last || {};
  $("state").textContent = last.new_part ? "normal (new part)" : (last.state || "–");
  $("cap").textContent = r.capture ? `capturing healthy: ${r.capture.have}/${r.capture.want}` : (last.state === "baseline" ? "baseline fitted" : "");
  const d = r.diagnosis;
  let t = "";
  if (d && d.bearing) t += `bearing: ${d.bearing.fault_class} — ${d.bearing.why}. `;
  if (d && d.rotating) t += `shaft ≈ ${d.shaft_hz} Hz · ${d.rotating.fault_class} — ${d.rotating.why}`;
  $("diag").textContent = t;
  $("err").textContent = "";
}

// ---- motion sensor ------------------------------------------------------------------------------------------
function onMotion(e) {
  const a = e.accelerationIncludingGravity || e.acceleration;
  if (!a || a.x === null) return;
  buf.push([a.x, a.y, a.z]);
  times.push(performance.now());
  if (buf.length > WIN) { buf.shift(); times.shift(); }
  sinceSend += 1;
  if (buf.length === WIN && sinceSend >= HOP && !sending) sendMotion();
}

async function sendMotion() {
  sinceSend = 0;
  const span = (times[times.length - 1] - times[0]) / 1000;
  const fs = (WIN - 1) / span;
  $("rate").textContent = `motion sensor ${fs.toFixed(1)} Hz · window ${span.toFixed(1)} s`;
  if (!(fs > 5 && fs < 1000)) return;
  sending = true;
  try { show(await api("/api/ingest/signal", { axes: buf.slice(), fs, rpm: rpm(), source: "phone-motion" })); }
  catch (err) { $("err").textContent = "⚠ " + err.message; }
  finally { sending = false; }
}

async function startMotion() {
  if (typeof DeviceMotionEvent !== "undefined" && typeof DeviceMotionEvent.requestPermission === "function") {
    const p = await DeviceMotionEvent.requestPermission();         // iOS: must be called from a user gesture
    if (p !== "granted") throw new Error("motion permission denied");
  }
  if (!started) { window.addEventListener("devicemotion", onMotion); started = true; }
  $("rate").textContent = "waiting for motion data…";
}

// ---- microphone ---------------------------------------------------------------------------------------------
function toBase64Pcm16(f32) {
  const bytes = new Uint8Array(f32.length * 2);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < f32.length; i++) {
    const v = Math.max(-1, Math.min(1, f32[i]));
    view.setInt16(i * 2, v < 0 ? v * 32768 : v * 32767, true);
  }
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

async function startMic() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw new Error("no microphone API in this browser");
  const stream = await navigator.mediaDevices.getUserMedia({ audio: {
    echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 } });
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  await ctx.audioWorklet.addModule("/static/mic-worklet.js");
  const src = ctx.createMediaStreamSource(stream);
  const tap = new AudioWorkletNode(ctx, "mic-tap");
  const fs = ctx.sampleRate;
  let chunk = new Float32Array(fs), n = 0;
  tap.port.onmessage = async (ev) => {
    const b = ev.data;
    for (let i = 0; i < b.length && n < chunk.length; i++) chunk[n++] = b[i];
    if (n < chunk.length) return;
    const full = chunk; chunk = new Float32Array(fs); n = 0;
    if (sending) return;                                          // never queue: drop a second if the network is slow
    sending = true;
    try {
      const r = await api("/api/ingest/audio", { pcm16_b64: toBase64Pcm16(full), fs, rpm: rpm(), source: "phone-mic" });
      $("rate").textContent = `microphone ${fs} Hz · level ${r.level_dbfs} dBFS`;
      show(r);
    } catch (err) { $("err").textContent = "⚠ " + err.message; }
    finally { sending = false; }
  };
  src.connect(tap);                                               // not connected to the speakers: no feedback
  audio = { ctx, stream };
  $("rate").textContent = `microphone ${fs} Hz · listening…`;
}

$("start").onclick = async () => {
  if (!window.isSecureContext) { $("err").textContent = "⚠ sensors need HTTPS: open this page with https://"; return; }
  try {
    if ($("mode").value === "mic") { if (!audio) await startMic(); }
    else await startMotion();
  } catch (err) { $("err").textContent = "⚠ " + err.message; }
};

$("capture").onclick = async () => {
  try { const r = await api("/api/baseline/capture", { windows: 40 }); $("cap").textContent = `capturing healthy: ${r.have}/${r.want}`; }
  catch (err) { $("err").textContent = "⚠ " + err.message; }
};
