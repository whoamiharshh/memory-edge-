"use strict";
// Phone accelerometer -> /api/ingest/signal. Text is set with textContent only (never innerHTML).
// Windows: 256 samples, sent every 128 new samples (50 % overlap), with the sampling rate MEASURED from the event
// timestamps of that window (phones deliver ~60 Hz but not exactly).
const $ = (id) => document.getElementById(id);
const WIN = 256, HOP = 128, KEY = "mm_operator_token";
$("tok").value = sessionStorage.getItem(KEY) || "";
let buf = [], times = [], sinceSend = 0, sending = false, started = false;

async function api(path, body) {
  const token = $("tok").value.trim();
  sessionStorage.setItem(KEY, token);
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json", "X-Operator-Token": token },
    body: JSON.stringify(body) });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof data.detail === "string" ? data.detail : "HTTP " + r.status);
  return data;
}

function onMotion(e) {
  const a = e.accelerationIncludingGravity || e.acceleration;
  if (!a || a.x === null) return;
  buf.push([a.x, a.y, a.z]);
  times.push(performance.now());
  if (buf.length > WIN) { buf.shift(); times.shift(); }
  sinceSend += 1;
  if (buf.length === WIN && sinceSend >= HOP && !sending) send();
}

async function send() {
  sinceSend = 0;
  const span = (times[times.length - 1] - times[0]) / 1000;
  const fs = (WIN - 1) / span;
  $("rate").textContent = `sampling ${fs.toFixed(1)} Hz · window ${span.toFixed(1)} s`;
  if (!(fs > 5 && fs < 1000)) return;
  sending = true;
  try {
    const r = await api("/api/ingest/signal", { axes: buf.slice(), fs, source: "phone" });
    const last = r.last || {};
    $("state").textContent = last.state || "–";
    $("cap").textContent = r.capture ? `capturing healthy: ${r.capture.have}/${r.capture.want}` : (last.state === "baseline" ? "baseline fitted" : "");
    const d = r.diagnosis;
    $("diag").textContent = d && d.rotating ? `shaft ≈ ${d.shaft_hz} Hz (${d.shaft_source}) · hint: ${d.rotating.fault_class} — ${d.rotating.why}` : "";
    $("err").textContent = "";
  } catch (err) {
    $("err").textContent = "⚠ " + err.message;
  } finally {
    sending = false;
  }
}

$("start").onclick = async () => {
  if (!window.isSecureContext) { $("err").textContent = "⚠ motion sensors need HTTPS: open this page with https://"; return; }
  try {
    if (typeof DeviceMotionEvent !== "undefined" && typeof DeviceMotionEvent.requestPermission === "function") {
      const p = await DeviceMotionEvent.requestPermission();         // iOS: must be called from a user gesture
      if (p !== "granted") throw new Error("motion permission denied");
    }
    if (!started) { window.addEventListener("devicemotion", onMotion); started = true; }
    $("rate").textContent = "waiting for sensor data…";
  } catch (err) { $("err").textContent = "⚠ " + err.message; }
};

$("capture").onclick = async () => {
  try { const r = await api("/api/baseline/capture", { windows: 40 }); $("cap").textContent = `capturing healthy: ${r.have}/${r.want}`; }
  catch (err) { $("err").textContent = "⚠ " + err.message; }
};
