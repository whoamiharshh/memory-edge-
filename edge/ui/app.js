"use strict";
// All data is rendered with textContent (never innerHTML) so note text can never inject markup (XSS).
const $ = (id) => document.getElementById(id);
const TOKEN_KEY = "mm_operator_token";
let token = sessionStorage.getItem(TOKEN_KEY) || "";
let selected = null;
let selectedVersion = null;   // episode version shown on screen: sent with every edit (409 CONFLICT if it moved)
let enums = null;
let lastStats = null;

function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "on") for (const [ev, fn] of Object.entries(v)) e.addEventListener(ev, fn);
    else if (v !== undefined && v !== null) e.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k !== null && k !== undefined) e.append(k instanceof Node ? k : document.createTextNode(String(k)));
  return e;
}
const badge = (txt, cls) => el("span", { class: "badge " + cls }, txt);
const short = (id) => (id ? id.slice(0, 8) : "–");
const fmtBytes = (n) => (n > 1e6 ? (n / 1e6).toFixed(1) + " MB" : n > 1e3 ? (n / 1e3).toFixed(1) + " kB" : n + " B");
const fmtTime = (t) => (t ? new Date(t * 1000).toLocaleTimeString() : "–");
const cls = (s) => ({ open: "b-warn", verifying: "b-info", closed: "b-ok", local: "b-mute", queued: "b-warn", uploading: "b-info",
  synced: "b-ok", failed: "b-bad", rejected: "b-bad", SHARE: "b-ok", KEEP_LOCAL: "b-warn", MERGE: "b-info", REJECT: "b-bad",
  normal: "b-ok", merge: "b-info", new: "b-bad", worked: "b-ok", pending: "b-mute" }[s] || "b-mute");

function toast(msg, err = false) {
  const t = el("div", { class: "toast" + (err ? " err" : "") }, msg);
  document.body.append(t);
  setTimeout(() => t.remove(), 3500);
}

async function api(path, body) {
  const opt = { headers: { "X-Operator-Token": token } };
  if (body !== undefined) { opt.method = "POST"; opt.headers["Content-Type"] = "application/json"; opt.body = JSON.stringify(body); }
  const r = await fetch(path, opt);
  if (r.status === 401) { showLogin(); throw new Error("sign-in required"); }
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || r.status));
  return data;
}
const act = (fn) => async (...a) => {
  try { await fn(...a); }
  catch (e) {
    toast(e.message, true);
    if (e.message.startsWith("CONFLICT")) { $("noteTxt").dataset.dirty = ""; $("noteTxt").dataset.ep = ""; refreshDetail().catch(() => {}); }  // reload
  }
};

// ---------------- login ----------------
function showLogin() { $("login").hidden = false; $("tok").focus(); }
$("loginForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  token = $("tok").value.trim();
  try { await api("/api/stats"); sessionStorage.setItem(TOKEN_KEY, token); $("login").hidden = true; boot(); }
  catch (err) { $("loginErr").textContent = "invalid token"; }
});
$("logout").onclick = () => { sessionStorage.removeItem(TOKEN_KEY); token = ""; showLogin(); };

// ---------------- chart ----------------
function drawChart(recent, gate) {
  const c = $("chart"), ctx = c.getContext("2d");
  const W = (c.width = c.clientWidth * devicePixelRatio), H = (c.height = 190 * devicePixelRatio);
  ctx.clearRect(0, 0, W, H);
  const pad = 30 * devicePixelRatio;
  const vals = recent.map((r) => r.d);
  const maxV = Math.max(gate ? gate.tau_merge * 1.1 : 10, ...vals, 10);
  const y = (v) => H - pad - (v / maxV) * (H - 2 * pad);
  const x = (i) => pad + (i / Math.max(1, 239)) * (W - 2 * pad);
  ctx.font = `${11 * devicePixelRatio}px sans-serif`;
  if (gate) {
    ctx.fillStyle = "rgba(63,185,80,.10)"; ctx.fillRect(pad, y(gate.tau_normal), W - 2 * pad, y(0) - y(gate.tau_normal));
    ctx.strokeStyle = "rgba(63,185,80,.8)"; ctx.setLineDash([6, 4]); ctx.beginPath(); ctx.moveTo(pad, y(gate.tau_normal)); ctx.lineTo(W - pad, y(gate.tau_normal)); ctx.stroke();
    ctx.setLineDash([]); ctx.fillStyle = "#3fb950"; ctx.fillText("healthy radius τ=" + gate.tau_normal.toFixed(1), pad + 4, y(gate.tau_normal) - 4);
  }
  ctx.fillStyle = "#8b98a8"; ctx.fillText(maxV.toFixed(0), 4, pad); ctx.fillText("0", 4, H - pad);
  ctx.lineWidth = 2 * devicePixelRatio; ctx.strokeStyle = "#5aa9ff"; ctx.beginPath();
  recent.forEach((r, i) => (i ? ctx.lineTo(x(i), y(r.d)) : ctx.moveTo(x(i), y(r.d))));
  ctx.stroke();
  recent.forEach((r, i) => {
    if (r.state === "new") { ctx.fillStyle = "#f85149"; ctx.beginPath(); ctx.arc(x(i), y(r.d), 4 * devicePixelRatio, 0, 7); ctx.fill(); }
  });
  if (!recent.length) { ctx.fillStyle = "#8b98a8"; ctx.fillText("no windows yet — fit the baseline, then press Play", pad, H / 2); }
}

// ---------------- stats / header ----------------
async function refreshStats() {
  const s = await api("/api/stats");
  lastStats = s;
  $("hDev").textContent = s.device_id; $("hSite").textContent = s.site_id; $("hMach").textContent = s.machine_id;
  $("hModel").textContent = s.embedder;
  const w = s.windows || {};
  $("kWin").textContent = w.windows || 0; $("kNorm").textContent = w.normal || 0; $("kNew").textContent = w.new || 0;
  const st = s.last_gate ? s.last_gate.state : "–";
  $("kState").replaceChildren(s.last_gate ? badge(st.toUpperCase(), cls(st)) : "–");
  $("kGate").textContent = s.gate_ms.p50 == null ? "–" : `${s.gate_ms.p50} / ${s.gate_ms.p95}`;
  $("kTau").textContent = s.gate ? `${s.gate.tau_normal.toFixed(1)} / ${s.gate.tau_merge.toFixed(1)}` : "not fitted";
  $("fitBtn").textContent = s.baseline_ready ? "Re-fit healthy baseline" : "Fit healthy baseline";
  drawChart(s.recent || [], s.gate);
  const o = s.outbox || {};
  $("oQ").textContent = (o.queued || 0) + (o.uploading || 0); $("oS").textContent = o.synced || 0;
  $("oF").textContent = o.failed || 0; $("oR").textContent = o.rejected || 0; $("mC").textContent = s.mirror_cases;
  $("pts").textContent = Object.values(s.local_points || {}).reduce((a, b) => a + b, 0);
  const sy = s.sync;
  $("bSent").textContent = fmtBytes(sy.bytes_sent); $("bRaw").textContent = fmtBytes(s.raw_bytes_kept_local);
  $("lPush").textContent = fmtTime(sy.last_push); $("lPull").textContent = fmtTime(sy.last_pull);
  $("syncErr").textContent = sy.last_error ? "⚠ " + sy.last_error : "";
  const mi = sy.mirror || {}, lr = mi.last_refresh;
  $("mMode").textContent = (mi.mode ? mi.mode + " fill" : "not pulled yet") + (mi.needs_full ? " · full snapshot due" : "") +
    (lr ? (lr.kind === "scroll"
      ? ` · last: scroll delta, ${fmtBytes(lr.wire_bytes)} on the wire, ${lr.cases_changed} case(s) changed`
      : ` · last: ${lr.kind} Qdrant snapshot, ${fmtBytes(lr.wire_bytes)} on the wire for ${fmtBytes(lr.snapshot_bytes)}, ${lr.cases_changed} case(s) changed`) : "");
  $("netSwitch").classList.toggle("on", sy.online);
  $("netLabel").textContent = sy.online ? "ONLINE" : "OFFLINE";
  const hs = $("hSync");
  hs.className = "badge " + (sy.auth_required ? "b-bad" : !sy.online ? "b-warn" : sy.last_error ? "b-bad" : "b-ok");
  hs.textContent = sy.auth_required ? "auth required" : !sy.online ? "offline · queueing" : sy.last_error ? "sync error" : "sync ok";
  const r = s.replay;
  $("replayBar").style.width = r.total ? (100 * r.done / r.total).toFixed(0) + "%" : "0%";
  $("replayTxt").textContent = r.playing ? `playing file ${r.fid}: ${r.done}/${r.total}` : r.fid ? `file ${r.fid} done (${r.done})` : "idle";
  $("llmInfo").textContent = s.llm.available ? s.llm.model : "model not installed: template mode";
}

// ---------------- episodes ----------------
async function refreshEpisodes() {
  const eps = await api("/api/episodes");
  $("epCount").textContent = eps.length ? `${eps.length} total` : "";
  const list = $("epList");
  if (!eps.length) { list.replaceChildren(el("div", { class: "empty" }, "No episodes yet. Fit the baseline, then replay a fault file.")); return; }
  list.replaceChildren(...eps.map((e) => {
    const fc = e.fault_class || (e.fault_hint || {}).fault_class;
    return el("div", { class: "item" + (e.episode_id === selected ? " sel" : ""), on: { click: () => select(e.episode_id) } },
      el("div", { class: "t" }, `#${e.seq}`, badge(e.status, cls(e.status)), badge(e.share_state, cls(e.share_state)),
        e.decision ? badge(e.decision.action, cls(e.decision.action)) : null, e.archived ? badge("archived", "b-mute") : null,
        e.dismissed ? badge("normal operation", "b-ok") : null),
      el("div", { class: "d" }, `${e.component} · ${fc}${e.fault_class ? " (confirmed)" : " (hint)"} · ${e.occurrences} windows`
        + (e.action_code ? ` · ${e.action_code} → ${e.outcome}` : "") + (e.recurrence_of ? " · recurrence" : "")));
  }));
  if (!selected && eps.length) select(eps[0].episode_id);
}

async function select(id) { selected = id; await refreshDetail(); refreshEpisodes().catch(() => {}); }

async function refreshDetail() {
  if (!selected) return;
  const e = await api("/api/episodes/" + selected);
  $("detailEmpty").hidden = true; $("detail").hidden = false;
  $("dId").textContent = e.episode_id;
  selectedVersion = e.version;
  $("dBadges").replaceChildren(badge("status: " + e.status, cls(e.status)), badge("share: " + e.share_state, cls(e.share_state)),
    e.outbox_status ? badge("outbox: " + e.outbox_status, cls(e.outbox_status)) : "", e.recurrence_of ? badge("recurrence of " + short(e.recurrence_of), "b-violet") : "",
    badge("v" + e.version, "b-mute"), e.archived ? badge("archived " + (e.archived_at || "").slice(0, 10), "b-mute") : "");
  const uo = e.untaught_operating_point;
  $("dOp").hidden = !uo;
  if (uo) $("dOp").textContent = "⚠ Untaught operating point (" + Object.entries(uo).filter(([k]) => k !== "suggestion")
    .map(([k, v]) => `${k} ${v.value}, taught ${v.taught[0]}–${v.taught[1]}`).join("; ") + "): " + uo.suggestion;
  $("dMeta").textContent = `first seen ${new Date(e.first_seen).toLocaleTimeString()} · last seen ${new Date(e.last_seen).toLocaleTimeString()} · ${e.occurrences} abnormal windows · ${e.n_exemplars} exemplars stored`;
  const h = e.fault_hint || {};
  const acc = h.measured_accuracy == null ? "n/a" : (h.measured_accuracy * 100).toFixed(0) + "%";
  $("dHint").replaceChildren(el("div", {}, "Physics hint: ", el("b", {}, h.fault_class || "–"), ` — ${h.why || ""}`),
    el("div", {}, `measured accuracy for this class on unseen bearings: ${acc} (overall ${h.measured_overall == null ? "n/a" : (h.measured_overall * 100).toFixed(0) + "%"}) · a hint, not a diagnosis`),
    el("div", {}, "Confirmed: ", el("b", {}, e.fault_class ? `${e.fault_class} (by technician)` : "not yet")));
  if (document.activeElement !== $("fcSel")) $("fcSel").value = e.fault_class || h.fault_class || "unknown";
  renderPhysics(e.physics);
  refreshProcedures(e).catch(() => {});
  const nt = $("noteTxt");   // reload the note unless the technician has unsaved typing in it
  if (document.activeElement !== nt && (nt.dataset.ep !== e.episode_id || !nt.dataset.dirty)) {
    nt.dataset.dirty = "";
    $("noteTxt").value = e.note_text || ""; $("noteOpt").checked = !!e.note_share_opt_in; $("noteTxt").dataset.ep = e.episode_id;
    $("noteTxt").dataset.ver = e.version;   // the note being edited is based on THIS version
  }
  if (e.action_code && document.activeElement !== $("actSel")) $("actSel").value = e.action_code;
  const v = e.verify;
  if (v) {
    const pct = v.verdict === "symptom_persists" ? 100 : Math.min(100, (100 * v.consecutive_ok) / v.required);
    $("vBar").style.width = pct + "%";
    $("vBar").style.background = v.verdict === "symptom_persists" ? "var(--bad)" : v.verdict === "symptom_resolved" ? "var(--ok)" : "var(--accent)";
    $("vTxt").textContent = v.verdict === "symptom_resolved" ? `symptom resolved for ${v.consecutive_ok} consecutive windows (not a root-cause proof)`
      : v.verdict === "symptom_persists" ? `symptom persists: ${v.consecutive_bad} consecutive abnormal windows after the action`
      : `verifying: ${v.consecutive_ok}/${v.required} consecutive healthy windows (${v.seen} seen)`;
  } else { $("vBar").style.width = "0%"; $("vTxt").textContent = "no action recorded yet"; }
  $("vTxt").append(e.technician_confirmed ? el("span", {}, " · technician: ", badge(e.outcome, cls(e.outcome === "failed" ? "failed" : e.outcome))) : "");
  const d = e.decision;
  $("decBadge").replaceChildren(d ? badge(d.action, cls(d.action)) : "", d && d.event_id ? el("span", { class: "mono muted" }, " event " + short(d.event_id)) : "",
    d && d.action === "SHARE" ? el("span", { class: "muted" }, d.note_shared ? " · redacted note included" : " · note stays local") : "");
  $("gates").replaceChildren(...(d ? d.reasons : []).map((r) =>
    el("li", {}, el("span", { class: "ic " + (r.ok ? "ok" : "no") }, r.ok ? "✓" : "…"), el("span", { class: "g" }, r.gate), el("span", {}, r.detail))));
}

// ---------------- physics + documented procedures ----------------
function renderPhysics(p) {
  $("dPhysBox").hidden = !p;
  if (!p) return;
  const lines = [];
  if (p.severity && p.severity.zone) lines.push(el("div", {}, "Severity zone ", badge(p.severity.zone, { A: "b-ok", B: "b-ok", C: "b-warn", D: "b-bad" }[p.severity.zone]),
    ` ${p.severity.velocity_mm_s} mm/s RMS — ${p.severity.text}`, el("span", { class: "muted" }, ` (${p.severity.reference})`)));
  if (p.shaft_hz) lines.push(el("div", {}, `Shaft ≈ ${p.shaft_hz} Hz (${Math.round(p.shaft_hz * 60)} rpm)${p.shaft_source ? " · " + p.shaft_source : ""}`));
  if (p.defect_frequencies_hz) lines.push(el("div", {}, "Defect frequencies: " + Object.entries(p.defect_frequencies_hz).map(([k, v]) => `${k.toUpperCase()} ${v} Hz`).join(" · ")));
  if (p.bearing) lines.push(el("div", {}, `Envelope rule → ${p.bearing.fault_class}: ${p.bearing.why}`));
  if (p.rotating) lines.push(el("div", {}, `Order rule → ${p.rotating.fault_class}: ${p.rotating.why}`));
  $("dPhys").replaceChildren(...lines);
}

let procKey = "";
async function refreshProcedures(e) {
  const fc = e.fault_class || (e.fault_hint || {}).fault_class;
  const key = e.episode_id + "|" + fc;
  if (key === procKey) return;                                   // only refetch when the class changes
  procKey = key;
  const r = await api("/api/procedures?episode_id=" + encodeURIComponent(e.episode_id));
  if (!r.procedures.length) { $("dProc").replaceChildren(el("div", { class: "muted" }, `no documented procedure for "${fc}" yet — sites can add their SOPs in knowledge/site_procedures.json`)); return; }
  $("dProc").replaceChildren(...r.procedures.map((p) => el("details", {},
    el("summary", {}, p.title, " ", badge(p.origin === "site" ? "site SOP" : "reference", p.origin === "site" ? "b-violet" : "b-info"),
      e.fault_class ? "" : el("span", { class: "muted" }, " (from the physics hint — confirm the class first)")),
    p.confirm_first ? el("div", {}, el("b", {}, "Confirm first"), el("ul", {}, ...p.confirm_first.map((s) => el("li", {}, s)))) : "",
    el("div", {}, el("b", {}, "Steps"), el("ol", {}, ...p.steps.map((s) => el("li", {}, s)))),
    p.system_verifies ? el("div", { class: "muted" }, "Afterwards: " + p.system_verifies) : "",
    el("div", { class: "muted" }, "Source: ", ...p.sources.map((s, i) => [i ? " · " : "", s.url ? el("a", { href: s.url, target: "_blank", rel: "noopener noreferrer" }, `${s.publisher || ""} — ${s.title}`) : `${s.title} (${s.kind})`]).flat()))));
}

// ---------------- outbox / activity / mirror ----------------
async function refreshOutbox() {
  const rows = await api("/api/outbox");
  $("obBody").replaceChildren(...(rows.length ? rows.map((r) => el("tr", {},
    el("td", { class: "mono" }, short(r.event_id)), el("td", { class: "mono" }, short(r.episode_id)),
    el("td", {}, badge(r.status, cls(r.status))), el("td", {}, r.attempts), el("td", { class: "muted" }, r.last_error || ""),
    el("td", {}, r.status === "synced" ? el("button", { on: { click: act(async () => { await api(`/api/outbox/${r.event_id}/resend`, {}); toast("re-queued: the cloud should answer 'duplicate'"); }) } }, "Resend") : "")))
    : [el("tr", {}, el("td", { colspan: 6, class: "muted" }, "empty — nothing has been decided SHARE yet"))]));
}
async function refreshActivity() {
  const rows = await api("/api/activity?limit=80");
  $("feed").replaceChildren(...rows.map((r) => el("div", {}, el("span", { class: "muted mono" }, r.ts.slice(11, 19)),
    el("span", { class: "k" }, r.kind), r.episode_id ? el("span", { class: "mono muted" }, short(r.episode_id) + " ") : "", r.message)));
}
function caseCard(c, extra) {
  const retracted = c.status === "retracted";
  return el("div", { class: "res" + (retracted ? " strike" : "") },
    el("div", { class: "t row" }, el("b", {}, `${c.component} / ${c.fault_class}`), badge(`${c.n_sites} site(s)`, "b-info"), badge(`${c.n_events} report(s)`, "b-mute"),
      ...(c.flags || []).map((f) => badge(f.kind, f.kind === "ALTERNATIVES" ? "b-info" : "b-bad")), extra || ""),
    el("table", {}, el("tbody", {}, ...(c.actions || []).map((a) => el("tr", {},
      el("td", {}, a.action_code), el("td", {}, badge(`worked ${a.worked}`, a.worked ? "b-ok" : "b-mute")), el("td", {}, badge(`failed ${a.failed}`, a.failed ? "b-bad" : "b-mute")),
      el("td", { class: "muted" }, `${a.machine_verified} machine-verified · sites ${[...a.sites_worked, ...a.sites_failed].join(", ")}` +
        ` · last confirmed ${a.last_confirmed ? a.last_confirmed.slice(0, 10) : "never"}`))))),
    c.n_duplicates_collapsed ? el("div", { class: "muted" }, `${c.n_duplicates_collapsed} duplicate report(s) of the same repair counted once`) : "",
    ...(c.flags || []).map((f) => el("div", { class: "muted" }, "⚑ " + f.detail)),
    ...(c.notes || []).slice(0, 2).map((n) => el("div", { class: "muted" }, `“${n.text}” — ${n.site_id}, ${n.action_code} ${n.outcome}`)));
}
async function refreshMirror() {
  const cs = await api("/api/mirror");
  $("mirror").replaceChildren(...(cs.length ? cs.map((c) => caseCard(c)) : [el("div", { class: "empty" }, "Mirror empty. Go online and sync to pull fleet evidence.")]));
}

// ---------------- search / brief ----------------
const legBadges = (legs) => el("div", { class: "legs" }, ...Object.entries(legs || {}).map(([k, v]) =>
  badge(`${k === "note_bm25" ? "bm25" : k} ${v ? "#" + v : "–"}`, v ? "b-info" : "b-mute")));
// usefulness feedback: a counter on this device only; it never changes the ranking
function feedbackRow(h, kind) {
  const fb = h.feedback || { helped: 0, not_helped: 0 };
  const out = el("span", { class: "muted" }, ` helped ${fb.helped} · didn't ${fb.not_helped}`);
  const send = (helped) => act(async () => {
    const n = await api("/api/search/feedback", { result_id: h.id, kind, helped });
    out.textContent = ` helped ${n.helped} · didn't ${n.not_helped}`;
  });
  return el("div", { class: "row fb" }, el("button", { class: "mini", on: { click: send(true) } }, "Helped"),
    el("button", { class: "mini", on: { click: send(false) } }, "Didn't help"), out);
}
function renderSearch(res) {
  const q = res.query;
  $("sInfo").textContent = `${res.latency_ms} ms · ${q.episode_id ? "episode " + short(q.episode_id) + " · " : ""}` +
    (q.fleet_filter ? `fleet filter: ${Object.entries(q.fleet_filter).map(([k, v]) => k + "=" + v).join(", ")}` +
      (q.fault_class_source ? ` (fault class from ${q.fault_class_source})` : "") : "fleet: mirror empty or off");
  $("sLocal").replaceChildren(...(res.local.length ? res.local.map((h) => {
    const e = h.episode;
    return el("div", { class: "res" }, el("div", { class: "row" }, el("b", {}, `#${e.seq}`), badge(e.status, cls(e.status)),
      el("span", { class: "muted" }, `${e.fault_class || (e.fault_hint || {}).fault_class} · ${e.action_code || "no action"} → ${e.outcome}`)),
      e.note_text ? el("div", { class: "muted" }, "“" + e.note_text + "”") : "", legBadges(h.legs), el("div", { class: "muted mono" }, "rrf " + h.rrf),
      feedbackRow(h, "local"));
  }) : [el("div", { class: "empty" }, "no match on this machine")]));
  $("sFleet").replaceChildren(...(res.fleet.length ? res.fleet.map((h) => { const card = caseCard(h.case, legBadges(h.legs)); card.append(feedbackRow(h, "fleet")); return card; })
    : [el("div", { class: "empty" }, res.fleet_error || "no fleet evidence for this filter")]));
}
function renderBrief(b) {
  const text = el("div", {});
  for (const part of b.text.split(/(\[E\d+\])/)) text.append(/^\[E\d+\]$/.test(part) ? el("span", { class: "cite" }, part) : document.createTextNode(part));
  $("briefOut").replaceChildren(el("div", { class: "brief" },
    el("div", { class: "label" }, b.label), el("div", { class: "row" }, badge(b.mode === "llm" ? "LLM · grounded" : b.mode, b.mode === "llm" ? "b-violet" : "b-mute"),
      b.model ? el("span", { class: "muted" }, b.model) : "", b.latency_ms ? el("span", { class: "muted mono" }, b.latency_ms + " ms") : "", b.why ? el("span", { class: "muted" }, b.why) : ""),
    text,
    b.dropped && b.dropped.length ? el("details", {}, el("summary", {}, `${b.dropped.length} sentence(s) removed by the grounding / no-advice check`),
      ...b.dropped.map((d) => el("div", { class: "muted" }, `✗ ${d.sentence} — ${d.why}`))) : "",
    el("details", { open: "" }, el("summary", {}, `evidence (${b.evidence.length})`),
      ...b.evidence.map((i) => el("div", { class: "muted", style: "margin-top:4px" }, el("span", { class: "cite" }, `[${i.key}] `), `${i.source}: ${i.text}`)))));
}

// ---------------- wiring ----------------
function wire() {
  $("fitBtn").onclick = act(async () => { const g = await api("/api/baseline/fit", {}); toast(`baseline fitted on ${g.n_windows} healthy windows; τ=${g.tau_normal.toFixed(2)}`); });
  $("playBtn").onclick = act(async () => { await api("/api/replay", { fid: +$("fileSel").value, n: +$("nWin").value || null, interval: +$("ivl").value }); });
  $("stopBtn").onclick = act(() => api("/api/replay/stop", {}));
  $("netSwitch").onclick = act(async () => { await api("/api/network", { online: !(lastStats && lastStats.sync.online) }); refreshStats(); });
  $("syncNow").onclick = act(async () => { const r = await api("/api/sync/now", {}); toast("push: " + JSON.stringify(r.push) + " · pull: " + JSON.stringify(r.pull)); });
  $("setTok").onclick = act(async () => { await api("/api/sync/token", { token: $("devTok").value.trim() }); $("devTok").value = ""; toast("device token updated"); });
  const v = () => selectedVersion;
  $("noteTxt").addEventListener("input", () => { $("noteTxt").dataset.dirty = "1"; });
  $("fcBtn").onclick = act(async () => { await api(`/api/episodes/${selected}/fault_class`, { fault_class: $("fcSel").value, expected_version: v() }); refreshDetail(); });
  $("noteBtn").onclick = act(async () => {
    await api(`/api/episodes/${selected}/note`, { text: $("noteTxt").value, share_opt_in: $("noteOpt").checked, expected_version: +$("noteTxt").dataset.ver || v() });
    $("noteTxt").dataset.dirty = ""; $("noteTxt").dataset.ep = ""; toast("note saved"); refreshDetail();
  });
  $("actBtn").onclick = act(async () => { await api(`/api/episodes/${selected}/action`, { action_code: $("actSel").value, root_cause: $("rcSel").value || null, required_windows: +$("reqWin").value, expected_version: v() }); toast("action recorded: now replay the post-repair signal"); refreshDetail(); });
  $("okBtn").onclick = act(async () => { await api(`/api/episodes/${selected}/confirm`, { outcome: "worked", expected_version: v() }); refreshDetail(); });
  $("failBtn").onclick = act(async () => { await api(`/api/episodes/${selected}/confirm`, { outcome: "failed", expected_version: v() }); refreshDetail(); });
  $("normalBtn").onclick = act(async () => {
    if (!confirm("Mark this episode as NORMAL operation (not a fault)? Its fingerprints become part of this machine's healthy baseline.")) return;
    await api(`/api/episodes/${selected}/normal`, { expected_version: v() }); toast("taught as normal operation"); refreshDetail();
  });
  $("searchBtn").onclick = act(async () => renderSearch(await api("/api/search", { text: $("q").value || null, use_fleet: $("useFleet").checked })));
  $("simBtn").onclick = act(async () => { if (!selected) throw new Error("select an episode first"); renderSearch(await api("/api/search", { episode_id: selected, text: $("q").value || null, use_fleet: $("useFleet").checked })); });
  $("q").addEventListener("keydown", (e) => { if (e.key === "Enter") $("searchBtn").click(); });
  $("briefBtn").onclick = act(async () => {
    $("briefOut").replaceChildren(el("div", { class: "muted" }, "retrieving evidence and running the local model…"));
    const b = await api("/api/brief", { question: $("bq").value, episode_id: selected, text: $("q").value || null, use_fleet: $("useFleet").checked });
    renderBrief(b); renderSearch(b.retrieval);
  });
}

async function boot() {
  enums = await api("/api/enums");
  $("fcSel").replaceChildren(...enums.fault_classes.map((f) => el("option", { value: f }, f)));
  $("actSel").replaceChildren(...enums.action_codes.map((a) => el("option", { value: a }, a)));
  $("rcSel").replaceChildren(el("option", { value: "" }, "root cause (optional)"), ...enums.root_causes.map((r) => el("option", { value: r }, r)));
  const cat = await api("/api/replay/catalogue");
  $("fileSel").replaceChildren(...cat.map((c) => el("option", { value: c.fid },
    `${c.fid} · ${c.fault_class}${c.size_mil ? " " + c.size_mil + " mil" : ""} · load ${c.load_hp} hp · ${c.role}`)));
  $("fileSel").value = "105";
  const loops = [[refreshStats, 1000], [refreshEpisodes, 2000], [refreshDetail, 1500], [refreshOutbox, 2500], [refreshActivity, 2500], [refreshMirror, 5000]];
  for (const [fn, ms] of loops) { fn().catch(() => {}); setInterval(() => fn().catch(() => {}), ms); }
}

wire();
if (!token) showLogin(); else api("/api/stats").then(boot).catch(() => showLogin());
