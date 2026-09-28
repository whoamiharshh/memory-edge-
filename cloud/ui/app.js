"use strict";
// All data rendered with textContent (never innerHTML): shared notes are untrusted text.
const $ = (id) => document.getElementById(id);
const KEY = "mm_fleet_token";
let token = sessionStorage.getItem(KEY) || "";
let me = null, selected = null;

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
const badge = (t, c) => el("span", { class: "badge " + c }, t);
const flagCls = (k) => (k === "ALTERNATIVES" ? "b-info" : "b-bad");
function toast(msg, err) { const t = el("div", { class: "toast" + (err ? " err" : "") }, msg); document.body.append(t); setTimeout(() => t.remove(), 3500); }

async function api(path, body) {
  const opt = { headers: { Authorization: "Bearer " + token } };
  if (body !== undefined) { opt.method = "POST"; opt.headers["Content-Type"] = "application/json"; opt.body = JSON.stringify(body); }
  const r = await fetch(path, opt);
  if (r.status === 401) { showLogin(); throw new Error("sign-in required"); }
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof d.detail === "string" ? d.detail : JSON.stringify(d.detail || r.status));
  return d;
}
const act = (fn) => async (...a) => { try { await fn(...a); } catch (e) { toast(e.message, true); } };

function showLogin() { $("login").hidden = false; }
$("loginForm").addEventListener("submit", async (e) => {
  e.preventDefault(); token = $("tok").value.trim();
  try { me = await api("/v1/whoami"); sessionStorage.setItem(KEY, token); $("login").hidden = true; boot(); }
  catch { $("loginErr").textContent = "invalid or revoked token"; }
});
$("logout").onclick = () => { sessionStorage.removeItem(KEY); token = ""; showLogin(); };

function actionsTable(c) {
  return el("table", {}, el("thead", {}, el("tr", {}, el("th", {}, "action"), el("th", {}, "worked"), el("th", {}, "failed"), el("th", {}, "sites"),
      el("th", {}, "machine-verified"), el("th", {}, "last confirmed"))),
    el("tbody", {}, ...(c.actions || []).map((a) => el("tr", {}, el("td", {}, a.action_code),
      el("td", {}, badge(String(a.worked), a.worked ? "b-ok" : "b-mute")), el("td", {}, badge(String(a.failed), a.failed ? "b-bad" : "b-mute")),
      el("td", { class: "muted" }, `✓ ${a.sites_worked.join(", ") || "–"}  ✗ ${a.sites_failed.join(", ") || "–"}`),
      el("td", {}, String(a.machine_verified)),
      el("td", { class: "muted" }, a.last_confirmed ? a.last_confirmed.slice(0, 10) : "never")))));
}

async function refreshCases() {
  let cs = await api("/v1/cases");
  if ($("onlyDisputes").checked) cs = cs.filter((c) => (c.flags || []).some((f) => f.kind !== "ALTERNATIVES"));
  $("cCount").textContent = `${cs.length} group(s)`;
  $("cases").replaceChildren(...(cs.length ? cs.map((c) => el("div", { class: "item" + (c.case_id === selected ? " sel" : "") + (c.status === "retracted" ? " strike" : ""), on: { click: () => { selected = c.case_id; refreshDetail(); refreshCases(); } } },
    el("div", { class: "t" }, el("span", {}, `${c.component} / ${c.fault_class}`), badge(`${c.n_sites} site(s)`, "b-info"), badge(`${c.n_events} report(s)`, "b-mute"),
      c.n_retracted ? badge(`${c.n_retracted} retracted`, "b-warn") : "", ...(c.flags || []).map((f) => badge(f.kind, flagCls(f.kind)))),
    el("div", { class: "d" }, (c.actions || []).map((a) => `${a.action_code}: ${a.worked}✓ ${a.failed}✗`).join(" · ") + ` · updated ${c.updated_at}`)))
    : [el("div", { class: "empty" }, "No evidence yet. Devices share only after their own sensor data verified the outcome.")]));
}

async function refreshDetail() {
  if (!selected) return;
  const c = await api("/v1/cases/" + selected);
  const kids = [el("div", { class: "row" }, el("b", {}, `${c.component} / ${c.fault_class}`), badge(c.status, c.status === "active" ? "b-ok" : "b-warn"), el("span", { class: "muted mono" }, "v" + c.version + " · seq " + c.seq)),
    ...(c.flags || []).map((f) => el("div", { class: "row" }, badge(f.kind, flagCls(f.kind)), el("span", {}, f.detail))),
    el("div", { class: "section" }, el("h3", {}, "Evidence tallies"), actionsTable(c)),
    Object.keys(c.root_causes || {}).length ? el("div", { class: "section" }, el("h3", {}, "Root-cause claims"),
      ...Object.entries(c.root_causes).map(([k, v]) => el("div", {}, `${k}: ${v.join(", ")}`))) : "",
    (c.notes || []).length ? el("div", { class: "section" }, el("h3", {}, "Shared (redacted) notes"),
      ...c.notes.map((n) => el("div", { class: "muted" }, `“${n.text}” — ${n.site_id}, ${n.action_code} ${n.outcome}`))) : ""];
  if (c.events) {
    kids.push(el("div", { class: "section" }, el("h3", {}, "Reports (admin)"), el("div", { class: "tablewrap" }, el("table", {},
      el("thead", {}, el("tr", {}, el("th", {}, "site"), el("th", {}, "action"), el("th", {}, "outcome"), el("th", {}, "verified"), el("th", {}, "status"), el("th", {}, ""))),
      el("tbody", {}, ...c.events.map((e) => el("tr", { class: e.status === "retracted" ? "strike" : "" },
        el("td", {}, e.site_id), el("td", {}, e.action_code), el("td", {}, badge(e.outcome, e.outcome === "worked" ? "b-ok" : "b-bad")),
        el("td", { class: "muted" }, `${e.verify_windows_ok}/${e.verify_windows_required} windows`),
        el("td", {}, e.status === "retracted" ? badge("retracted: " + (e.retract_reason || ""), "b-warn")
          : e.status === "quarantined" ? badge("quarantined: " + (e.quarantine_reason || ""), "b-warn")
          : e.retraction_request ? badge(`retraction requested by ${e.retraction_request.by}: ${e.retraction_request.reason}`, "b-warn")
          : badge("active", "b-ok"),
          e.followup ? badge(e.followup.status === "held" ? `held ${e.followup.hold_days} d` : `recurred after ${e.followup.days_after_fix} d`, e.followup.status === "held" ? "b-ok" : "b-bad") : ""),
        el("td", {}, e.status === "active" || !e.status ? el("button", { class: "danger", on: { click: act(async () => {
          const second = !!e.retraction_request;
          const reason = second ? e.retraction_request.reason : window.prompt("Reason for retracting this report (kept as a tombstone; a second admin must confirm):", "fabricated report");
          if (!reason) return;
          const r = await api(`/v1/events/${e.event_id}/retract`, { reason });
          toast(r.retraction === "done" ? "retracted (tombstoned, not deleted)" : "retraction requested: a second admin must confirm");
          refreshDetail(); refreshCases();
        }) } }, e.retraction_request ? "Confirm retraction" : "Retract") : ""))))))));
  }
  $("detail").replaceChildren(...kids);
}

async function refreshDevices() {
  if (!me || me.role !== "admin") return;
  const ds = await api("/v1/admin/devices");
  $("devBody").replaceChildren(...ds.map((d) => el("tr", {}, el("td", {}, d.device_id), el("td", {}, d.site_id), el("td", {}, d.role),
    el("td", {}, d.revoked ? badge("revoked", "b-bad") : badge("active", "b-ok")),
    el("td", { class: "muted" }, Object.entries(d.evidence || {}).map(([k, v]) => `${k} ${v}`).join(" · ") || "–"),
    el("td", {}, d.code ? badge(d.code, d.code === "DIFFERS" ? "b-bad" : "b-ok") : el("span", { class: "muted" }, "not seen yet"),
      d.clock_offset_s != null && Math.abs(d.clock_offset_s) > 120 ? badge(`clock ${(d.clock_offset_s / 3600).toFixed(1)} h off (times corrected)`, "b-warn") : ""),
    el("td", {}, d.role === "device" ? el("span", {},
      !d.revoked ? el("button", { class: "danger", on: { click: act(async () => { await api(`/v1/admin/devices/${d.device_id}/revoke`, {}); toast("revoked"); refreshDevices(); }) } }, "Revoke") : "",
      (d.evidence || {}).active ? el("button", { class: "danger", on: { click: act(async () => {
        const reason = window.prompt("Quarantine ALL evidence of this device (reversible) and revoke its tokens. Reason:", "suspected compromised device");
        if (!reason) return;
        const r = await api(`/v1/admin/devices/${d.device_id}/quarantine`, { reason });
        toast(`quarantined ${r.quarantined} report(s); ${r.cases_recomputed} case(s) recomputed`); refreshDevices(); refreshCases(); }) } }, "Quarantine") : "",
      (d.evidence || {}).quarantined ? el("button", { on: { click: act(async () => {
        const reason = window.prompt("Restore this device's quarantined evidence. Reason:", "investigated: device is fine");
        if (!reason) return;
        await api(`/v1/admin/devices/${d.device_id}/unquarantine`, { reason }); toast("restored (issue a new token for the device)"); refreshDevices(); refreshCases(); }) } }, "Restore") : "") : ""))));
}

async function refreshAudit() {
  if (!me || me.role !== "admin") return;
  const a = await api("/v1/admin/audit?limit=100");
  $("auditChain").replaceChildren(a.chain.ok ? badge(`chain intact · ${a.chain.entries} entries`, "b-ok") : badge(`CHAIN BROKEN at entry ${a.chain.broken_at}`, "b-bad"));
  $("auditBody").replaceChildren(...a.entries.slice().reverse().map((e) => el("tr", {}, el("td", { class: "mono" }, e.at), el("td", {}, e.actor),
    el("td", {}, e.action), el("td", { class: "muted" }, Object.entries(e.detail).map(([k, v]) => `${k}: ${v}`).join(" · ")))));
}

async function boot() {
  me = me || await api("/v1/whoami");
  $("hTen").textContent = me.tenant_id; $("hWho").textContent = `${me.device_id} (${me.role})`;
  $("devCard").hidden = me.role !== "admin";
  $("auditCard").hidden = me.role !== "admin";
  const h = await fetch("/v1/health").then((r) => r.json()); $("hBackend").textContent = "Qdrant: " + h.backend;
  $("onlyDisputes").onchange = act(refreshCases);
  $("issueBtn").onclick = act(async () => { const r = await api("/v1/admin/devices", { device_id: $("nDev").value.trim(), site_id: $("nSite").value.trim() }); $("issued").textContent = `token (shown once): ${r.token}`; refreshDevices(); });
  for (const [fn, ms] of [[refreshCases, 3000], [refreshDetail, 3000], [refreshDevices, 5000], [refreshAudit, 5000]]) { fn().catch(() => {}); setInterval(() => fn().catch(() => {}), ms); }
}
if (!token) showLogin(); else api("/v1/whoami").then((w) => { me = w; boot(); }).catch(showLogin);
