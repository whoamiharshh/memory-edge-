"use strict";
/* Rendered with textContent only — stored text can never inject markup. */

const $ = id => document.getElementById(id);
const DEV = "/proxy/device/";
const CLD = "/proxy/cloud/";

const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "on") for (const [ev, fn] of Object.entries(v)) e.addEventListener(ev, fn);
    else if (v != null) e.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k != null) e.append(k instanceof Node ? k : document.createTextNode(String(k)));
  return e;
};

const sum = o => Object.values(o || {}).reduce((a, b) => a + (typeof b === "number" ? b : 0), 0);
const bytes = n => !n ? "0 B" : n > 1e6 ? (n / 1e6).toFixed(1) + " MB" : n > 1e3 ? (n / 1e3).toFixed(1) + " kB" : n + " B";
const clock = t => {
  if (t == null || t === "") return "never";
  const d = typeof t === "number" ? new Date(t * 1000) : new Date(t);
  return isNaN(d) ? "—" : d.toLocaleTimeString([], { hour12: false });
};
const titleCase = s => String(s || "").replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());

function toast(msg, err = false) {
  const t = el("div", { class: "toast" + (err ? " err" : "") }, msg);
  $("toasts").append(t);
  setTimeout(() => t.remove(), 3600);
}

async function api(path, body, method) {
  const opt = { method: method || (body === undefined ? "GET" : "POST"), headers: {} };
  if (body !== undefined) { opt.headers["Content-Type"] = "application/json"; opt.body = JSON.stringify(body); }
  const r = await fetch(path, opt);
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof d.detail === "string" ? d.detail : `HTTP ${r.status}`);
  return d;
}

/* ══════════ navigation ══════════ */
const LOADERS = { memory: loadMemory, broadcast: loadBroadcast, devices: loadDevices,
                  sync: loadSync, qdrant: renderQdrant };
document.querySelectorAll(".tab").forEach(btn => btn.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t === btn));
  const page = btn.dataset.page;
  document.querySelectorAll(".page").forEach(p => p.classList.toggle("active", p.id === "page-" + page));
  LOADERS[page]?.();
}));

/* ══════════ chat ══════════ */
const SUGGESTIONS = [
  "What problems have you seen recently?",
  "Has anything here been fixed before?",
  "What do you know about me?",
  "Is anything still unresolved?",
];

function buildChips() {
  $("chips").replaceChildren(...SUGGESTIONS.map(s =>
    el("button", { class: "chip", on: { click: () => { $("askInput").value = s; send(); } } }, s)));
}

function bubble(role, text, extra) {
  const wrap = el("div", { class: "turn " + role });
  const body = el("div", { class: "bubble" });
  if (role === "you") body.append(el("div", { class: "btext" }, text));
  else {
    body.append(el("div", { class: "btext" }, ...citeParts(text)));
    if (extra) body.append(extra);
  }
  wrap.append(body);
  return wrap;
}

/* renders [E1] markers as small superscript chips */
function citeParts(text) {
  const out = [];
  let last = 0;
  for (const m of String(text).matchAll(/\[E\d+\]/g)) {
    if (m.index > last) out.push(text.slice(last, m.index));
    out.push(el("span", { class: "cite" }, m[0].slice(1, -1)));
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

function sourcesBlock(used) {
  if (!used?.length) return null;
  const det = el("details", { class: "sources" });
  det.append(el("summary", {}, `Where this came from (${used.length})`));
  const WHERE = { memory: "you added", shared: "shared with you", record: "this device noticed", fleet: "from the fleet" };
  used.forEach(u => det.append(el("div", { class: "src" },
    el("span", { class: "src-key" }, u.key),
    el("span", { class: "src-text" },
      el("span", { class: "src-where" }, WHERE[u.source] || u.source), u.text))));
  return det;
}

let busy = false;
let lastRetrieval = null;

async function send() {
  const q = $("askInput").value.trim();
  if (!q || busy) return;
  busy = true;
  $("askBtn").disabled = true;
  $("welcome")?.remove();
  const scroll = $("chatScroll");
  scroll.append(bubble("you", q));
  $("askInput").value = "";

  const thinking = el("div", { class: "turn bot" },
    el("div", { class: "bubble" }, el("div", { class: "typing" },
      el("i"), el("i"), el("i"))));
  scroll.append(thinking);
  scroll.scrollTop = scroll.scrollHeight;

  try {
    const r = await api(DEV + "ask", { text: q, use_fleet: true, limit: 5 });
    thinking.remove();
    lastRetrieval = r.retrieval || null;
    const b = bubble("bot", r.answer, sourcesBlock(r.used));
    const strip = photoStrip(r.pictures);
    if (strip) b.querySelector(".bubble").append(strip);
    if (!r.grounded) {
      const act = el("div", { class: "offer" },
        el("button", { class: "btn small", on: { click: () => openTeach(q) } }, "Teach it about this"));
      b.querySelector(".bubble").append(act);
    }
    scroll.append(b);
    $("composerNote").textContent =
      `${r.latency_ms} ms · answered on the device` + (r.mode === "llm" ? " · local model" : "");
  } catch (e) {
    thinking.remove();
    scroll.append(bubble("bot", "Something went wrong: " + e.message));
  } finally {
    busy = false;
    $("askBtn").disabled = false;
    scroll.scrollTop = scroll.scrollHeight;
    $("askInput").focus();
  }
}

$("askBtn").addEventListener("click", send);
$("askInput").addEventListener("keydown", e => { if (e.key === "Enter") send(); });

$("clearChat").addEventListener("click", async () => {
  try {
    await api(DEV + "memory/chat", undefined, "DELETE");
    $("chatScroll").replaceChildren();
    buildWelcome();
    loadConversations();
    toast("Started a new conversation. The earlier one is saved.");
  } catch (e) { toast(e.message, true); }
});

function buildWelcome() {
  const w = el("div", { class: "welcome", id: "welcome" },
    el("h1", { class: "hero-title" }, "Ask this device anything"),
    el("p", { class: "hero-sub" }, "It answers from what it remembers — and says so plainly when it has nothing."),
    el("div", { class: "chips", id: "chips" }));
  $("chatScroll").append(w);
  buildChips();
}

/* ── teach modal ── */
function openTeach(prefill) {
  $("teachModal").hidden = false;
  $("teachText").value = prefill ? "" : "";
  $("teachText").placeholder = prefill
    ? `Tell it about "${prefill}" — it will remember and use this next time.`
    : "Anything at all — a fact, a procedure, a preference, a note to your future self.";
  $("teachText").focus();
}
$("teachBtn").addEventListener("click", () => openTeach());
$("addMem")?.addEventListener("click", () => openTeach());
$("teachCancel").addEventListener("click", () => { $("teachModal").hidden = true; });
$("teachModal").addEventListener("click", e => { if (e.target === $("teachModal")) $("teachModal").hidden = true; });

$("teachSave").addEventListener("click", async () => {
  const text = $("teachText").value.trim();
  if (!text) return;
  const btn = $("teachSave");
  btn.disabled = true;
  try {
    await api(DEV + "memory", { text, kind: "fact" });
    $("teachModal").hidden = true;
    $("teachText").value = "";
    toast("Saved. It stays on this device.");
    if ($("page-memory").classList.contains("active")) loadMemory();
  } catch (e) { toast(e.message, true); }
  finally { btn.disabled = false; }
});

/* ══════════ broadcast ══════════ */
const bcAudience = () => document.querySelector('input[name="bcAud"]:checked')?.value || "everyone";
document.querySelectorAll('input[name="bcAud"]').forEach(r => r.addEventListener("change", () => {
  const a = bcAudience();
  $("bcRecipients").hidden = a === "everyone";
  $("bcRecipients").placeholder = a === "site"
    ? "site names, comma separated — e.g. north, depot-2"
    : "device names, comma separated — e.g. phone-2, phone-7";
}));

$("bcSend").addEventListener("click", async () => {
  const text = $("bcText").value.trim();
  if (!text) return toast("Write something first.", true);
  const audience = bcAudience();
  const recipients = audience === "everyone" ? []
    : $("bcRecipients").value.split(",").map(s => s.trim()).filter(Boolean);
  if (audience !== "everyone" && !recipients.length) {
    return toast(`Name at least one ${audience}, or send it to every device.`, true);
  }
  const btn = $("bcSend");
  btn.disabled = true;
  try {
    await api(DEV + "knowledge/publish", { text, audience, recipients,
                                           topic: $("bcTopic").value.trim() || "general" });
    $("bcText").value = "";
    $("bcNote").textContent = audience === "everyone"
      ? "sent to every device" : `sent to ${recipients.length} ${audience}(s)`;
    toast("Broadcast sent");
    loadBroadcast();
  } catch (e) { toast(e.message, true); }
  finally { btn.disabled = false; }
});

$("bcPull").addEventListener("click", async () => {
  const btn = $("bcPull");
  btn.disabled = true;
  try {
    const r = await api(DEV + "knowledge/pull", {});
    $("bcPullNote").textContent = r.skipped ? "offline — cannot check right now"
      : r.error ? "could not reach the other devices"
      : r.received ? `${r.received} new` : "nothing new";
    loadBroadcast();
  } catch (e) { toast(e.message, true); }
  finally { btn.disabled = false; }
});

function bcCard(r, mine) {
  const who = mine
    ? (r.audience === "everyone" ? "to every device"
       : `to ${(r.recipients || []).join(", ") || "nobody"}`)
    : `from ${r.author_device || "another device"}`;
  return el("div", { class: "bc-item" },
    el("div", { class: "bc-top" },
      el("span", { class: "badge " + (r.audience === "everyone" ? "info" : "ok") }, r.topic || "general"),
      el("span", { class: "bc-who" }, who),
      el("span", { class: "bc-time" }, clock(r.created_at || r.updated_at))),
    el("div", { class: "bc-text" }, r.note_text || r.text || ""));
}

async function loadBroadcast() {
  try {
    const inbox = await api(DEV + "knowledge?limit=100");
    $("bcInbox").replaceChildren(
      ...(inbox.length ? inbox.map(r => bcCard(r, false))
                       : [el("div", { class: "empty" }, "Nothing received yet.")]));
  } catch (e) {
    $("bcInbox").replaceChildren(el("div", { class: "empty" }, "Could not read received messages."));
  }
  try {
    const sent = await api(DEV + "knowledge/mine");
    const live = (sent || []).filter(r => r.status === "active");
    $("bcSent").replaceChildren(
      ...(live.length ? live.map(r => bcCard(r, true))
                      : [el("div", { class: "empty" }, "You have not sent anything yet.")]));
  } catch (e) {
    $("bcSent").replaceChildren(el("div", { class: "empty" },
      "Cannot reach the shared store — reconnect to see what you sent."));
  }
}

$("pullShared")?.addEventListener("click", async () => {
  const btn = $("pullShared");
  btn.disabled = true;
  try {
    const r = await api(DEV + "knowledge/pull", {});
    $("pullNote").textContent = r.skipped ? "offline — nothing to check"
      : r.error ? "could not reach the shared store"
      : r.received ? `${r.received} new item(s)` : "already up to date";
    loadMemory();
  } catch (e) { toast(e.message, true); }
  finally { btn.disabled = false; }
});

/* ══════════ conversation history ══════════ */
async function loadConversations() {
  try {
    const rows = await api(DEV + "conversations?limit=30");
    const box = $("convList");
    if (!rows.length) {
      box.replaceChildren(el("div", { class: "empty" }, "No past conversations."));
      return;
    }
    box.replaceChildren(...rows.map(c => {
      const item = el("button", { class: "side-item" },
        el("div", { class: "side-title" }, c.title || "Conversation"),
        el("div", { class: "side-meta" },
          `${c.turns} message${c.turns === 1 ? "" : "s"} · ${clock(c.last_at)}`));
      item.addEventListener("click", () => openConversation(c.session, item));
      return item;
    }));
  } catch (e) { /* history simply stays empty */ }
}

async function openConversation(session, item) {
  try {
    const turns = await api(DEV + "conversations/" + session);
    document.querySelectorAll(".side-item").forEach(i => i.classList.remove("on"));
    item?.classList.add("on");
    const scroll = $("chatScroll");
    scroll.replaceChildren();
    turns.forEach(t => {
      const m = /^Q:\s*([\s\S]*?)\nA:\s*([\s\S]*)$/.exec(t.note_text || "");
      if (!m) return;
      scroll.append(bubble("you", m[1].trim()));
      scroll.append(bubble("bot", m[2].trim()));
    });
    scroll.append(el("div", { class: "replay-note" },
      "You are looking at an earlier conversation. Ask something to start a new one."));
    scroll.scrollTop = scroll.scrollHeight;
  } catch (e) { toast(e.message, true); }
}

$("newChat").addEventListener("click", () => $("clearChat").click());

/* ══════════ qdrant inspector ══════════ */
function renderQdrant() {
  const box = $("qdrantBody");
  if (!lastRetrieval) {
    box.replaceChildren(el("div", { class: "empty" }, "Ask something first, then come back here."));
    return;
  }
  const r = lastRetrieval;
  box.replaceChildren();

  box.append(el("div", { class: "q-hero" },
    el("div", { class: "q-stat" }, el("b", {}, String(r.total_points)), el("span", {}, "vectors searched")),
    el("div", { class: "q-stat" }, el("b", {}, r.search_ms + " ms"), el("span", {}, "on this device")),
    el("div", { class: "q-stat" }, el("b", {}, String(r.kept_sources.length)),
      el("span", {}, "source(s) used"))));

  box.append(el("div", { class: "panel" },
    el("h2", {}, "The question Qdrant was given"),
    el("div", { class: "q-query" }, r.query_sent),
    r.follow_up ? el("p", { class: "hint", style: "margin-top:8px" },
      "This was a follow-up, so the subject of the previous question was added before searching.") : null,
    el("p", { class: "hint", style: "margin:8px 0 0" }, "Searched in: " + r.where)));

  const rows = r.searched.map(sc => el("tr", {},
    el("td", {}, sc.label),
    el("td", { class: "num" }, sc.stored == null ? "—" : String(sc.stored)),
    el("td", { class: "num" }, String(sc.returned)),
    el("td", { class: "num " + (sc.kept ? "w" : "") }, String(sc.kept))));
  box.append(el("div", { class: "panel" },
    el("h2", {}, "Where it looked"),
    el("p", { class: "hint" }, "Every one of these is a filtered query against the same Qdrant Edge shard."),
    el("table", { class: "tally" },
      el("thead", {}, el("tr", {}, el("th", {}, "Kind of memory"), el("th", {}, "Stored"),
        el("th", {}, "Returned"), el("th", {}, "Used in the answer"))),
      el("tbody", {}, ...rows))));

  const vrows = r.vectors.map(v => el("tr", {},
    el("td", {}, el("code", {}, v.name)),
    el("td", {}, v.kind),
    el("td", { class: "num" }, v.dim == null ? "sparse" : String(v.dim)),
    el("td", {}, v.model || "—"),
    el("td", {}, el("span", { class: "badge " + (v.used ? "ok" : "mut") }, v.used ? "compared" : "not used"))));
  box.append(el("div", { class: "panel" },
    el("h2", {}, "Vectors compared"),
    el("p", { class: "hint" }, r.fusion),
    el("table", { class: "tally" },
      el("thead", {}, el("tr", {}, el("th", {}, "Vector"), el("th", {}, "What it holds"),
        el("th", {}, "Size"), el("th", {}, "Made by"), el("th", {}, ""))),
      el("tbody", {}, ...vrows))));
}

/* ══════════ photos ══════════ */
/* Stored and retrieved by CLIP vectors in Qdrant Edge. Nothing describes the picture: typing words finds
   photos, and the note written beside each one is what carries the meaning. */
let pendingPhoto = null;

$("photoBtn").addEventListener("click", () => $("photoFile").click());
$("photoFile").addEventListener("change", async e => {
  const f = e.target.files?.[0];
  e.target.value = "";
  if (!f) return;
  if (f.size > 10 * 1024 * 1024) return toast("That photo is larger than 10 MB.", true);
  pendingPhoto = { name: f.name, b64: await fileToB64(f) };
  $("photoPreview").src = "data:" + (f.type || "image/jpeg") + ";base64," + pendingPhoto.b64;
  $("photoNote").value = "";
  $("photoModal").hidden = false;
  $("photoNote").focus();
});

$("photoCancel").addEventListener("click", () => { $("photoModal").hidden = true; pendingPhoto = null; });
$("photoModal").addEventListener("click", e => {
  if (e.target === $("photoModal")) { $("photoModal").hidden = true; pendingPhoto = null; }
});

$("photoSave").addEventListener("click", async () => {
  if (!pendingPhoto) return;
  const btn = $("photoSave");
  btn.disabled = true;
  btn.textContent = "Saving…";
  try {
    await api(DEV + "pictures", { image_b64: pendingPhoto.b64, filename: pendingPhoto.name,
                                  note: $("photoNote").value.trim() });
    $("photoModal").hidden = true;
    pendingPhoto = null;
    toast("Photo saved. Describe it later to find it again.");
    if ($("page-memory").classList.contains("active")) loadMemory();
  } catch (e) { toast(e.message, true); }
  finally { btn.disabled = false; btn.textContent = "Save photo"; }
});

function fileToB64(file) {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(String(r.result).split(",")[1]);
    r.onerror = rej;
    r.readAsDataURL(file);
  });
}

function photoStrip(pics) {
  if (!pics?.length) return null;
  const wrap = el("div", { class: "pic-strip" },
    el("div", { class: "pic-strip-label" }, "Photos on this device that match"));
  const row = el("div", { class: "pic-row" });
  pics.forEach(p => {
    const fig = el("figure", { class: "pic" },
      el("img", { src: DEV + "pictures/" + p.id + "/file", alt: p.note || p.filename || "photo",
                  loading: "lazy" }));
    if (p.note) fig.append(el("figcaption", {}, p.note));
    row.append(fig);
  });
  wrap.append(row);
  return wrap;
}

/* ══════════ dictation ══════════ */
/* Recorded here, transcribed on the device by shared/speech.py. The audio is never uploaded, which is
   the whole reason this is not the browser's own speech API. */
let recorder = null, chunks = [], recording = false;

async function initMic() {
  try {
    const st = await api(DEV + "speech");
    if (!st.available || !navigator.mediaDevices?.getUserMedia) return;
    $("micBtn").hidden = false;
  } catch (e) { /* dictation simply stays hidden */ }
}

$("micBtn").addEventListener("click", async () => {
  if (recording) { recorder?.stop(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    chunks = [];
    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = e => chunks.push(e.data);
    recorder.onstop = async () => {
      stream.getTracks().forEach(t => t.stop());
      recording = false;
      $("micBtn").classList.remove("rec");
      $("composerNote").textContent = "transcribing on this device…";
      try {
        const wav = await blobToWav(new Blob(chunks));
        const r = await api(DEV + "speech", { audio_b64: wav });
        if (r.text) {
          $("askInput").value = ($("askInput").value + " " + r.text).trim();
          $("composerNote").textContent = `heard ${r.seconds}s · check it before sending`;
          $("askInput").focus();
        } else {
          $("composerNote").textContent = "nothing recognised — try again closer to the microphone";
        }
      } catch (e) { toast(e.message, true); $("composerNote").textContent = ""; }
    };
    recorder.start();
    recording = true;
    $("micBtn").classList.add("rec");
    $("composerNote").textContent = "listening — press again to stop";
  } catch (e) { toast("Could not use the microphone: " + e.message, true); }
});

/* MediaRecorder gives WebM/Opus; the recogniser wants 16-bit PCM WAV, so decode and re-wrap here
   rather than shipping a decoder onto the device. */
async function blobToWav(blob) {
  const buf = await blob.arrayBuffer();
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const audio = await ctx.decodeAudioData(buf);
  const pcm = audio.getChannelData(0);
  const out = new DataView(new ArrayBuffer(44 + pcm.length * 2));
  const str = (o, v) => { for (let i = 0; i < v.length; i++) out.setUint8(o + i, v.charCodeAt(i)); };
  str(0, "RIFF"); out.setUint32(4, 36 + pcm.length * 2, true); str(8, "WAVEfmt ");
  out.setUint32(16, 16, true); out.setUint16(20, 1, true); out.setUint16(22, 1, true);
  out.setUint32(24, audio.sampleRate, true); out.setUint32(28, audio.sampleRate * 2, true);
  out.setUint16(32, 2, true); out.setUint16(34, 16, true);
  str(36, "data"); out.setUint32(40, pcm.length * 2, true);
  for (let i = 0; i < pcm.length; i++) {
    const v = Math.max(-1, Math.min(1, pcm[i]));
    out.setInt16(44 + i * 2, v < 0 ? v * 0x8000 : v * 0x7fff, true);
  }
  let bin = "", bytes = new Uint8Array(out.buffer);
  for (let i = 0; i < bytes.length; i += 0x8000) {
    bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  }
  return btoa(bin);
}

/* ══════════ memory ══════════ */
let memFilter = "all", memSel = null;

document.querySelectorAll("#memFilter .seg-btn").forEach(b =>
  b.addEventListener("click", () => {
    document.querySelectorAll("#memFilter .seg-btn").forEach(x => x.classList.toggle("active", x === b));
    memFilter = b.dataset.f;
    loadMemory();
  }));

async function loadMemory() {
  const list = $("memList");
  const want = k => memFilter === "all" || memFilter === k;
  let facts = [], shared = [], records = [], pics = [], convs = [];
  if (want("fact")) { try { facts = await api(DEV + "memory?kind=fact&limit=50"); } catch (e) { facts = []; } }
  if (want("picture")) { try { pics = await api(DEV + "pictures?limit=100"); } catch (e) { pics = []; } }
  if (want("chat")) { try { convs = await api(DEV + "conversations?limit=50"); } catch (e) { convs = []; } }
  if (want("shared")) { try { shared = await api(DEV + "knowledge?limit=100"); } catch (e) { shared = []; } }
  if (want("record")) { try { records = await api(DEV + "episodes"); } catch (e) { records = []; } }

  const rows = [
    ...(facts || []).map(f => ({
      key: "fact", id: f.id, title: "You added this",
      sub: clock(f.created_at), text: f.note_text || "", raw: f,
    })),
    ...(shared || []).map(s => ({
      key: "shared", id: s.id,
      title: `Shared by ${s.author_device || "another device"}`,
      sub: `${s.topic || "general"} · ${s.audience === "everyone" ? "sent to everyone" : "sent to you"}`,
      text: s.note_text || "", raw: s,
    })),
    ...(pics || []).map(p => ({
      key: "picture", id: p.id, title: p.filename || "Photo",
      sub: clock(p.created_at), text: p.note_text || "", raw: p,
    })),
    ...(convs || []).map(c => ({
      key: "chat", id: c.session, title: c.title || "Conversation",
      sub: `${c.turns} message${c.turns === 1 ? "" : "s"} · ${clock(c.last_at)}`, text: "", raw: c,
    })),
    ...(records || []).map(ep => ({
      key: "record", id: ep.episode_id,
      title: titleCase(ep.fault_class || (ep.fault_hint || {}).fault_class || "Something unusual"),
      sub: `${titleCase(ep.component || "part")} · ${ep.occurrences || 1} readings`,
      text: "", raw: ep,
    })),
  ];

  if (!rows.length) {
    list.replaceChildren(el("div", { class: "empty" },
      memFilter === "fact" ? "Nothing added yet. Use “Add to memory”."
      : memFilter === "shared" ? "Nothing has been shared with this device yet."
      : "Nothing recorded yet."));
    $("memDetail").replaceChildren(el("div", { class: "empty" }, "Select an item to see it."));
    return;
  }

  const TAG = { fact: ["you", "info"], shared: ["shared", "ok"], picture: ["photo", "warn"],
                chat: ["chat", "info"], record: ["noticed", "mut"] };
  list.replaceChildren(...rows.map(r => {
    const [label, cls] = TAG[r.key];
    const item = el("div", { class: "mitem" + (memSel === r.id ? " sel" : "") },
      el("div", { class: "mi-top" },
        el("span", { class: "mi-title" }, r.title),
        el("span", { class: "badge " + cls }, label)),
      el("div", { class: "mi-meta" }, r.text ? r.text.slice(0, 90) : r.sub));
    if (r.key === "picture") {
      item.append(el("img", { class: "mi-thumb", loading: "lazy",
                              src: DEV + "pictures/" + r.id + "/file", alt: r.text || "photo" }));
    }
    item.addEventListener("click", () => {
      memSel = r.id;
      document.querySelectorAll(".mitem").forEach(m => m.classList.remove("sel"));
      item.classList.add("sel");
      r.key === "record" ? showRecord(r.id)
        : r.key === "chat" ? showConversation(r)
        : showText(r);
    });
    return item;
  }));
}

async function showConversation(r) {
  const box = $("memDetail");
  try {
    const turns = await api(DEV + "conversations/" + r.id);
    box.replaceChildren(
      el("h2", { style: "font-size:16px;font-weight:650;margin-bottom:4px" }, r.title),
      el("div", { class: "mi-meta", style: "margin-bottom:14px" }, r.sub + " · stays on this device"));
    turns.forEach(t => {
      const m = /^Q:\s*([\s\S]*?)\nA:\s*([\s\S]*)$/.exec(t.note_text || "");
      if (!m) return;
      box.append(el("div", { class: "conv-q" }, m[1].trim()),
                 el("div", { class: "conv-a" }, m[2].trim()));
    });
  } catch (e) {
    box.replaceChildren(el("div", { class: "empty" }, "Could not load that conversation."));
  }
}

function showText(r) {
  const s = r.raw, isShared = r.key === "shared";
  if (r.key === "picture") {
    $("memDetail").replaceChildren(
      el("img", { class: "detail-photo", src: DEV + "pictures/" + r.id + "/file", alt: r.text || "photo" }),
      el("h2", { style: "font-size:16px;font-weight:650;margin-bottom:4px" }, s.filename || "Photo"),
      el("div", { class: "mi-meta", style: "margin-bottom:12px" },
        clock(s.created_at) + " · stays on this device"),
      el("div", { style: "font-size:14.5px;line-height:1.7;white-space:pre-wrap" },
        r.text || "No note was written for this photo."));
    return;
  }
  const meta = isShared
    ? `From ${s.author_device || "another device"} · ${s.topic || "general"} · ` +
      (s.audience === "everyone" ? "sent to every device" : "sent only to this device")
    : clock(s.created_at) + " · stays on this device";
  $("memDetail").replaceChildren(
    el("h2", { style: "font-size:16px;font-weight:650;margin-bottom:4px" },
      isShared ? "Shared with this device" : "You added this"),
    el("div", { class: "mi-meta", style: "margin-bottom:14px" }, meta),
    el("div", { style: "font-size:14.5px;line-height:1.7;white-space:pre-wrap" }, r.text));
}

const GATE_NAMES = {
  privacy: "Personal data", validation: "Record check", duplicate: "Already sent?",
  evidence: "Evidence", verification: "Proof it worked", human: "Person confirmed",
};
function humanGate(d) {
  return String(d)
    .replace(/redactor found \d+ \w+/i, "a personal name was found in the text")
    .replace(/schema, sizes, enums and numbers valid/i, "all fields look sensible")
    .replace(/no identical evidence shared before/i, "this has not been shared before")
    .replace(/awaiting action: no intervention recorded yet/i, "nobody has recorded a fix yet")
    .replace(/\bname_after_cue\b/gi, "a personal name");
}

async function showRecord(eid) {
  const box = $("memDetail");
  try {
    const ep = await api(DEV + "episodes/" + eid);
    const ph = ep.physics || {}, vf = ep.verify || {};
    box.replaceChildren(
      el("h2", { style: "font-size:16px;font-weight:650;margin-bottom:10px" },
        titleCase(ep.fault_class || ph.bearing?.fault_class || "Something unusual")));

    const facts = [
      ["Where", titleCase(ep.component)],
      ["Seen for", `${ep.occurrences} readings`],
      ["Confirmed by a person", ep.technician_confirmed ? "yes" : "not yet"],
      ["Fix applied", titleCase(ep.action_code)],
      ["Did the fix hold", vf.windows_ok != null ? `${vf.windows_ok} of ${vf.required || "?"} readings back to normal` : null],
    ].filter(([, v]) => v != null && v !== "");
    box.append(el("div", { class: "kv-list" }, ...facts.map(([k, v]) =>
      el("div", { class: "kv-row" }, el("span", {}, k), el("b", {}, String(v))))));

    if (ep.note_text) {
      box.append(el("div", { class: "dsec" },
        el("h3", {}, "Note"),
        el("div", { style: "font-size:14px;line-height:1.65" }, ep.note_text)));
    }

    const dec = ep.decision || {};
    if (dec.reasons?.length) {
      box.append(el("div", { class: "dsec" },
        el("h3", {}, dec.action === "SHARE" ? "May be shared" : "Stays on this device"),
        el("ul", { class: "gates" }, ...dec.reasons.map(r =>
          el("li", { class: r.ok ? "pass" : "stop" },
            el("span", { class: "g-ic" }, r.ok ? "✓" : "✕"),
            el("span", {}, el("b", { style: "font-weight:600" }, GATE_NAMES[r.gate] || titleCase(r.gate)),
              " — ", humanGate(r.detail)))))));
    }
  } catch (e) {
    box.replaceChildren(el("div", { class: "empty" }, "Could not load that."));
  }
}

/* ══════════ devices ══════════ */
const DEVICE_TYPES = [
  { key: "rotating-hf", name: "Industrial systems", blurb: "Motors, pumps, fans, gearboxes." },
  { key: "force-torque", name: "Robots", blurb: "Arms and grippers — collisions, obstructions, slip." },
  { key: "telemetry", name: "Vehicles", blurb: "Usage and load patterns between readouts." },
  { key: "events", name: "Kiosks & software", blurb: "Error codes. A fix counts once the codes stay away." },
  { key: "lowrate-accel", name: "Phones & tablets", blurb: "The device's own motion sensor. No extra hardware." },
  { key: "acoustic", name: "Anything with a microphone", blurb: "Sound picks up problems a phone can hear." },
  { key: "bearing-12k", name: "Test benches", blurb: "High-rate rigs with known parts." },
];

async function loadDevices() {
  let cur = null;
  try { cur = await api(DEV + "profile"); } catch (e) { /* offline */ }
  $("devGrid").replaceChildren(...DEVICE_TYPES.map(d => {
    const on = cur && cur.name === d.key;
    const card = el("div", { class: "dcard" + (on ? " on" : "") });
    if (on) card.append(el("span", { class: "dc-live" }, "RUNNING HERE"));
    card.append(el("div", { class: "dc-name" }, d.name), el("div", { class: "dc-desc" }, d.blurb));
    return card;
  }));
}

/* ══════════ sync ══════════ */
async function loadSync() {
  try {
    const s = await api(DEV + "stats");
    const sy = s.sync || {}, ob = sy.outbox || s.outbox || {};
    $("fnLocal").textContent = sum(s.local_points);
    $("fnQueued").textContent = ob.queued || 0;
    $("fnCloud").textContent = s.mirror_cases || 0;
    $("syState").textContent = sy.online ? "online" : "offline — saving up";
    $("syQueued").textContent = ob.queued || 0;
    $("sySynced").textContent = ob.synced || 0;
    $("syBytes").textContent = bytes(sy.bytes_sent);
    $("syPush").textContent = clock(sy.last_push);
    $("plRaw").textContent = bytes(s.raw_bytes_kept_local) + " held back";
  } catch (e) { /* offline */ }

  try {
    const pd = await api(DEV + "sync/pending");
    $("pendCount").textContent = pd.count;
    $("pendSize").textContent = pd.count ? "~" + bytes(pd.approx_bytes) : "";
    $("pendAdvice").textContent = pd.advice;
    $("pendingBox").classList.toggle("has", pd.count > 0);
    $("syncNow").disabled = !pd.online || !pd.count;
  } catch (e) { /* offline */ }

  try {
    const cases = await api(CLD + "cases");
    const box = $("fleetCases");
    if (!cases.length) {
      box.replaceChildren(el("div", { class: "empty" },
        "Nothing shared yet. Something appears here only after a fix was confirmed to work."));
      return;
    }
    box.replaceChildren(...cases.map(c => {
      const card = el("div", { class: "ccard" });
      const head = el("div", { class: "cc-head" },
        el("span", { class: "cc-title" }, `${titleCase(c.component)} · ${titleCase(c.fault_class)}`));
      (c.flags || []).forEach(f => head.append(
        el("span", { class: "badge " + (f.kind === "ALTERNATIVES" ? "info" : "bad") }, f.kind)));
      card.append(head, el("div", { class: "cc-meta" }, `${c.n_events || 0} confirmed reports`));
      const tallies = c.tallies || {};
      if (Object.keys(tallies).length) {
        card.append(el("table", { class: "tally" },
          el("thead", {}, el("tr", {}, el("th", {}, "Fix"), el("th", {}, "Held"), el("th", {}, "Did not"))),
          el("tbody", {}, ...Object.entries(tallies).map(([a, t]) =>
            el("tr", {}, el("td", {}, titleCase(a)),
              el("td", { class: "num w" }, String(t.worked || 0)),
              el("td", { class: "num f" }, String(t.failed || 0)))))));
      }
      return card;
    }));
  } catch (e) {
    $("fleetCases").replaceChildren(el("div", { class: "empty" },
      "Cannot reach the shared store — this device keeps working on its own."));
  }
}

$("syncNow").addEventListener("click", async () => {
  try { await api(DEV + "sync/now", {}); toast("Sync finished"); loadSync(); }
  catch (e) { toast(e.message, true); }
});

$("netSwitch").addEventListener("click", async () => {
  const goingOffline = $("netSwitch").classList.contains("on");
  try {
    await api(DEV + "network", { online: !goingOffline });
    $("netSwitch").classList.toggle("on", !goingOffline);
    $("netLabel").textContent = goingOffline ? "Offline" : "Online";
    toast(goingOffline ? "Network off — asking still works" : "Back online");
  } catch (e) { toast(e.message, true); }
});

/* ══════════ boot ══════════ */
async function restoreChat() {
  try {
    const turns = await api(DEV + "memory?limit=12");
    if (!Array.isArray(turns) || !turns.length) return false;
    $("welcome")?.remove();
    const scroll = $("chatScroll");
    turns.forEach(t => {
      const m = /^Q:\s*([\s\S]*?)\nA:\s*([\s\S]*)$/.exec(t.note_text || "");
      if (!m) return;
      scroll.append(bubble("you", m[1].trim()));
      scroll.append(bubble("bot", m[2].trim()));
    });
    scroll.scrollTop = scroll.scrollHeight;
    return true;
  } catch (e) { return false; }
}

async function boot() {
  buildChips();
  try {
    const s = await api(DEV + "stats");
    $("connStatus").textContent = "ready";
    $("connStatus").classList.remove("off");
    $("netSwitch").classList.toggle("on", (s.sync || {}).online !== false);
    $("netLabel").textContent = (s.sync || {}).online === false ? "Offline" : "Online";
  } catch (e) {
    $("connStatus").textContent = "device unreachable";
    $("connStatus").classList.add("off");
  }
  await restoreChat();
  loadConversations();
  initMic();
  $("askInput").focus();
}
boot();
