"use strict";
// Service worker for the installable device web app. It caches ONLY the app shell (page, script, styles, icon) so
// the app opens instantly and still shows its screen when the device is briefly unreachable. It NEVER caches /api/*:
// technician notes and episodes must not sit in the phone browser's cache (a shared phone would leak them).
const SHELL = "mm-shell-v1";
const FILES = ["/", "/sensor", "/static/app.js", "/static/style.css", "/static/sensor.js", "/static/icon.svg",
  "/static/manifest.webmanifest"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(SHELL).then((c) => c.addAll(FILES)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== SHELL).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin || url.pathname.startsWith("/api/")) return;
  // network first (always the newest UI when the device is reachable), cache as fallback
  e.respondWith(fetch(e.request).then((r) => {
    const copy = r.clone();
    caches.open(SHELL).then((c) => c.put(e.request, copy));
    return r;
  }).catch(() => caches.match(e.request)));
});
