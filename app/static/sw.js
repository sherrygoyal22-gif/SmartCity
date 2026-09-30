/* SmartCity service worker: makes the site installable and keeps the shell
   usable when the connection drops. Pages and API calls always go to the
   network first, so reports, cases and live detection are never stale. */
const VERSION = "smartcity-v3";
const STATIC_CACHE = `${VERSION}-static`;
const OFFLINE_URL = "/offline";
const PRECACHE = [OFFLINE_URL, "/static/css/app.css", "/static/css/portal.css", "/static/css/mobile-app.css", "/static/icons/icon-192.png"];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(STATIC_CACHE).then((cache) => cache.addAll(PRECACHE)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((key) => !key.startsWith(VERSION)).map((key) => caches.delete(key))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;           // map tiles, Leaflet CDN: leave to the browser
  if (url.pathname.startsWith("/api/")) return;              // detection / live camera calls: never cached
  if (url.pathname.startsWith("/static/uploads/") || url.pathname.startsWith("/static/results/")) return;

  if (request.mode === "navigate") {                         // pages: network first, offline page as fallback
    event.respondWith(fetch(request).catch(() => caches.match(OFFLINE_URL)));
    return;
  }

  if (url.pathname.startsWith("/static/")) {                 // css / js / images: serve cached, refresh in background
    event.respondWith(
      caches.open(STATIC_CACHE).then(async (cache) => {
        const cached = await cache.match(request);
        const network = fetch(request).then((response) => {
          if (response.ok) cache.put(request, response.clone());
          return response;
        }).catch(() => cached);
        return cached || network;
      })
    );
  }
});
