// Service worker: caches the app shell, libraries and model weights so the app works offline.
const VERSION = 'v1';
const CORE = [
 "./",
 "index.html",
 "style.css",
 "app.js",
 "i18n.js",
 "labels.js",
 "manifest.json",
 "lib/coco-ssd.min.js",
 "lib/knn-classifier.min.js",
 "lib/mobilenet.min.js",
 "lib/tf.min.js",
 "icons/icon-192.png",
 "icons/icon-512.png",
 "models/mobilenet/group1-shard1of4.bin",
 "models/mobilenet/group1-shard2of4.bin",
 "models/mobilenet/group1-shard3of4.bin",
 "models/mobilenet/group1-shard4of4.bin",
 "models/mobilenet/model.json"
];
// COCO-SSD weights (~18 MB) are cached too, but failure to cache them does not block install.
const OPTIONAL = [
 "models/coco/group1-shard1of5",
 "models/coco/group1-shard2of5",
 "models/coco/group1-shard3of5",
 "models/coco/group1-shard4of5",
 "models/coco/group1-shard5of5",
 "models/coco/model.json"
];
self.addEventListener('install', (e) => {
  e.waitUntil((async () => {
    const c = await caches.open(VERSION);
    await c.addAll(CORE);
    try { await c.addAll(OPTIONAL); } catch (err) { console.warn('optional cache failed', err); }
    await self.skipWaiting();
  })());
});
self.addEventListener('activate', (e) => {
  e.waitUntil((async () => {
    for (const k of await caches.keys()) if (k !== VERSION) await caches.delete(k);
    await self.clients.claim();
    for (const cl of await self.clients.matchAll()) cl.postMessage('cached');
  })());
});
self.addEventListener('fetch', (e) => {
  if (e.request.method !== 'GET' || new URL(e.request.url).origin !== location.origin) return;
  e.respondWith((async () => {
    const c = await caches.open(VERSION);
    const hit = await c.match(e.request, { ignoreSearch: true });
    if (hit) return hit;
    try {
      const res = await fetch(e.request);
      if (res.ok) c.put(e.request, res.clone());
      return res;
    } catch (err) {
      if (e.request.mode === 'navigate') { const idx = await c.match('index.html'); if (idx) return idx; }
      throw err;
    }
  })());
});
