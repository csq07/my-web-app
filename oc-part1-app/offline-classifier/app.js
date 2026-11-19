/* Offline Object Classifier - all inference runs locally with TensorFlow.js */
(() => {
'use strict';
const $ = (id) => document.getElementById(id);
const CATS = ['electronics', 'food', 'animals', 'furniture', 'clothing', 'vehicles', 'person', 'other'];
const CAT_COLORS = { electronics: '#1e88e5', food: '#fb8c00', animals: '#43a047', furniture: '#8d6e63', clothing: '#d81b60',
  vehicles: '#5e35b1', person: '#00897b', other: '#757575' };
const abs = (p) => new URL(p, location.href).href;

const S = {
  lang: localStorage.getItem('lang') || 'zh',
  net: null, coco: null, cocoLoading: null, knn: null,
  stream: null, facing: 'environment', running: false, busy: false,
  mode: 'imagenet', thresh: 0.5, k: 5,
  classes: [],            // {id, name, cat, thumb}
  counts: Object.fromEntries(CATS.map(c => [c, 0])),
  log: [], streak: new Map(), present: new Map(), lastPreds: [], lastBoxes: [],
  snaps: [], activeClass: null,
};
window.APP_STATE = S; // exposed for debugging / automated tests

// ---------------- i18n ----------------
const t = (k) => (I18N[S.lang] && I18N[S.lang][k]) || I18N.en[k] || k;
const catName = (c) => t('cat_' + c);
function applyLang() {
  document.documentElement.lang = S.lang === 'zh' ? 'zh-CN' : 'en';
  document.querySelectorAll('[data-i18n]').forEach(el => { el.textContent = t(el.dataset.i18n); });
  document.querySelectorAll('[data-i18n-ph]').forEach(el => { el.placeholder = t(el.dataset.i18nPh); });
  document.title = t('title');
  $('langBtn').textContent = S.lang === 'zh' ? 'EN' : '中文';
  $('startBtn').textContent = S.stream ? t('stop') : t('start');
  $('newClassCat').innerHTML = CATS.map(c => `<option value="${c}">${t('classCategory')}: ${catName(c)}</option>`).join('');
  $('newClassCat').value = 'other';
  renderPreds(S.lastPreds); renderCatVerdict(S.lastCatProb); renderStats(); renderLog(); renderClasses(); renderSnaps();
  setStatus(S._statusKey || 'ready', S._statusExtra || '');
}
function setStatus(key, extra = '') {
  S._statusKey = key; S._statusExtra = extra;
  if (key === 'ready' && window.tf && S.net) extra = `${t('backend')}: ${tf.getBackend()}`;
  $('status').textContent = t(key) + (extra ? ' · ' + extra : '');
}
let toastTimer;
function toast(msg) {
  const el = $('toast'); el.textContent = msg; el.classList.add('show');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => el.classList.remove('show'), 2200);
}
const esc = (s) => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

// ---------------- models ----------------
async function init() {
  applyLang();
  setStatus('loadingTf');
  try { await tf.setBackend('webgl'); } catch (e) { /* fall through */ }
  await tf.ready();
  if (tf.getBackend() !== 'webgl') { await tf.setBackend('cpu'); }
  setStatus('loadingModel');
  S.net = await mobilenet.load({ version: 2, alpha: 1.0, modelUrl: abs('models/mobilenet/model.json') });
  S.knn = knnClassifier.create();
  // warm-up
  tf.tidy(() => S.net.infer(tf.zeros([224, 224, 3]), true));
  setStatus('ready', `${t('backend')}: ${tf.getBackend()}`);
  window.APP_READY = true;
  try { await loadFromDB(true); } catch (e) { console.warn(e); }
}
async function ensureCoco() {
  if (S.coco) return S.coco;
  if (!S.cocoLoading) {
    setStatus('loadingCoco');
    S.cocoLoading = cocoSsd.load({ base: 'lite_mobilenet_v2', modelUrl: abs('models/coco/model.json') }).then(m => {
      S.coco = m; setStatus('ready', `${t('backend')}: ${tf.getBackend()}`); return m;
    });
  }
  return S.cocoLoading;
}

// Run the current mode on an image/video/canvas element. Returns {preds:[{en,zh,cat,prob}], boxes:[...]}
async function predict(el) {
  if (!S.net) return { preds: [], boxes: [] };
  if (S.mode === 'coco') {
    const m = await ensureCoco();
    const dets = await m.detect(el, 20, 0.3);
    const boxes = dets.map(d => {
      const info = COCO_LABELS[d.class] || [d.class, 'other'];
      return { en: d.class, zh: info[0], cat: info[1], prob: d.score, bbox: d.bbox };
    });
    // de-duplicate by class for the top-3 list
    const seen = new Set(); const preds = [];
    for (const b of boxes.slice().sort((a, b2) => b2.prob - a.prob)) { if (!seen.has(b.en)) { seen.add(b.en); preds.push(b); } }
    return { preds: preds.slice(0, 3), boxes };
  }
  if (S.mode === 'custom') {
    if (S.knn.getNumClasses() === 0) return { preds: [], boxes: [], msg: t('noClasses') };
    const emb = tf.tidy(() => S.net.infer(el, true));
    const res = await S.knn.predictClass(emb, Math.max(1, S.k));
    emb.dispose();
    const preds = Object.entries(res.confidences).map(([id, p]) => {
      const c = S.classes.find(x => x.id === id) || { name: id, cat: 'other' };
      return { en: c.name, zh: c.name, cat: c.cat, prob: p, custom: true };
    }).sort((a, b) => b.prob - a.prob).slice(0, 3);
    return { preds, boxes: [] };
  }
  const probs = tf.tidy(() => tf.softmax(S.net.infer(el)));
  const data = await probs.data(); probs.dispose();
  const idx = Array.from(data.keys()).sort((a, b) => data[b] - data[a]).slice(0, 3);
  const preds = idx.map(i => {
    const [en, zh, cat] = IMAGENET_LABELS[i];
    return { en: en.split(',')[0], zh, cat, prob: data[i] };
  });
  // Category-level verdict: sum probability mass of all 1000 classes per broad category
  const catProb = {}; for (let i = 0; i < data.length; i++) { const c = IMAGENET_LABELS[i][2]; catProb[c] = (catProb[c] || 0) + data[i]; }
  return { preds, boxes: [], catProb };
}

// ---------------- rendering ----------------
const labelOf = (p) => S.lang === 'zh' ? p.zh : p.en;
const otherLabelOf = (p) => p.custom ? '' : (S.lang === 'zh' ? p.en : p.zh);
const badge = (cat) => `<span class="badge" style="background:${CAT_COLORS[cat] || '#757575'}">${esc(catName(cat))}</span>`;
function renderCatVerdict(cp) {
  const el = $('catVerdict'); if (!el) return;
  if (!cp) { el.innerHTML = ''; return; }
  const top = Object.entries(cp).sort((a, b) => b[1] - a[1]).slice(0, 3);
  el.innerHTML = `${esc(t('catVerdict'))}: ` + top.map(([c, p]) => `${badge(c)} ${(p * 100).toFixed(0)}%`).join(' &nbsp;');
}
function renderPreds(preds, msg) {
  const ol = $('preds');
  if (!preds || !preds.length) { ol.innerHTML = `<li class="hint">${esc(msg || t('noPred'))}</li>`; return; }
  ol.innerHTML = preds.map(p => `<li><div class="pred-top"><span><span class="pred-name">${esc(labelOf(p))}</span>` +
    `<span class="pred-en">${esc(otherLabelOf(p))}</span></span><span>${badge(p.cat)} ${(p.prob * 100).toFixed(1)}%</span></div>` +
    `<div class="bar"><div style="width:${(p.prob * 100).toFixed(1)}%;background:${CAT_COLORS[p.cat]}"></div></div></li>`).join('');
}
function renderStats() {
  $('catStats').innerHTML = CATS.map(c => `<div class="cat-stat" style="background:${CAT_COLORS[c]}">${esc(catName(c))}<b>${S.counts[c]}</b></div>`).join('');
}
function renderLog() {
  $('log').innerHTML = S.log.slice(-200).reverse().map(e => `<li><time>${e.time.toLocaleTimeString()}</time>${badge(e.cat)}` +
    `<span>${esc(S.lang === 'zh' ? e.zh : e.en)}</span><span class="pred-en">${(e.prob * 100).toFixed(0)}%</span></li>`).join('');
}
function drawBoxes(ctx, boxes, scale = 1) {
  ctx.lineWidth = Math.max(2, 3 * scale); ctx.font = `${Math.round(16 * scale)}px sans-serif`; ctx.textBaseline = 'top';
  for (const b of boxes) {
    const [x, y, w, h] = b.bbox; const col = CAT_COLORS[b.cat] || '#fff';
    ctx.strokeStyle = col; ctx.strokeRect(x, y, w, h);
    const txt = `${labelOf(b)} ${(b.prob * 100).toFixed(0)}%`;
    const tw = ctx.measureText(txt).width + 8;
    ctx.fillStyle = col; ctx.fillRect(x, Math.max(0, y - 22 * scale), tw, 22 * scale);
    ctx.fillStyle = '#fff'; ctx.fillText(txt, x + 4, Math.max(0, y - 22 * scale) + 3 * scale);
  }
}
function drawOverlay(boxes, srcW, srcH, bg) {
  const c = $('overlay'); const ctx = c.getContext('2d');
  if (c.width !== srcW || c.height !== srcH) { c.width = srcW; c.height = srcH; }
  ctx.clearRect(0, 0, c.width, c.height);
  if (bg) ctx.drawImage(bg, 0, 0, srcW, srcH);
  // When the video is mirrored (front camera), mirror box geometry instead of the canvas so label text stays readable
  const mirror = $('video').classList.contains('mirror') && !bg;
  const bx = mirror ? boxes.map(b => ({ ...b, bbox: [srcW - b.bbox[0] - b.bbox[2], b.bbox[1], b.bbox[2], b.bbox[3]] })) : boxes;
  drawBoxes(ctx, bx, srcW / 640);
}

// ---------------- logging / counting ----------------
function updateLog(preds, boxes) {
  // Labels above threshold this frame (classification: top-1, detection: every box)
  const items = (S.mode === 'coco' ? boxes : preds.slice(0, 1)).filter(p => p.prob >= S.thresh);
  const now = new Map(); items.forEach(p => { if (!now.has(p.en) || now.get(p.en).prob < p.prob) now.set(p.en, p); });
  let changed = false;
  for (const [key, p] of now) {
    const s = (S.streak.get(key) || 0) + 1; S.streak.set(key, s);
    if (s >= 3 && !S.present.has(key)) {
      S.present.set(key, 0);
      S.counts[p.cat] = (S.counts[p.cat] || 0) + 1;
      S.log.push({ time: new Date(), en: p.en, zh: p.zh, cat: p.cat, prob: p.prob, mode: S.mode });
      changed = true;
    }
    if (S.present.has(key)) S.present.set(key, 0);
  }
  for (const key of [...S.streak.keys()]) if (!now.has(key)) S.streak.delete(key);
  for (const [key, miss] of [...S.present]) {
    if (!now.has(key)) { if (miss + 1 >= 8) S.present.delete(key); else S.present.set(key, miss + 1); }
  }
  if (changed) { renderStats(); renderLog(); }
}

// ---------------- camera ----------------
async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { toast(t('cameraNeedsSecure')); return; }
  stopCamera(true);
  try {
    S.stream = await navigator.mediaDevices.getUserMedia({ audio: false,
      video: { facingMode: { ideal: S.facing }, width: { ideal: 640 }, height: { ideal: 480 } } });
  } catch (e) {
    try { S.stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false }); }
    catch (e2) { toast(t('cameraError') + e2.message); S.stream = null; return; }
  }
  const v = $('video'); v.srcObject = S.stream; await v.play().catch(() => {});
  const settings = S.stream.getVideoTracks()[0].getSettings ? S.stream.getVideoTracks()[0].getSettings() : {};
  const mirror = (settings.facingMode || S.facing) === 'user';
  v.classList.toggle('mirror', mirror);
  $('placeholder').classList.add('hidden');
  $('startBtn').textContent = t('stop');
  S.running = true; loop();
}
function stopCamera(silent) {
  S.running = false;
  if (S.stream) { S.stream.getTracks().forEach(tr => tr.stop()); S.stream = null; }
  const v = $('video'); v.srcObject = null;
  if (!silent) {
    $('placeholder').classList.remove('hidden');
    const c = $('overlay'); c.getContext('2d').clearRect(0, 0, c.width, c.height);
  }
  $('startBtn').textContent = t('start');
}
async function loop() {
  if (!S.running) return;
  const v = $('video');
  if (v.readyState >= 2 && v.videoWidth && S.net && !S.busy && !S.trainingBusy) {
    S.busy = true;
    try {
      const t0 = performance.now();
      const r = await predict(v);
      $('ms').textContent = Math.round(performance.now() - t0);
      S.lastPreds = r.preds; S.lastBoxes = r.boxes;
      renderPreds(r.preds, r.msg); S.lastCatProb = r.catProb; renderCatVerdict(r.catProb);
      drawOverlay(r.boxes, v.videoWidth, v.videoHeight);
      updateLog(r.preds, r.boxes);
    } catch (e) { console.error(e); }
    S.busy = false;
  }
  setTimeout(() => requestAnimationFrame(loop), 60);
}

// ---------------- snapshots ----------------
function snapshot() {
  const v = $('video'); let w, h; const c = document.createElement('canvas'); const ctx = c.getContext('2d');
  if (S.stream && v.videoWidth) { w = v.videoWidth; h = v.videoHeight; c.width = w; c.height = h; ctx.drawImage(v, 0, 0, w, h); drawBoxes(ctx, S.lastBoxes, w / 640); }
  else { const o = $('overlay'); if (!o.width) return; w = o.width; h = o.height; c.width = w; c.height = h; ctx.drawImage(o, 0, 0); }
  const p = S.lastPreds[0];
  if (p) {
    const fs = Math.max(14, Math.round(w / 32));
    ctx.fillStyle = 'rgba(0,0,0,.55)'; ctx.fillRect(0, h - fs * 1.8, w, fs * 1.8);
    ctx.fillStyle = '#fff'; ctx.font = `${fs}px sans-serif`; ctx.textBaseline = 'middle';
    ctx.fillText(`${labelOf(p)} · ${catName(p.cat)} · ${(p.prob * 100).toFixed(0)}%`, fs / 2, h - fs * 0.9);
  }
  S.snaps.unshift({ url: c.toDataURL('image/jpeg', 0.85), pred: p ? { ...p } : null, time: new Date() });
  if (S.snaps.length > 60) S.snaps.pop();
  renderSnaps(); toast(t('snapSaved'));
}
function renderSnaps() {
  $('snaps').innerHTML = S.snaps.map((s, i) => `<div class="snap"><img src="${s.url}" alt="">` +
    `<div>${s.pred ? esc(labelOf(s.pred)) + ' ' + badge(s.pred.cat) : ''}</div>` +
    `<a download="snapshot-${i}.jpg" href="${s.url}">${t('download')}</a></div>`).join('');
}

// ---------------- recognize uploaded image ----------------
async function fileToCanvas(file, maxSide = 640) {
  const bmp = await createImageBitmap(file);
  const sc = Math.min(1, maxSide / Math.max(bmp.width, bmp.height));
  const c = document.createElement('canvas'); c.width = Math.round(bmp.width * sc); c.height = Math.round(bmp.height * sc);
  c.getContext('2d').drawImage(bmp, 0, 0, c.width, c.height); bmp.close && bmp.close();
  return c;
}
async function recognizeFile(file) {
  if (!file || !S.net) return;
  stopCamera(true); $('placeholder').classList.add('hidden');
  $('video').classList.remove('mirror');
  const c = await fileToCanvas(file);
  const t0 = performance.now(); const r = await predict(c); $('ms').textContent = Math.round(performance.now() - t0);
  S.lastPreds = r.preds; S.lastBoxes = r.boxes;
  renderPreds(r.preds, r.msg); S.lastCatProb = r.catProb; renderCatVerdict(r.catProb); drawOverlay(r.boxes, c.width, c.height, c);
  const top = (S.mode === 'coco' ? r.boxes : r.preds.slice(0, 1)).filter(p => p.prob >= S.thresh);
  top.forEach(p => { S.counts[p.cat]++; S.log.push({ time: new Date(), en: p.en, zh: p.zh, cat: p.cat, prob: p.prob, mode: S.mode }); });
  renderStats(); renderLog();
}

// ---------------- custom training ----------------
const newId = () => 'c' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
function thumbOf(el) {
  const c = document.createElement('canvas'); c.width = c.height = 96;
  const w = el.videoWidth || el.width, h = el.videoHeight || el.height; const s = Math.min(w, h);
  c.getContext('2d').drawImage(el, (w - s) / 2, (h - s) / 2, s, s, 0, 0, 96, 96);
  return c.toDataURL('image/jpeg', 0.7);
}
function addClass(name, cat = 'other') {
  name = (name || '').trim();
  if (!name) { toast(t('enterName')); return null; }
  const ex = S.classes.find(c => c.name === name); if (ex) return ex;
  const c = { id: newId(), name, cat, thumb: '' }; S.classes.push(c); renderClasses(); return c;
}
function addExample(el, cls) {
  const emb = tf.tidy(() => S.net.infer(el, true));
  S.knn.addExample(emb, cls.id); emb.dispose();
  if (!cls.thumb) cls.thumb = thumbOf(el);
}
const countOf = (id) => (S.knn && S.knn.getClassExampleCount()[id]) || 0;
function renderClasses() {
  const el = $('classes'); if (!el) return;
  if (!S.classes.length) { el.innerHTML = `<p class="hint">${esc(t('noClasses'))}</p>`; return; }
  el.innerHTML = S.classes.map(c => `<div class="cls" data-id="${c.id}"><div class="cls-head">` +
    `<img src="${c.thumb || 'icons/icon-192.png'}" alt=""><div><div class="cls-name">${esc(c.name)}</div>` +
    `<div class="cls-count"><span class="cnt">${countOf(c.id)}</span> ${t('samples')}</div></div></div>` +
    `<div class="controls"><select data-act="cat">${CATS.map(k => `<option value="${k}" ${k === c.cat ? 'selected' : ''}>${esc(catName(k))}</option>`).join('')}</select></div>` +
    `<div class="controls"><button data-act="hold">${t('holdCapture')}</button><button data-act="burst">${t('burst')}</button>` +
    `<label class="btn"><span>${t('upload')}</span><input type="file" data-act="upload" accept="image/*" multiple hidden></label>` +
    `<button data-act="del" class="danger">${t('del')}</button></div></div>`).join('');
}
function refreshCounts() {
  document.querySelectorAll('.cls').forEach(d => {
    const c = S.classes.find(x => x.id === d.dataset.id); if (!c) return;
    d.querySelector('.cnt').textContent = countOf(c.id);
    const img = d.querySelector('img'); if (c.thumb && img.src !== c.thumb) img.src = c.thumb;
  });
}
function captureFromCamera(cls) {
  const v = $('video');
  if (!S.stream || !v.videoWidth) { toast(t('cameraOff')); return false; }
  addExample(v, cls); refreshCounts(); return true;
}
let holdTimer = null;
function startHold(cls, el) {
  stopHold(); if (!captureFromCamera(cls)) return;
  el.closest('.cls').classList.add('active');
  holdTimer = setInterval(() => captureFromCamera(cls), 120);
}
function stopHold() {
  if (holdTimer) { clearInterval(holdTimer); holdTimer = null; }
  document.querySelectorAll('.cls.active').forEach(e => e.classList.remove('active'));
}
async function burst(cls, n = 30) {
  for (let i = 0; i < n; i++) { if (!captureFromCamera(cls)) break; await new Promise(r => setTimeout(r, 100)); }
}
function showProgress(done, total) {
  const p = $('progress'); p.classList.toggle('hidden', done >= total);
  $('progressBar').style.width = (total ? (done / total * 100) : 0) + '%';
  $('progressText').textContent = `${t('processing')} ${done}/${total}`;
}
// Train from many files. pairs: [{file, cls}]
async function trainFiles(pairs) {
  if (!S.net || !pairs.length) return;
  S.trainingBusy = true; let done = 0; showProgress(0, pairs.length);
  for (const { file, cls } of pairs) {
    try { const c = await fileToCanvas(file, 320); addExample(c, cls); } catch (e) { console.warn('skip', file.name, e); }
    done++;
    if (done % 5 === 0 || done === pairs.length) { showProgress(done, pairs.length); refreshCounts(); await tf.nextFrame(); }
  }
  S.trainingBusy = false; refreshCounts(); toast(`${t('done')}: ${done}`);
}
function deleteClass(cls) {
  if (!confirm(t('confirmDelete'))) return;
  if (countOf(cls.id)) S.knn.clearClass(cls.id);
  S.classes = S.classes.filter(c => c !== cls); renderClasses();
}

// ---- persistence (IndexedDB + JSON export/import) ----
function idb() {
  return new Promise((res, rej) => {
    const r = indexedDB.open('offline-classifier', 1);
    r.onupgradeneeded = () => r.result.createObjectStore('models');
    r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error);
  });
}
async function idbOp(mode, fn) {
  const db = await idb();
  return new Promise((res, rej) => {
    const tx = db.transaction('models', mode); const st = tx.objectStore('models'); const req = fn(st);
    tx.oncomplete = () => res(req && req.result); tx.onerror = () => rej(tx.error);
  });
}
async function serialize() {
  const ds = S.knn.getClassifierDataset(); const out = {};
  for (const [id, tns] of Object.entries(ds)) out[id] = { shape: tns.shape, data: await tns.data() };
  return { format: 'offline-classifier-knn', version: 1, embedding: 'mobilenet_v2_1.0_224', classes: S.classes, dataset: out };
}
function deserialize(obj) {
  if (!obj || !obj.dataset || !obj.classes) throw new Error('bad model file');
  S.knn.clearAllClasses();
  const ds = {};
  for (const [id, v] of Object.entries(obj.dataset)) ds[id] = tf.tensor2d(v.data, v.shape);
  if (Object.keys(ds).length) S.knn.setClassifierDataset(ds);
  S.classes = obj.classes; renderClasses();
}
async function saveModel() {
  const data = await serialize(); // resolve tensors first; IDB transactions must not span awaits
  await idbOp('readwrite', st => st.put(data, 'custom'));
  toast(t('saved'));
}
async function loadFromDB(silent) {
  const obj = await idbOp('readonly', st => st.get('custom'));
  if (!obj) { if (!silent) toast(t('nothingSaved')); return; }
  deserialize(obj); if (!silent) toast(t('loaded'));
}
function f32ToB64(f32) {
  const u8 = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength); let s = '';
  for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
  return btoa(s);
}
function b64ToF32(b64) {
  const s = atob(b64); const u8 = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u8[i] = s.charCodeAt(i);
  return new Float32Array(u8.buffer);
}
async function exportModel() {
  const obj = await serialize();
  for (const v of Object.values(obj.dataset)) { v.b64 = f32ToB64(v.data); delete v.data; }
  const blob = new Blob([JSON.stringify(obj)], { type: 'application/json' });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = `custom-model-${new Date().toISOString().slice(0, 10)}.json`; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000); toast(t('exported'));
}
async function importModel(file) {
  const obj = JSON.parse(await file.text());
  for (const v of Object.values(obj.dataset || {})) { if (v.b64) { v.data = b64ToF32(v.b64); delete v.b64; } }
  deserialize(obj); toast(t('imported'));
}

// ---------------- wiring ----------------
function exportCsv() {
  const rows = [['time', 'label_en', 'label_zh', 'category', 'category_zh', 'confidence', 'mode']]
    .concat(S.log.map(e => [e.time.toISOString(), e.en, e.zh, e.cat, I18N.zh['cat_' + e.cat], e.prob.toFixed(4), e.mode]));
  const csv = '\ufeff' + rows.map(r => r.map(x => `"${String(x).replace(/"/g, '""')}"`).join(',')).join('\n');
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
  a.download = 'recognition-log.csv'; a.click();
}
function wire() {
  $('langBtn').onclick = () => { S.lang = S.lang === 'zh' ? 'en' : 'zh'; localStorage.setItem('lang', S.lang); applyLang(); };
  $('startBtn').onclick = () => S.stream ? stopCamera() : startCamera();
  $('switchBtn').onclick = () => { S.facing = S.facing === 'user' ? 'environment' : 'user'; if (S.stream) startCamera(); };
  $('snapBtn').onclick = snapshot;
  $('recogFile').onchange = (e) => { recognizeFile(e.target.files[0]); e.target.value = ''; };
  $('modeSel').onchange = (e) => { S.mode = e.target.value; S.streak.clear(); S.present.clear(); if (S.mode === 'coco') ensureCoco(); };
  $('thresh').oninput = (e) => { S.thresh = e.target.value / 100; $('threshVal').textContent = e.target.value + '%'; };
  $('kVal').onchange = (e) => { S.k = parseInt(e.target.value, 10) || 5; };
  $('clearLogBtn').onclick = () => { S.log = []; CATS.forEach(c => S.counts[c] = 0); S.present.clear(); renderStats(); renderLog(); };
  $('exportLogBtn').onclick = exportCsv;
  $('clearSnapBtn').onclick = () => { S.snaps = []; renderSnaps(); };
  $('addClassBtn').onclick = () => { if (addClass($('newClassName').value, $('newClassCat').value)) $('newClassName').value = ''; };
  $('newClassName').onkeydown = (e) => { if (e.key === 'Enter') $('addClassBtn').click(); };
  $('folderInput').onchange = async (e) => {
    const files = [...e.target.files].filter(f => /^image\//.test(f.type) || /\.(jpe?g|png|webp|bmp|gif)$/i.test(f.name));
    const pairs = [];
    for (const f of files) {
      const parts = (f.webkitRelativePath || f.name).split('/');
      const name = parts.length >= 3 ? parts[parts.length - 2] : (parts.length === 2 ? parts[0] : 'default');
      pairs.push({ file: f, cls: addClass(name, 'other') });
    }
    e.target.value = ''; await trainFiles(pairs);
  };
  $('saveModelBtn').onclick = () => saveModel().catch(e => toast(e.message));
  $('loadModelBtn').onclick = () => loadFromDB(false).catch(e => toast(e.message));
  $('exportModelBtn').onclick = () => exportModel().catch(e => toast(e.message));
  $('importModel').onchange = (e) => { if (e.target.files[0]) importModel(e.target.files[0]).catch(er => toast(er.message)); e.target.value = ''; };
  $('resetModelBtn').onclick = () => { if (!confirm(t('confirmReset'))) return; S.knn.clearAllClasses(); S.classes = []; renderClasses(); };

  const box = $('classes');
  const clsOf = (el) => S.classes.find(c => c.id === el.closest('.cls').dataset.id);
  box.addEventListener('pointerdown', (e) => { const b = e.target.closest('[data-act="hold"]'); if (b) { e.preventDefault(); startHold(clsOf(b), b); } });
  ['pointerup', 'pointerleave', 'pointercancel'].forEach(ev => box.addEventListener(ev, stopHold));
  window.addEventListener('pointerup', stopHold);
  box.addEventListener('contextmenu', (e) => { if (e.target.closest('[data-act="hold"]')) e.preventDefault(); });
  box.addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return; const c = clsOf(b);
    if (b.dataset.act === 'burst') burst(c); else if (b.dataset.act === 'del') deleteClass(c);
  });
  box.addEventListener('change', (e) => {
    const el = e.target; const c = clsOf(el);
    if (el.dataset.act === 'cat') c.cat = el.value;
    if (el.dataset.act === 'upload') { trainFiles([...el.files].map(f => ({ file: f, cls: c }))); el.value = ''; }
  });
}

// expose a few hooks for automated tests
window.APP = { predict, addClass, addExample, saveModel, loadFromDB, serialize, deserialize, startCamera, stopCamera, trainFiles };

if ('serviceWorker' in navigator && /^https?:$/.test(location.protocol)) {
  navigator.serviceWorker.register('sw.js').then(reg => {
    navigator.serviceWorker.addEventListener('message', (e) => { if (e.data === 'cached') toast(t('offlineReady')); });
  }).catch(e => console.warn('SW failed', e));
}
wire();
init().catch(e => { console.error(e); $('status').textContent = 'Error: ' + e.message; });
})();
