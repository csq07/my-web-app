# 离线物品识别分类器 / Offline Object Classifier

用摄像头在**本机离线**识别物品并自动归类（电子产品、食物、动物、家具、衣物、交通工具、人物、其他），界面支持**中文 / English** 一键切换（默认中文）。所有模型（TensorFlow.js + MobileNet v2 + COCO-SSD + KNN）都已放在本文件夹中，运行时**不需要联网**，图像不会上传。

Recognize objects with your camera **fully offline** and group them into broad categories. UI switches between **Chinese / English**. All libraries and model weights are bundled in this folder; nothing is downloaded at runtime and no image leaves the device.

---

## 中文说明

### 功能
- **开启/关闭摄像头、切换前/后摄像头**，实时显示前 3 个识别结果和置信度。
- **三种识别模式**：
  1. 通用分类：MobileNet v2，ImageNet 1000 类（全部带中文名称）。
  2. 目标检测：COCO-SSD，80 类，画出边框（带中文名称），可同时识别多个物品。
  3. 自定义模型：你自己训练的类别（KNN）。
- **大类归类 + 统计**：每个识别结果映射到大类，按大类计数，并有识别记录（可导出 CSV）。“记录阈值”控制多高的置信度才计入。
- **拍照快照**：带识别标签，可下载。
- **识别图片**：不开摄像头也能选择一张图片识别。
- **自定义训练（适合大量数据）**：
  - 添加类别（填名称、选所属大类）。
  - “按住采集”：按住按钮连续采集摄像头画面；“连拍 30 张”。
  - “上传图片”：一次选择大量图片作为样本。
  - “导入文件夹”：选择一个文件夹，**每个子文件夹名 = 类别名**，里面的图片自动训练（可上千张）。
  - 保存到本机（IndexedDB）/ 从本机加载（打开页面时自动加载）/ 导出 JSON / 导入 JSON（可在其他设备上使用）。
- **PWA**：首次打开后所有文件被缓存，以后断网也能打开；手机上可“添加到主屏幕”像 App 一样使用。

### 在电脑上运行
摄像头只能在 `localhost` 或 `https` 下使用，所以不能直接双击 `index.html`，需要启动一个本地小服务器（需安装 Python 3）：
- Windows：双击 `start-windows.bat`
- Mac / Linux：运行 `./start-mac-linux.sh`
- 或手动：在本文件夹打开终端，运行 `python3 -m http.server 8000`（Windows 用 `python -m http.server 8000`），然后浏览器打开 <http://localhost:8000>
- 推荐 Chrome / Edge / Safari 最新版。点击“开启摄像头”并允许权限。

### 在手机上运行
手机浏览器要求 **HTTPS** 才能使用摄像头（localhost 除外）。任选一种方式：
1. **（推荐）放到任意 HTTPS 静态网站**：把整个文件夹上传到 GitHub Pages / Netlify / Vercel / 自己的服务器，手机打开一次网址，等提示“已缓存，可离线使用”，然后在浏览器菜单中“添加到主屏幕”。之后**断网也能用**。
2. **局域网自签名 HTTPS**：电脑和手机连同一个 Wi‑Fi，电脑运行 `python3 start-server.py --https`（需要电脑装有 `openssl`），手机打开屏幕上显示的 `https://电脑IP:8443`，接受“证书不安全”提示即可使用。注意：自签名证书下浏览器通常不允许离线缓存（PWA），每次需电脑开着服务器。
3. **Android + USB 数据线**：手机开启 USB 调试，电脑运行 `python3 -m http.server 8000` 和 `adb reverse tcp:8000 tcp:8000`，手机 Chrome 打开 `http://localhost:8000`（视为安全环境，PWA 离线缓存也可用）。

### 小贴士
- 自定义训练：每个类别建议 30–300 张不同角度、不同背景的样本；类别越多越需要更多样本。K 值（默认 5）可调。
- 自定义模型存的是 MobileNet 特征（每张约 5KB），所以上千张图片也很轻量。
- 首次加载约 33MB（模型文件），之后从缓存秒开。如更新了文件，请把 `sw.js` 里的 `VERSION` 改一下以刷新缓存。

---

## English

### Features
- Start/stop camera, switch front/back camera, live top‑3 predictions with confidence.
- Three modes: **MobileNet v2** (ImageNet 1000 classes, all with Chinese names), **COCO‑SSD** object detection (80 classes, boxes, Chinese names), and **Custom KNN** model.
- Every prediction is mapped to a broad category (Electronics, Food, Animals, Furniture, Clothing, Vehicles, People, Other) with running counts and a log (CSV export). The *log threshold* slider sets the minimum confidence that is counted.
- Snapshots with labels (downloadable). Recognize a picked image without the camera.
- **Custom training for large datasets**: add classes, *hold to capture* or *burst 30* from the camera, bulk-upload images, or *import a folder* where each sub‑folder name is a class (thousands of images are fine). Save/load on device (IndexedDB, auto-loaded on start) and export/import as a JSON file to move the model between devices.
- **PWA**: after the first load everything is cached; works with no network and can be installed to the home screen.

### Run on a PC
Cameras only work on `localhost` or `https`, so don't open `index.html` directly — start a tiny local server (Python 3):
- Windows: double‑click `start-windows.bat`; macOS/Linux: `./start-mac-linux.sh`
- Or: `python3 -m http.server 8000` in this folder, then open <http://localhost:8000>, click **Start camera** and allow access.

### Run on a phone
Mobile browsers require **HTTPS** for the camera (except `localhost`):
1. **Recommended:** upload this folder to any HTTPS static host (GitHub Pages, Netlify, Vercel, your server). Open it once on the phone, wait for “Cached for offline use”, then *Add to Home Screen*. It then works fully offline.
2. **LAN with self‑signed HTTPS:** same Wi‑Fi; run `python3 start-server.py --https` on the PC (needs `openssl`), open the printed `https://<PC-IP>:8443` on the phone and accept the certificate warning. (Browsers usually won't enable offline caching with a self‑signed cert.)
3. **Android over USB:** enable USB debugging, run `python3 -m http.server 8000` and `adb reverse tcp:8000 tcp:8000`, then open `http://localhost:8000` in Chrome on the phone.

### Files
`index.html`, `style.css`, `app.js`, `i18n.js` (UI strings), `labels.js` (ImageNet + COCO labels in EN/ZH with category mapping), `sw.js` (service worker), `manifest.json`, `lib/` (TensorFlow.js 4.x, mobilenet, knn-classifier, coco-ssd), `models/` (MobileNet v2 1.0 224 and SSDLite MobileNet v2 weights), `start-server.py`, `start-*.{bat,sh}`.

### Credits / licenses
TensorFlow.js and tfjs-models (Apache 2.0). Chinese ImageNet labels from OFA-Sys/Chinese-CLIP (MIT).
