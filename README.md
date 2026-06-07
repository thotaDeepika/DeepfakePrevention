# 🛡️ SafeUpload: Transferable Identity Cloaking for Public Images

> **Research Prototype** | Defensive Privacy Tool | Ethical Use Only

SafeUpload applies subtle transferable adversarial perturbations to face images before public upload, reducing AI identity consistency while preserving human-recognizable appearance.

---

## 🚀 The Application Suite (v2.0)
SafeUpload has evolved from a Python research script into a comprehensive, production-ready privacy suite. It is composed of three main parts:

1. **The Backend Engine (Flask + Celery)**
   The core ML processing takes 30–60 seconds per image. To prevent the server from freezing, we use a robust asynchronous architecture. Flask handles the API requests, while a Celery Worker (backed by Redis) securely processes the heavy AI cloaking models in the background.

2. **The Premium Web Dashboard (React + Vite)**
   A beautiful, responsive web application serving as both a marketing landing page and the main upload dashboard. It features interactive Framer Motion animations and data visualization (Recharts) to explain the technology to non-technical users.

3. **The Smart Browser Extension (React + CRXJS)**
   The most convenient way to use SafeUpload. It features:
   - **Universal Popup Menu:** A sleek extension popup that lets you upload and cloak images directly from your browser toolbar. It uses a **Background Service Worker** so you can freely browse or switch tabs while the AI processes your image without interruption!
   - **Twitter (X) Direct Integration:** The extension features a custom site-specific injection script. When composing a tweet on Twitter, a "🛡️ SafeUpload" button is seamlessly injected right next to the native media button, allowing you to cloak images and drop them into your tweet with one click.
   - **Universal DOM Injection:** Automatically detects standard `<input type="file">` elements on any website and attaches a cloaking button nearby.

---

## ⚙️ How to Run the Servers

There are two ways to run the SafeUpload backend and web app: using **Docker** (Recommended) or **Manually**.

### Method 1: The Easy Way (Docker Compose)
If you have Docker Desktop installed, this handles all networking and dependencies automatically.

1. Clone the repository and navigate to the project root:
   ```bash
   cd safeupload
   ```
2. Build and start all services (Frontend, Backend, Celery Worker, Redis Broker):
   ```bash
   docker-compose up --build
   ```
3. Once running, open your browser:
   - **Marketing & Web App:** `http://localhost:3000`
   - **Backend API:** `http://localhost:7860`

*(To stop the server, press `Ctrl+C` or run `docker-compose down`).*

### Method 2: Manual Local Setup (Windows/Linux)
If you prefer running the servers manually, you will need 3 separate terminal windows:

**1. Start the Flask Backend:**
```bash
cd safeupload
pip install -r requirements.txt
python app/app.py
```

**2. Start the Celery Worker (Requires a local Redis server running on port 6379):**
```bash
cd safeupload
celery -A app.celery_worker.celery_app worker --loglevel=info -P solo
```

**3. Start the React Web App:**
```bash
cd safeupload/web
npm install
npm run dev
```

---

## 🧩 How to Install & Use the Browser Extension

The SafeUpload browser extension allows you to cloak your images seamlessly across the web.

**Installation:**
1. Build the extension:
   ```bash
   cd safeupload/extension-v2
   npm install
   npm run build
   ```
2. Open Google Chrome (or Edge/Brave) and go to `chrome://extensions/`.
3. Turn on **Developer mode** (top right corner).
4. Click **Load unpacked** (top left corner) and select the `safeupload/extension-v2/dist` folder.

**Usage:**
- **On Twitter (X):** Click the "What is happening?!" box to compose a tweet. You will see the purple `🛡️ SafeUpload` button injected directly into the tweet UI. Click it, select your photo, and it will automatically cloak and insert it into your draft!
- **On Other Sites:** Click the extension icon in your browser toolbar to open the Popup Menu. Select your image and click "Cloak Identity". You can safely close the popup while it processes. It will automatically download the cloaked image to your Downloads folder when finished.

---

## 📁 Project Structure

```
safeupload/
├── app/                         ← Flask API & Celery background workers
├── web/                         ← React + Vite Web Application & Dashboard
├── extension-v2/                ← React Browser Extension (Popup & Content Scripts)
├── docker-compose.yml           ← Unified Docker Orchestration
│
├── attacks/                     ← ML: MCA-based identity cloaking core
├── models/                      ← ML: FaceNet, ArcFace, CLIP models
├── losses/                      ← ML: Identity diversion loss functions
├── utils/                       ← ML: Data, preprocessing, face detection
└── SafeUpload_Colab.ipynb       ← Original Google Colab notebook
```

---

## 🧠 The Algorithm: MCA-Inspired Identity Cloaking

Adapted from the M-Attack-V2 paper ("Pushing the Frontier of Black-Box LVLM Attacks via Fine-Grained Detail Targeting"):

### Key Components

| Component | Purpose |
|-----------|---------|
| **Multi-Crop Alignment (MCA)** | Average gradients over K=8 random crops per step — reduces variance, improves transferability |
| **Patch Momentum (Adam)** | Stable directional optimization with gradient replay |
| **Ensemble Attack** | Joint optimization across FaceNet + ArcFace + CLIP-B/16 + CLIP-L/14 |
| **Directional Identity Diversion** | Push embeddings away from identity center toward synthetic pseudo-target |
| **Visual Quality Preservation** | Gaussian smoothing + edge-aware masking + TV regularization |

---

## ⚖️ Ethical Guidelines

✅ **Allowed**:
- Protecting your own photos before public upload
- Privacy research and evaluation
- Defensive identity cloaking

❌ **Not Allowed**:
- Targeting real people without consent
- Face swapping or impersonation
- Creating malicious identity manipulation tools

---

## Citation

This work adapts techniques from:
> Zhao et al., "Pushing the Frontier of Black-Box LVLM Attacks via Fine-Grained Detail Targeting", MBZUAI VILA Lab, 2026.
