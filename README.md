# 🛡️ SafeUpload: Transferable Identity Cloaking for Public Images

> **Research Prototype** | Defensive Privacy Tool | Ethical Use Only

SafeUpload applies subtle transferable adversarial perturbations to face images before public upload, reducing AI identity consistency while preserving human-recognizable appearance.

---

## 🚀 What's New in v2.0
SafeUpload has evolved from a Python research script into a full-stack, production-ready privacy suite:
- **Premium Web Dashboard:** A responsive React + Vite web application with interactive data visualization (Recharts).
- **Asynchronous Backend:** Flask + Celery + Redis architecture for handling heavy GPU cloaking processes without freezing the server.
- **Browser Extension:** A React-based Chrome extension that automatically injects a "Cloak with SafeUpload" button into social media file uploads.
- **Dockerized:** Easily deploy the entire application suite locally via Docker Compose.

---

## ⚙️ How to Run

There are two ways to run the SafeUpload backend and web app: using **Docker** (Recommended) or **Manually**.

### Method 1: The Easy Way (Docker Compose)
If you have Docker Desktop installed, this is the easiest way to run the entire stack.

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

## 🧩 How to Install the Browser Extension

The SafeUpload browser extension allows you to cloak your images directly from Twitter, Facebook, or any other website.

1. Build the extension:
   ```bash
   cd safeupload/extension-v2
   npm install
   npm run build
   ```
2. Open Google Chrome (or Edge/Brave) and go to `chrome://extensions/`.
3. Turn on **Developer mode** (top right corner).
4. Click **Load unpacked** (top left corner).
5. Select the `safeupload/extension-v2/dist` folder.
6. The extension is now active! Go to any website with an upload form, and you will see a `🛡️ Cloak with SafeUpload` button magically appear.

---

## 📁 Project Structure

```
safeupload/
├── app/                         ← Flask API & Celery background workers
├── web/                         ← React + Vite Web Application & Dashboard
├── extension-v2/                ← React-based Browser Extension & Content Scripts
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

### Pseudo-Target Generation

No real people are targeted. Synthetic targets are generated via:
- **Orthogonal PCA**: find direction orthogonal to identity subspace
- **Gram-Schmidt orthogonalization** against identity center
- Ensures face-space semantics are preserved while identity is diverted

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
