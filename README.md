# 🏥 MedAI Diagnostics — AI Medical Image Analyzer

> CNN-based medical image analysis for Chest X-Ray, Brain MRI & CT Scan with
> Grad-CAM heatmaps, PDF reports, Neon PostgreSQL, and login authentication.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?style=flat-square)
![PostgreSQL](https://img.shields.io/badge/Neon-PostgreSQL-green?style=flat-square)

---

## 🚀 Deploy to Streamlit Cloud

```bash
git init && git add . && git commit -m "🏥 MedAI"
git remote add origin https://github.com/YOUR_USERNAME/medical-analyzer.git
git push -u origin main
# share.streamlit.io → New app → app.py → Deploy
```

**Demo login:** `demo_doctor` / `Demo@1234`

---

## ✨ Features

| Page | Features |
|---|---|
| 🔐 **Login** | Signup / sign-in with bcrypt · demo account auto-created |
| 📊 **Dashboard** | Patient count · scan stats · severity breakdown · recent scans |
| 🔬 **Analyze** | Upload image · CLAHE enhancement · AI prediction · Grad-CAM heatmap · save to record · PDF |
| 👥 **Patients** | Add · search · view history · scan timeline |
| 📋 **History** | Filter by type/severity · CSV export |
| 📄 **Reports** | Generate doctor-style PDF with heatmap + probability table |
| ⚙️ **Settings** | Profile · DB config · model status · dataset guide |

---

## 📁 Project Structure

```
medical-analyzer/
│
├── app.py                  ← Main Streamlit app (7 pages)
├── model.py                ← CNN inference + Grad-CAM + demo fallback
├── train_brain.py          ← EfficientNet-B0 trainer for Brain MRI
├── train_chest.py          ← DenseNet121 trainer for Chest X-Ray
├── generate_synthetic.py   ← Create synthetic training data (no download needed)
├── download_data.py        ← Kaggle downloader (small datasets only)
├── database.py             ← Neon PostgreSQL + in-memory fallback
├── auth.py                 ← Login/signup with bcrypt
├── report.py               ← ReportLab PDF report generator
├── utils.py                ← Image processing · CSS · UI helpers
├── requirements.txt
├── README.md
│
├── .streamlit/
│   ├── config.toml         ← Dark medical theme
│   └── secrets.toml.example
│
├── data/
│   └── sample_images/      ← 12 synthetic demo images (pre-included)
│
└── weights/                ← Put .pth files here after training
    ├── brain_mri.pth       ← from: python train_brain.py
    └── chest_xray.pth      ← from: python train_chest.py
```

---

## 🤖 Models

| Model | Architecture | Dataset | Classes | Size |
|---|---|---|---|---|
| **ChestNet** | DenseNet121 | Chest X-ray Pneumonia (Kaggle) | Normal / Pneumonia | ~1.2 GB |
| **BrainNet** | EfficientNet-B0 | Brain Tumor MRI (Kaggle) | Glioma / Meningioma / No Tumor / Pituitary | ~150 MB |
| **CTNet** | ResNet50 | Demo only | Normal / Hemorrhage / Infarction / Mass / Edema | — |

---

## ⚙️ Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/medical-analyzer.git
cd medical-analyzer
pip install -r requirements.txt
streamlit run app.py          # → http://localhost:8501
```

---

## 📥 Get Training Data

### Option A — Kaggle (real data)
```bash
# Setup credentials (one-time):
# kaggle.com → Profile → Settings → API → Create Token → kaggle.json
mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

python download_data.py --dataset brain    # ~150 MB
python download_data.py --dataset chest    # ~1.2 GB (NOT the 45 GB NIH set)
python download_data.py --notebook brain   # reference notebook (99% accuracy)
```

### Option B — Synthetic (no download)
```bash
python generate_synthetic.py               # generates all datasets instantly
python generate_synthetic.py --samples 300 # 300 images/class
```

### Train models
```bash
python train_brain.py --epochs 20          # → weights/brain_mri.pth  (~95-99% acc)
python train_chest.py --epochs 10          # → weights/chest_xray.pth (~90-95% acc)
```

The app **auto-detects** weights on startup and switches from Demo → Real inference.

---

## 🗄️ Neon PostgreSQL Setup

1. Free account at **[neon.tech](https://neon.tech)** → new project → copy connection string
2. Add to Streamlit secrets:
   ```toml
   DATABASE_URL = "postgresql://neondb_owner:PASSWORD@ep-XXXX.neon.tech/neondb?sslmode=require"
   ```
3. Tables auto-create on first launch (`users`, `patients`, `scans`, `reports`)

Without DB configured, the app runs in **in-memory demo mode** (data resets on restart).

---

## 📐 Architecture

```
Upload Image
    ↓
CLAHE Enhancement (OpenCV)
    ↓
CNN Inference (PyTorch)
├── ChestNet  → Normal / Pneumonia
├── BrainNet  → Glioma / Meningioma / No Tumor / Pituitary
└── CTNet     → 5-class CT
    ↓
Grad-CAM Heatmap
    ↓
Severity Assessment
    ↓
Findings Text Generation
    ↓
PDF Report (ReportLab) + DB Save (Neon PostgreSQL)
```

---

## 📝 Resume Points

**MedAI Diagnostics — AI Medical Image Analyzer** · PyTorch, Streamlit, PostgreSQL

- Built CNN classifier (DenseNet121 for chest X-ray, EfficientNet-B0 for brain MRI) with transfer learning from ImageNet pretrained weights
- Implemented Grad-CAM visual explainability showing regions driving AI diagnosis
- Generated synthetic medical training data (chest X-ray, brain MRI) using procedural image synthesis with augmentation
- Designed doctor-style PDF reports using ReportLab with patient info, heatmap, probability tables, clinical recommendations
- Integrated Neon serverless PostgreSQL for persistent patient/scan/report storage with graceful in-memory fallback
- Built login/signup auth with bcrypt password hashing

---

## ⚠️ Disclaimer

Educational and research use only. All AI findings require validation by qualified medical professionals.

---

## 📜 License

MIT
