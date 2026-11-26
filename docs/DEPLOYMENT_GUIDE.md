# Cloud Deployment Guide (Render + Vercel)

This repository includes container and blueprint configurations so you can deploy all three layers and share a single public Vercel URL:

- **Frontend (`frontend/`)** → Vercel (`frontend/vercel.json`)
- **Spring Boot Gateway (`backend/`)** → Render (`backend/Dockerfile`)
- **Python AI Microservice (`ai_service/`)** → Render (`ai_service/Dockerfile`) or Hugging Face Spaces (Docker)

---

## 1. Deploy the Backend Services (Render Blueprint)

1. Push your repository to GitHub.
2. In **Render**, click **New → Blueprint** and select this repository (uses `render.yaml`).
3. Render will create two Docker web services:
   - `bafv-pcta-ai` (Python FastAPI + CPU-only PyTorch, health check `/health`)
   - `bafv-pcta-gateway` (Java 21 Spring Boot, health check `/health`)
4. Once `bafv-pcta-ai` deploys, copy its public URL (e.g. `https://bafv-pcta-ai.onrender.com`) and set it as `PYTHON_AI_URL` on `bafv-pcta-gateway` in the Render dashboard.
5. Verify both `/health` endpoints return `"status": "healthy"`.

> **Free-Tier RAM Note:** Render's free tier provides 512 MB RAM. `ai_service/Dockerfile` uses CPU-only PyTorch wheels to minimize footprint, and `backend/Dockerfile` caps JVM heap to `-Xmx256m`. If PyTorch + Librosa still hits memory limits on Render Free, deploy `ai_service/Dockerfile` on **Hugging Face Spaces (Docker SDK, free 16 GB RAM CPU)** and point `PYTHON_AI_URL` on Render's `bafv-pcta-gateway` to that Space URL.

---

## 2. Deploy the React Portal (Vercel)

1. In **Vercel**, click **Add New → Project** and import your GitHub repository.
2. Set **Root Directory** to `frontend`.
3. Add Environment Variable:
   - `VITE_GATEWAY_URL` = `https://your-gateway-url.onrender.com` (no trailing slash)
4. Click **Deploy**.

Once deployed, opening your single Vercel URL connects directly to the live Spring Boot gateway and Python AI engine.
