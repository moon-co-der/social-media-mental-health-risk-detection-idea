---
title: Social Media NLP Inference API
emoji: 🧪
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---

# Python inference API

Docker-based FastAPI service for the research frontend. CPU Basic Spaces are documented as 2 vCPU and 16 GB RAM, but current HF documentation says creating a Docker/Gradio Space requires a paid plan; free CPU availability should not be assumed. Free CPU Spaces sleep after inactivity. Check current account and plan terms before selecting a host. Copy the contents of `backend/` into a separate Docker Space repository root. The local E2 checkpoint is excluded from Git; provide it through protected mounted storage at `/app/models/e2/best_model` or set `E2_MODEL_PATH` to an authorized location. GoEmotions downloads at the pinned revision on first inference and needs outbound Hugging Face access or a populated cache. Set `HF_TOKEN` only as a Space secret for private weights, and restrict `ALLOWED_ORIGINS` to the actual frontend origin. Review E2 model and training-data redistribution rights before uploading weights. See the project README for local development and caveats.
