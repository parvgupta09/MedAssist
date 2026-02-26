---
title: MedAssist API
sdk: docker
app_port: 7860
---

# MedAssist API

This Space runs the FastAPI backend from `api/app.py`.

## Secrets

Set `GROQ_API_KEY` in the Space Settings → Secrets (do not commit `.env`).

## Endpoints

- `GET /` health check
- `POST /session/new` create a chat session
- `POST /chat` send a message
