# Hosting Guide — FloorPlan Studio Inference API

This covers deploying the `inference-api/` service (FastAPI + YOLO11n) so your
Vercel frontend can call it directly, without Roboflow credits and without
your laptop needing to stay on.

Pick **one** platform below. Render and Railway are the easiest for a first
deployment; Hugging Face Spaces is free-friendly but sleeps when idle; Cloud
Run is the most scalable but has the steepest setup.

---

## 0. Get your model reachable by the server

Your `best.pt` lives in Google Drive. The API needs to read it from disk at
startup, so pick one option:

- **Bake it into the repo/image** (simplest if it's under ~100MB): copy
  `best.pt` into `inference-api/` and uncomment the `COPY best.pt .` line in
  the Dockerfile, or commit it to the repo the platform builds from.
- **Download it at startup** (better for larger files or private repos): set
  `MODEL_URL` to a direct-download link and leave `best.pt` out of the repo.
  For Google Drive, use a link in the form:
  `https://drive.google.com/uc?export=download&id=YOUR_FILE_ID`
  (make sure the file's sharing setting is "Anyone with the link").

Either way, `app.py` already handles both cases — it downloads only if the
file isn't already present.

---

## Option A — Render

1. Push `inference-api/` (with `app.py`, `requirements.txt`, `Dockerfile`) to
   a GitHub repo.
2. On [render.com](https://render.com), click **New → Web Service**, connect
   the repo.
3. Environment: choose **Docker** (Render will detect the Dockerfile
   automatically).
4. Set environment variables under **Environment**:
   - `ALLOWED_ORIGINS` = your Vercel URL, e.g. `https://floorplan-studio.vercel.app`
   - `MODEL_URL` = your Google Drive direct-download link (skip if you baked
     in the model)
5. Instance type: start with the free tier to test, upgrade to a paid plan
   for production (free tier spins down after inactivity, causing a ~30–60s
   cold start on the next request).
6. Deploy. Render gives you a URL like `https://floorplan-api.onrender.com`.
7. Test: `curl https://floorplan-api.onrender.com/health`

## Option B — Railway

1. Push `inference-api/` to GitHub.
2. On [railway.app](https://railway.app), **New Project → Deploy from GitHub
   repo**, select it. Railway auto-detects the Dockerfile.
3. Under **Variables**, add `ALLOWED_ORIGINS` and `MODEL_URL` as above.
4. Railway auto-assigns a port via `$PORT` — `app.py` already reads
   `PORT` from the environment, so no change needed.
5. Under **Settings → Networking**, click **Generate Domain** to get a public
   URL.
6. Test the `/health` endpoint the same way as above.

## Option C — Hugging Face Spaces

Good free option, but Spaces sleep after inactivity (cold start delay) unless
you pay for an "always on" upgrade.

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space),
   SDK = **Docker**.
2. Push `app.py`, `requirements.txt`, and `Dockerfile` to the Space's repo
   (Spaces work like a git repo).
3. Spaces expect the app to listen on port `7860` by default — either set
   `ENV PORT=7860` in the Dockerfile, or add a Space secret `PORT=7860`.
4. Add secrets (Settings → Repository secrets): `ALLOWED_ORIGINS`, `MODEL_URL`.
5. The Space builds automatically; your API is reachable at
   `https://YOUR_USERNAME-YOUR_SPACE.hf.space`.

## Option D — Google Cloud Run

Most scalable, pay-per-request, but requires the `gcloud` CLI and a GCP
project with billing enabled.

```bash
cd inference-api
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Build and push the container
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/floorplan-inference

# Deploy
gcloud run deploy floorplan-inference \
  --image gcr.io/YOUR_PROJECT_ID/floorplan-inference \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --set-env-vars ALLOWED_ORIGINS=https://floorplan-studio.vercel.app,MODEL_URL=YOUR_MODEL_URL
```

Cloud Run gives you a URL like
`https://floorplan-inference-xxxxx-uc.a.run.app`. Note `--memory 2Gi`: YOLO
inference needs more than Cloud Run's 512MB default, so don't skip that flag.

---

## 1. Connect the Vercel frontend

In your Vercel project, add an environment variable (Project Settings →
Environment Variables):

```
NEXT_PUBLIC_INFERENCE_API_URL=https://<your-chosen-platform-url>
```

Then call it from wherever the frontend currently calls Roboflow, e.g.:

```javascript
async function detectFurniture(imageFile) {
  const formData = new FormData();
  formData.append("file", imageFile);

  const response = await fetch(
    `${process.env.NEXT_PUBLIC_INFERENCE_API_URL}/predict`,
    { method: "POST", body: formData }
  );

  if (!response.ok) {
    throw new Error(`Inference API error: ${response.status}`);
  }

  const data = await response.json();
  return data.detections; // [{ label, confidence, bbox }, ...]
}
```

Redeploy the Vercel project after adding the env var so it picks it up.

---

## 2. Testing checklist

- [ ] `GET /health` returns `{"status": "ok", "model_loaded": true}`
- [ ] `POST /predict` with a sample room photo returns a non-empty
      `detections` array with sensible labels
- [ ] CORS: open the deployed Vercel site in a browser, upload an image, and
      confirm the request to the inference API succeeds (check the Network
      tab — no CORS error)
- [ ] Cold start: if using a free tier, time the first request after a period
      of inactivity so you know what delay to expect / display a loading
      state for
- [ ] Load: try 2–3 uploads in a row to confirm the model doesn't reload on
      every request (it shouldn't — it's loaded once at startup)

---

## 3. Common pitfalls

- **"Model not found and MODEL_URL is not set"** — you deployed without
  baking in `best.pt` and without setting `MODEL_URL`. Pick one of the two.
- **CORS errors in the browser console** — `ALLOWED_ORIGINS` doesn't include
  your exact Vercel domain (must match scheme + host exactly, no trailing
  slash).
- **Out-of-memory crashes** — YOLO11n plus `torch` needs more RAM than the
  smallest free tier on some platforms offers; bump the instance/memory size
  a notch if the service crashes on the first prediction.
- **Slow first request** — normal on free/sleeping tiers; the model is
  loaded lazily on cold start. Consider a paid "always-on" tier for a smoother
  demo, or add a loading spinner on the frontend.
