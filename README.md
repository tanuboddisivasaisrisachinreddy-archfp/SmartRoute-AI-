# SmartRoute AI – Reinforcement Learning for Dynamic Route Optimization (Hyderabad)

SmartRoute AI routes trips across the **real OpenStreetMap road network of Hyderabad, Telangana** and compares three approaches on identical conditions:

| | Algorithm | Map colour |
|---|---|---|
| 1 | **Dijkstra** – classical shortest path on the departure-time traffic snapshot (NetworkX) | blue |
| 2 | **Q-Learning** – goal-conditioned tabular RL (ε-greedy, α, γ, Q-table per destination) | green |
| 3 | **Deep Q-Network** – PyTorch Double-DQN with replay buffer, target network and action masking | purple |

A time-dependent traffic simulator (scenarios, incidents, surges, recovery) drives the environment. Optional live traffic comes from the TomTom Flow API. Without a key, every screen says **"Simulated Traffic – Not Live Traffic"**.

Stack: React + Vite + TypeScript + Tailwind + React Leaflet + Recharts (frontend), FastAPI + NetworkX + OSMnx + NumPy + PyTorch (backend/ML), MongoDB Atlas (storage), Render (hosting).

---

## Contents
1. [Architecture](#architecture) · 2. [Features](#features) · 3. [Datasets](#datasets-and-citations) · 4. [Install on macOS](#installation-macos) · 5. [VS Code](#vs-code-setup) · 6. [Run backend](#backend-startup) · 7. [Run frontend](#frontend-startup) · 8. [Train](#model-training) · 9. [Evaluate](#model-evaluation) · 10. [Tests](#tests) · 11. [Environment variables](#environment-variables) · 12. [GitHub](#push-to-github) · 13. [Render deployment](#render-deployment) · 14. [API](#api-documentation) · 15. [Troubleshooting](#troubleshooting)

## Architecture

```
 Browser (React, Leaflet, Recharts)              Render Static Site
        │  axios, VITE_API_URL
        ▼
 FastAPI (backend/app)                            Render Web Service
   api/routes.py ── services/engine.py ──┬── routing/dijkstra.py (NetworkX)
        │                                ├── reinforcement_learning/q_learning  (tabular Q)
        │                                ├── reinforcement_learning/dqn         (Q-network, NumPy inference)
        │                                └── reinforcement_learning/environment (graph, traffic, MDP)
        ├── services/training_jobs.py ── subprocess: scripts/train_*.py (never inside a request)
        ├── traffic/tomtom.py (optional live)        services/geocoding.py (landmarks + Nominatim)
        └── database/store.py ── MongoDB Atlas  (fallback: data/runtime/*.jsonl)

 Offline (your Mac): download_data → preprocess_data → validate_data → train_q_learning / train_dqn → evaluate_models
```

```
SmartRoute-AI/
├── frontend/                 React app (pages/, components/, map/, charts/, services/, hooks/, layouts/)
├── backend/app/              FastAPI: api/, core/, schemas/, services/, routing/, traffic/, database/
├── reinforcement_learning/   environment/, q_learning/, dqn/, training/, evaluation/, checkpoints/
├── data/                     raw/, processed/, external/, README.md (dataset documentation)
├── scripts/                  download / preprocess / validate / train / evaluate / package / deploy / smoke test
├── tests/                    pytest suite (+ frontend/src/services/api.test.ts with Vitest)
├── docs/                     rl_design.md, traffic_model.md, api.md
├── render.yaml  .env.example  requirements-ml.txt  requirements-dev.txt
```

## Features
* **Route optimizer**: search Hyderabad places (19 built-in landmarks + OSM Nominatim), click the map to set start/destination, swap them, and pick fastest, shortest or balanced plus a traffic scenario and an optional mid-trip event. All three routes are drawn on road-following geometry. Each one lists distance, time, congestion, segments, fuel cost (₹), score and compute time. Dijkstra alternatives, a traffic layer and an explanation of the best route are included.
* **Traffic simulation**: change scenario and intensity, set or advance the clock, place incidents on the map or at random, clear them, trigger a sudden increase, and reset. A live congestion map and a history chart update as you go. TomTom live mode turns on when a key is configured.
* **RL training**: start or stop background jobs with configurable episodes, learning rate, discount, ε, batch, buffer and seed. You can watch the episode, average reward, episode reward, ε, loss and completion rate live, and see saved training curves.
* **Algorithm comparison**: batch experiments on landmark pairs × scenarios × events. Tables and charts show success rate, time, distance, congestion exposure, reward and compute time, and the page says where RL is worse.
* **Analytics / Overview**: all statistics are computed from stored comparisons, experiments and training logs. Nothing is hard-coded.
* **Datasets**: provenance, licence and coverage for every source, plus graph statistics and validation results. METR-LA/PEMS-BAY time-of-day profiles are clearly labelled as US data.
* Dark/light theme, responsive layout, loading, empty, error and success states throughout.

## Datasets and citations
Full documentation: [`data/README.md`](data/README.md).

| Dataset | Covers Hyderabad? | Use |
|---|---|---|
| OpenStreetMap via OSMnx (ODbL) | **yes** | routing graph |
| TomTom Traffic Flow API (optional) | **yes** | live congestion samples |
| METR-LA (Zenodo 5146275, DCRNN) | no – Los Angeles | benchmark profiles |
| PEMS-BAY (Zenodo 4263971 / 4264445) | no – SF Bay Area | benchmark profiles |
| india-geodata urban roads | unverified | inventoried, not integrated |

Citations: © OpenStreetMap contributors. Boeing, G. (2017) *OSMnx*. Li, Y., Yu, R., Shahabi, C., Liu, Y. (2018) *Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting*, ICLR. Mnih et al. (2015) *Human-level control through deep RL*, Nature. van Hasselt et al. (2016) *Double Q-learning*, AAAI. Ng, Harada, Russell (1999) *Policy invariance under reward transformations*, ICML.

**Limitations**: traffic is simulated unless TomTom is configured. Hotspots and the daily profile are assumptions. The graph keeps arterial roads only (motorway to secondary). Tabular Q-learning needs a table per destination, so other destinations are trained on demand with a small budget. The METR-LA/PEMS-BAY profiles describe US freeways.

## Installation (macOS)

```bash
# 1. Tools (Homebrew: https://brew.sh)
brew install python@3.11 node git

# 2. Project folder
cd ~/"ML PROJ"/SmartRoute-AI

# 3. Python virtual environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt -r requirements-ml.txt -r requirements-dev.txt

# 4. Config
cp .env.example .env                    # edit MONGODB_URI / TOMTOM_API_KEY if you have them
cp frontend/.env.example frontend/.env  # VITE_API_URL=http://localhost:8000

# 5. Frontend packages
cd frontend && npm install && cd ..
```
(With Anaconda instead: `conda create -n smartroute python=3.11 -y && conda activate smartroute`, then the same `pip install` line.)

### Download and prepare the data
```bash
source .venv/bin/activate
python scripts/download_data.py --osm --landmarks   # Hyderabad road network (2–5 min) + geocoding
python scripts/download_data.py --traffic           # optional: METR-LA + PEMS-BAY (~215 MB)
python scripts/download_data.py --geodata           # optional: india-geodata inventory
python scripts/preprocess_data.py
python scripts/validate_data.py
```

## VS Code setup
1. `code ~/"ML PROJ"/SmartRoute-AI`
2. Install the extensions *Python*, *Pylance*, *ESLint*, *Tailwind CSS IntelliSense*.
3. Run `⌘⇧P` → *Python: Select Interpreter* → `.venv/bin/python`.
4. Open two terminals (`⌃\``): one for the backend, one for the frontend (commands below).
5. Tests: *Testing* sidebar → configure **pytest** with root folder `tests`.

## Backend startup
```bash
cd ~/"ML PROJ"/SmartRoute-AI
source .venv/bin/activate
uvicorn backend.app.main:app --reload --port 8000
```
Open http://localhost:8000/docs for the interactive API and http://localhost:8000/api/health to check status.

## Frontend startup
```bash
cd ~/"ML PROJ"/SmartRoute-AI/frontend
npm run dev
```
Open http://127.0.0.1:5173.

## Model training
```bash
source .venv/bin/activate
# Q-learning: one table per (preference, landmark). 19 landmarks × 3 preferences × 3000 episodes.
python scripts/train_q_learning.py --episodes 3000
#   faster first run:  python scripts/train_q_learning.py --episodes 1500 --preferences fastest
# DQN (PyTorch)
python scripts/train_dqn.py --episodes 1500
```
All hyper-parameters are flags (`--alpha/--learning-rate --gamma --epsilon --epsilon-decay --batch-size --buffer-size --target-update --seed`). Checkpoints and histories go to `reinforcement_learning/checkpoints/`. Restart the backend, or let an API training job finish, to load them. You can also start smaller jobs from the **RL Training** page. They run as a separate process.

## Model evaluation
```bash
python scripts/evaluate_models.py --pairs 15 --scenarios low,moderate,heavy,peak --events none,incident_on_route
```
This prints a summary table and saves `data/processed/experiments/latest.json`, which is shown on the Comparison page (and stored in MongoDB if configured).

## Tests
```bash
source .venv/bin/activate
pytest -q                       # graph, geocoding, Dijkstra, Q-learning, DQN, invalid actions, traffic, API
cd frontend && npm test         # frontend ↔ API contract (Vitest)
npm run typecheck               # optional strict TypeScript check
python scripts/smoke_test.py http://localhost:8000   # end-to-end against a running backend
```
Tests that need the real graph or PyTorch skip themselves with a clear reason when those are missing.

## Environment variables
| Variable | Where | Purpose |
|---|---|---|
| `MONGODB_URI` | backend | MongoDB Atlas connection string (empty → local JSON files) |
| `MONGODB_DB` | backend | database name (`smartroute`) |
| `TOMTOM_API_KEY` | backend | optional live traffic. Never put it in the frontend. |
| `FRONTEND_ORIGIN` | backend | allowed CORS origin(s), comma-separated |
| `MODEL_CHECKPOINT_DIR` | backend | where `q_learning.npz`, `dqn.npz` are read/written |
| `ARTIFACTS_URL` | backend (Render) | URL of `smartroute-artifacts.tar.gz` fetched at build time |
| `BOOTSTRAP_FROM_OSM` | backend (Render) | `true` = download OSM + quick Q-learning during build when no artifacts exist |
| `ENABLE_SERVER_TRAINING`, `MAX_TRAINING_EPISODES_QL/DQN` | backend | API training switch and limits |
| `QL_ONDEMAND_EPISODES`, `QL_ONDEMAND_BUDGET_S` | backend | budget for non-landmark destinations |
| `DQN_INFERENCE_BACKEND` | backend | `numpy` (default) or `torch` |
| `NOMINATIM_ENABLED`, `CONTACT_EMAIL` | backend | geocoding and OSM-policy User-Agent |
| `FUEL_PRICE_INR_PER_L`, `MILEAGE_KM_PER_L` | backend | cost-model assumptions |
| `VITE_API_URL` | frontend (build time) | backend URL, e.g. `https://smartroute-api.onrender.com` |

## Push to GitHub
```bash
cd ~/"ML PROJ"/SmartRoute-AI
git init -b main
git add .
git commit -m "SmartRoute AI: RL route optimisation for Hyderabad"
# create an empty repo named SmartRoute-AI on github.com (no README), then:
git remote add origin https://github.com/<your-username>/SmartRoute-AI.git
git push -u origin main
```
`.gitignore` excludes raw data, processed data, checkpoints, `.env` and `node_modules`.

## Render deployment
Render's disk is wiped on every deploy and restart, so the trained models travel as a release asset.

1. **Package data + models** (after training on your Mac):
   ```bash
   python scripts/package_artifacts.py          # -> dist/smartroute-artifacts.tar.gz
   ```
2. **Upload it as a GitHub Release**: on github.com go to your repo → *Releases* → *Draft a new release* → tag `v1.0.0` → attach `dist/smartroute-artifacts.tar.gz` → *Publish*. (With GitHub CLI: `gh release create v1.0.0 dist/smartroute-artifacts.tar.gz`.) Copy the asset link: `https://github.com/<user>/SmartRoute-AI/releases/download/v1.0.0/smartroute-artifacts.tar.gz`. The repo must be public, or the asset must be reachable without a login.
3. **MongoDB Atlas**: create a free M0 cluster, then *Database Access* → add a user and password. Under *Network Access* add `0.0.0.0/0`, because Render free services have no fixed IP. Then *Connect* → *Drivers* → copy the `mongodb+srv://…` string.
4. **Render Blueprint**: dashboard → *New* → *Blueprint* → select the repo. Render reads `render.yaml` and creates:
   * `smartroute-api` (Python web service; build `pip install -r backend/requirements.txt && python scripts/bootstrap_deploy.py`; start `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`)
   * `smartroute-ai` (static site; root `frontend`, build `npm install && npm run build`, publish `dist`, SPA rewrite to `index.html`)
5. When prompted, set the secret variables on `smartroute-api`: `MONGODB_URI`, `ARTIFACTS_URL`, optional `TOMTOM_API_KEY`, `CONTACT_EMAIL`.
6. If Render gives your services different URLs than `smartroute-api.onrender.com` / `smartroute-ai.onrender.com`, update `VITE_API_URL` on the static site and `FRONTEND_ORIGIN` on the API, then redeploy both. The API URL is baked in at build time.
7. **Verify** before announcing it:
   ```bash
   python scripts/smoke_test.py https://smartroute-api.onrender.com --origin https://smartroute-ai.onrender.com
   ```
   Then open the static site, plan Hitech City → Charminar and confirm three coloured routes appear.

Notes: the free web service sleeps after inactivity, so the first request can take about a minute (the frontend allows 90 s). DQN training needs PyTorch, which is not installed on Render by default. Train it on your Mac and ship it in the artifacts. Q-learning API jobs work on Render, but their checkpoints are lost on the next deploy.

## Deploy without Git (Docker image on Render)
Render can run a prebuilt Docker image instead of a Git repo. The `Dockerfile` packs the frontend, the API, the Hyderabad graph and your trained models into one image, so it's a single Render service on a single URL with no CORS and no `ARTIFACTS_URL`.

1. Install Docker Desktop (https://www.docker.com/products/docker-desktop/) and create a free Docker Hub account.
2. Train first, so `data/processed/` and `reinforcement_learning/checkpoints/` exist (they are copied into the image).
3. Build for Render's x86 servers and push:
   ```bash
   cd ~/"ML PROJ"/SmartRoute-AI
   docker login
   docker buildx build --platform linux/amd64 -t <dockerhub-user>/smartroute-ai:latest --push .
   ```
4. Render dashboard → *New* → *Web Service* → *Existing image* → `docker.io/<dockerhub-user>/smartroute-ai:latest`, free instance, region Singapore.
5. Environment: `MONGODB_URI` (optional), `TOMTOM_API_KEY` (optional), `CONTACT_EMAIL`. Render sets `PORT` itself.
6. Verify: `python scripts/smoke_test.py https://<your-service>.onrender.com`, then open the same URL in a browser.

To update, rebuild and push the image, then click *Manual Deploy* on Render.

## API documentation
Interactive docs: `/docs` (Swagger) and `/redoc`. Summary in [`docs/api.md`](docs/api.md).

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | status, graph, models, database, traffic mode |
| GET | `/api/locations/search?q=` | landmarks + Nominatim (Hyderabad box) |
| GET | `/api/locations/landmarks` · `/api/locations/reverse` | built-in places · map-click naming |
| GET | `/api/map/network` | road segments for the traffic layer |
| POST | `/api/routes/compare` | Dijkstra + Q-Learning + DQN with geometry, metrics, explanation |
| POST | `/api/routes/optimize` | single best (or chosen) route |
| GET | `/api/routes/recent` | stored comparisons |
| GET | `/api/traffic/current` | simulation state (`include_edges=true` for per-edge congestion) |
| POST | `/api/traffic/simulate` · `/advance` · `/incident` · `/incident/clear` · `/reset` | simulation controls |
| POST | `/api/traffic/live/refresh` · `/live/disable` | TomTom live mode |
| POST | `/api/training/start` · `/training/stop` | background training jobs |
| GET | `/api/training/status` · `/training/history` | progress and saved curves |
| POST/GET | `/api/experiments/run` · `/experiments/latest` | algorithm comparison |
| GET | `/api/analytics/summary` · `/api/datasets` · `/api/models` | dashboards, provenance, model info |

## Screenshots
Add your own after running locally (save in `docs/screenshots/`):
`overview.png`, `route-optimizer.png`, `traffic.png`, `training.png`, `comparison.png`.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `Road network not loaded` | run `download_data.py --osm --landmarks` and `preprocess_data.py`; check `data/processed/hyderabad_graph.json.gz` exists |
| OSMnx download times out | Overpass is busy. Re-run, since partial results are cached in `data/raw/osmnx_cache`. |
| `pip install osmnx` fails on Mac | `brew install gdal` (rare with recent wheels) or use conda: `conda install -c conda-forge osmnx` |
| Frontend shows "Cannot reach the backend" | backend running on :8000? `frontend/.env` has `VITE_API_URL=http://localhost:8000`? restart `npm run dev` after editing `.env` |
| CORS error in the browser console | add the frontend URL to `FRONTEND_ORIGIN` and restart the backend |
| DQN shows "checkpoint not loaded" | run `python scripts/train_dqn.py`, then restart the backend |
| Q-Learning slow for a clicked destination | non-landmark destinations train on demand (≤ 8 s). Pick a landmark for the pre-trained table. |
| MongoDB "connection failed" in `/api/health` | check the URI, the user's password and Atlas *Network Access*. The app keeps working on local files. |
| Render build: "no road graph available" | set `ARTIFACTS_URL` (or `BOOTSTRAP_FROM_OSM=true`) and redeploy |
| `.h5` files can't be read | `pip install tables`. The CSV versions are used automatically when present. |
