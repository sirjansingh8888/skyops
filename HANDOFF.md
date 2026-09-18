# Moving SkyOps between laptops

The project is designed so that **only the source tree needs to travel**. Everything heavy is rebuilt or
re-downloaded on the other machine in a few minutes:

| What | Size | Travels via |
|---|---|---|
| Source (`skyops/`, `web/`, `scripts/`, `tests/`, docs) | ~4 MB | git (recommended) or the zip bundle |
| `.venv/` | ~3 GB | **never** — recreate with `scripts/setup.ps1` |
| `data/raw/` | ~550 MB default subset | re-download with `scripts/download_data.py --all` (5-10 min), or the zip bundle with `-IncludeData` |
| `models/` (trained weights) | ~60 MB | git: the trained weights are committed, so a fresh clone runs fully trained |
| `.env` (API key) | 1 KB | retype it; never commit it |

## Option A — git + GitHub (recommended, keeps history, both laptops stay in sync)

The repository is public at https://github.com/sirjansingh8888/skyops.

On the teammate's laptop:

```powershell
git clone https://github.com/sirjansingh8888/skyops.git
cd skyops
.\scripts\setup.ps1 -Cuda                                   # RTX 5060 -> CUDA 12.8 PyTorch
.\.venv\Scripts\python.exe scripts\download_data.py --all   # or add: --only visdrone --visdrone-train 3000
copy .env.example .env  # add your own GEMINI_API_KEY (steps below); the GPU is picked up automatically
.\scripts\run.ps1
```

Keep working on either machine: `git pull` before you start, `git add -A && git commit -m "..." && git push`
when you stop. Trained weights in `models/` travel with git too.

## Option B — zip bundle (no account needed, works over USB / WhatsApp / Drive)

```powershell
.\scripts\make_bundle.ps1                 # -> ..\skyops_bundle_<date>.zip (source only, ~5 MB)
.\scripts\make_bundle.ps1 -IncludeData    # also packs data\raw (~550 MB) so no re-download is needed
.\scripts\make_bundle.ps1 -IncludeModels  # also packs trained weights
```

Unzip on the other laptop, then run `scripts\setup.ps1 -Cuda` and `scripts\run.ps1` as above. The bundle never
contains `.venv`, `.git` internals are included so history survives.

## Assistant: get a free Gemini API key (2 minutes, no card needed)

The Assistant tab talks to Google's Gemini API on its **free tier**. Each teammate should create their own key:
keys belong to a Google account and the free quota is per project, so sharing one key means sharing one quota.

**Get the key**

1. Open https://aistudio.google.com and sign in with a Google account. Use a personal Gmail if your college
   Workspace account blocks AI Studio.
2. Accept the terms of service when asked. For a first-time user AI Studio creates a default project for you.
3. Go to the API keys page: https://aistudio.google.com/apikey (or left panel, Dashboard, API Keys).
4. Click **Create API key**, pick the default project, and copy the key that is shown.
   New keys are created as "auth keys" restricted to the Gemini API, which is what we want.
5. Do **not** set up billing. Without a billing account the project stays on the Free tier and cannot be charged.

**Put it in the project**

1. In the `skyops` folder, make sure a file named `.env` exists (`scripts\setup.ps1` creates it from `.env.example`).
2. Open `.env` in any editor and fill in the first setting, with no quotes and no spaces:

   ```
   GEMINI_API_KEY=paste-your-key-here
   ```

3. Restart the server (`.\scripts\run.ps1`). The chip in the top bar turns green and reads
   `assistant: gemini-3.6-flash`. You can also check http://127.0.0.1:8000/api/assistant/status.

`.env` is git-ignored. Never commit the key, paste it into chat, or put it in the README. If it leaks, delete it on
the API keys page and create a new one.

**Good to know**

- Without a key the Assistant tab still works in *offline rules* mode: it picks a tool from keywords in the question and
  fills a template, so the preset questions always answer with live numbers. The same fallback kicks in if Gemini
  cannot be reached or the quota runs out during the demo.

- Free-tier limits are per minute and per day and are shown in AI Studio (Dashboard, Usage and rate limits). One
  question costs 2 to 4 requests because the assistant calls tools.
- Speed: we tested every free Flash model with a real tool question. `gemini-3.6-flash` answered in about 10 s and
  is the default; `gemini-3.5-flash-lite` took about 4 s but made a unit mistake; the newest `gemini-3.7-flash` and
  `gemini-3.8-flash` were overloaded on the free tier (timeouts, 90 to 160 s). Each call is capped at 25 s; after a
  timeout or rate limit the question finishes on the lite model, and if that fails too, the offline rules answer.
- Google may use free-tier prompts to improve its products, so do not paste private data.
- Change the model with `SKYOPS_GEMINI_MODEL=...` in `.env` (any Flash or Flash-Lite model with a free tier on
  https://ai.google.dev/gemini-api/docs/pricing works).
- The key never reaches the browser: only the FastAPI server calls Gemini.
- Optional: the assistant can run on Claude instead (`pip install anthropic`, `SKYOPS_ASSISTANT_PROVIDER=claude`,
  `ANTHROPIC_API_KEY=...`).

## Trained weights

The trained weights live in `models/` and are committed to git (all under 60 MB), so `git pull` is all the other
laptop needs:

- `models/rul_lgbm/` fleet RUL boosters + manifest
- `models/visdrone_yolo.pt` VisDrone fine-tuned detector
- `models/landuse_unet.pt` (+ `.json` metrics) land-use U-Net, stored in half precision
- `models/traj_gru.pt` (+ `.json` report) trajectory residual GRU

If you retrain on the RTX laptop, commit the new files and push. `GET /api/health` shows which models are loaded,
and the Metrics tab shows their scores. Large intermediate outputs (`runs/`) stay out of git.

## Checking the GPU laptop

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
nvidia-smi
```

If `cuda.is_available()` is False on the RTX 5060, the CPU wheels were installed: run
`.\.venv\Scripts\python.exe -m pip install --force-reinstall -r requirements-torch-cu128.txt`.

## Before the event checklist

- [ ] Both laptops: `pytest -q` passes, `run.ps1` serves the console, `/api/health` shows the datasets.
- [ ] `git pull` on both machines; `/api/health` shows `visdrone_yolo`, `landuse_unet`, `rul_lgbm` true.
- [ ] `.env` with each person's own `GEMINI_API_KEY` on both machines (assistant chip turns green).
- [ ] Basemap tiles need internet; the rest works offline.
