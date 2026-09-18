# Moving SkyOps between laptops

The project is designed so that **only the source tree needs to travel**. Everything heavy is rebuilt or
re-downloaded on the other machine in a few minutes:

| What | Size | Travels via |
|---|---|---|
| Source (`skyops/`, `web/`, `scripts/`, `tests/`, docs) | ~4 MB | git (recommended) or the zip bundle |
| `.venv/` | ~3 GB | **never** — recreate with `scripts/setup.ps1` |
| `data/raw/` | ~550 MB default subset | re-download with `scripts/download_data.py --all` (5-10 min), or the zip bundle with `-IncludeData` |
| `models/` (trained weights) | 10-100 MB | zip bundle, USB, or a shared Drive folder; git-ignored on purpose |
| `.env` (API key) | 1 KB | retype it; never commit it |

## Option A — git + GitHub (recommended, keeps history, both laptops stay in sync)

On this laptop, once:

```powershell
cd <path-to>\skyops
git add -A
git commit -m "SkyOps: airspace, perception, land use, fleet, assistant, console"
# create an empty private repo on GitHub (no README), then:
git remote add origin https://github.com/<you>/skyops.git
git push -u origin main
```

On the teammate's laptop:

```powershell
git clone https://github.com/<you>/skyops.git
cd skyops
.\scripts\setup.ps1 -Cuda                                   # RTX 5060 -> CUDA 12.8 PyTorch
.\.venv\Scripts\python.exe scripts\download_data.py --all   # or add: --only visdrone --visdrone-train 3000
copy .env.example .env  # add ANTHROPIC_API_KEY, set SKYOPS_DEVICE=cuda
.\scripts\run.ps1
```

Keep working on either machine: `git pull` before you start, `git add -A && git commit -m "..." && git push`
when you stop. Trained weights go in `models/` and are shared separately (see below).

## Option B — zip bundle (no account needed, works over USB / WhatsApp / Drive)

```powershell
.\scripts\make_bundle.ps1                 # -> ..\skyops_bundle_<date>.zip (source only, ~5 MB)
.\scripts\make_bundle.ps1 -IncludeData    # also packs data\raw (~550 MB) so no re-download is needed
.\scripts\make_bundle.ps1 -IncludeModels  # also packs trained weights
```

Unzip on the other laptop, then run `scripts\setup.ps1 -Cuda` and `scripts\run.ps1` as above. The bundle never
contains `.venv`, `.git` internals are included so history survives.

## Sharing trained weights back

After training on the RTX laptop the outputs are:

- `models/rul_lgbm/` (three .txt boosters + manifest.json, ~5 MB)
- `models/visdrone_yolo.pt` (~20-50 MB depending on the base model)
- `models/landuse_unet.pt` (~60 MB)

Zip the `models` folder (or run `make_bundle.ps1 -IncludeModels`) and drop it into the same place on the other
machine; the API picks the files up automatically at startup (`GET /api/health` shows which models are loaded).
If you want weights in git, install git-lfs and run `git lfs track "models/**"` before committing them.

## Checking the GPU laptop

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
nvidia-smi
```

If `cuda.is_available()` is False on the RTX 5060, the CPU wheels were installed: run
`.\.venv\Scripts\python.exe -m pip install --force-reinstall -r requirements-torch-cu128.txt`.

## Before the event checklist

- [ ] Both laptops: `pytest -q` passes, `run.ps1` serves the console, `/api/health` shows the datasets.
- [ ] Trained weights copied to both machines (`/api/health` shows `visdrone_yolo`, `landuse_unet`, `rul_lgbm` true).
- [ ] `.env` with the API key on both machines (assistant chip turns green).
- [ ] Basemap tiles need internet; the rest works offline.
