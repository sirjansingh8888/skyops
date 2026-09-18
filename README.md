# SkyOps — an AI control tower for the drone era

Code Cortex 3.0 (TAM-VIT), Drone Tech & Aviation track. SkyOps fuses four data layers into one question a
drone operator or a UTM (unmanned traffic management) desk actually asks: **"Is it safe to fly here, now?"**

| Layer | Data (organisers' pack) | What SkyOps does |
|---|---|---|
| Airspace | OpenSky ADS-B capture over India (214 aircraft, 20 snapshots) | Replay on a live map, 5-minute trajectory prediction, loss-of-separation detection (5 NM / 1000 ft, 3 NM terminal), explainable anomaly flags |
| Drone perception | AU-AIR frames with synced GPS / altitude / IMU; VisDrone-DET | YOLO detections of people and vehicles, geo-projected to the ground from the drone's telemetry, landing-zone GO / CAUTION / NO-GO |
| Land use | Dubai aerial segmentation tiles (6 classes) | Surface segmentation and emergency-landing suitability |
| Fleet health | NASA C-MAPSS turbofan run-to-failure | Remaining Useful Life per engine with p10–p90 band, grounding list |
| Ops assistant | Claude (tool use over all layers) | Natural-language briefs: "Can I launch at the mission site now?" |

The mission brief combines airport zoning (DigitalSky-style red < 5 km, yellow < 12 km), the 120 m ceiling,
low-level manned traffic, predicted intrusions, nearby conflicts/anomalies and (optionally) the camera and
land-use assessments into one score with reasons.

## Quick start (Windows, PowerShell)

```powershell
cd skyops
.\scripts\setup.ps1            # creates .venv, installs CPU PyTorch + deps   (add -Cuda on the RTX laptop)
.\.venv\Scripts\python.exe scripts\download_data.py --all      # ~700 MB: full small sets + VisDrone/AU-AIR subsets
copy .env.example .env         # put ANTHROPIC_API_KEY in .env for the assistant (optional)
.\scripts\run.ps1              # http://127.0.0.1:8000  (API docs at /docs)
```

Linux/macOS: `python -m venv .venv && . .venv/bin/activate && pip install -r requirements-torch-cpu.txt -r requirements.txt`,
then the same `download_data.py` and `python -m uvicorn skyops.api.main:app --port 8000`.

The first detection request downloads `yolov8n.pt` (6 MB) from Ultralytics. The map basemap needs internet
(CARTO tiles); without it the layers render on a plain dark background.

## What runs without any training

Everything. Placeholders keep every panel live until the real models are trained:

| Module | Before training | After training |
|---|---|---|
| Perception | COCO-pretrained YOLOv8n mapped to VisDrone labels (weak on oblique low-altitude frames; the UI can show dataset boxes instead) | `models/visdrone_yolo.pt` fine-tuned on VisDrone (picked up automatically) |
| Land use | brightness/texture colour rules (~63 % pixel accuracy) | `models/landuse_unet.pt` U-Net ResNet-18 |
| Fleet RUL | training-free k-nearest-neighbour over C-MAPSS windows (RMSE ≈ 22 cycles) | `models/rul_lgbm/` LightGBM point + quantile models |
| Trajectory | curvilinear dead reckoning (median error 125 m @ 60 s, 272 m @ 120 s on the capture) | (optional) learned GRU through the same `Predictor` interface |

## Training commands (run explicitly, GPU laptop recommended)

```powershell
# 1. Fleet RUL, CPU, ~1 min
.\.venv\Scripts\python.exe -m skyops.fleet.train --subsets FD001 FD002 FD003 FD004

# 2. VisDrone YOLO fine-tune, GPU, ~20-40 min for 30 epochs on the 800-image subset (fetch more with --visdrone-train 3000)
.\.venv\Scripts\python.exe -m skyops.perception.train_visdrone --model yolov8s.pt --epochs 30 --imgsz 1024 --device 0

# 3. Land-use U-Net, GPU, ~5-10 min
.\.venv\Scripts\python.exe -m skyops.landuse.train --epochs 40 --device cuda
```

Set `SKYOPS_DEVICE=cuda` in `.env` on the GPU laptop so inference uses the GPU too.

### RTX 50-series note
The RTX 5060 (Blackwell, sm_120) needs PyTorch ≥ 2.7 built for CUDA 12.8. `scripts/setup.ps1 -Cuda` installs
from `requirements-torch-cu128.txt`. Older wheels fail with "no kernel image is available".

## Project layout

```
skyops/
  scripts/download_data.py     Drive fetcher; reads the multi-GB zips remotely with HTTP range requests
  scripts/setup.ps1, run.ps1, make_bundle.ps1
  skyops/config.py             paths, constants, settings (SKYOPS_* env vars)
  skyops/airspace/             loader, predict (dead reckoning + evaluation), conflicts, anomalies, airports
  skyops/perception/           detect (YOLO), auair (telemetry), geoproject, landing, analyze, train_visdrone
  skyops/landuse/              data, infer (U-Net or colour rules), analyze, train
  skyops/fleet/                features, predict (LightGBM or k-NN fallback), train
  skyops/mission.py            the go / no-go risk brief
  skyops/assistant/            Claude tool-use agent (tools.py, agent.py)
  skyops/api/main.py           FastAPI app + static UI
  web/                         console (MapLibre + deck.gl, vanilla JS)
  tests/test_smoke.py          pytest smoke tests
  data/raw/                    datasets (git-ignored), models/ (git-ignored)
```

## API (selected)

`GET /api/airspace/snapshot/{t}` states + predicted paths + conflicts + anomalies · `GET /api/airspace/evaluate`
· `POST /api/mission/brief` · `GET /api/fleet/{subset}` · `GET /api/drone/frame/{name}` (+`/image`)
· `POST /api/drone/detect` (upload) · `GET /api/landuse/tile/{name}` (+`/image?mode=overlay|mask|gt`)
· `POST /api/landuse/segment` (upload) · `POST /api/assistant/chat`. Full docs at `/docs`.

## Demo script (3 minutes)

1. Press play: aircraft move over India, predicted paths and separation conflicts appear (alert BDA354 / GFA130
   converging near Delhi), anomalies list explains itself.
2. Mission tab, preset "Delhi (near IGI)": NO-GO with reasons (yellow zone, low-level traffic, predicted
   intrusions). Preset "VIT Vellore": GO. Pick any point on the map.
3. Drone tab: play the flight; detections, telemetry, landing verdict; the footprint and objects appear on the
   map at the mission site.
4. Land use tab: overlay vs truth, suitability score.
5. Fleet tab: grounding list, engine degradation curve with uncertainty band.
6. Assistant: "Can I launch a drone at the mission site right now?" — watch the tool calls.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
