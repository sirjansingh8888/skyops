# SkyOps pitch kit

Everything needed to present SkyOps at Code Cortex 3.0 (Drone Tech & Aviation track): talk tracks, the demo
path, the numbers, and honest answers to the questions judges are likely to ask.

## One line

**SkyOps is an AI control tower for the drone era: it fuses live air traffic, drone cameras, land-use maps and
fleet health to answer one question, "is it safe to fly this mission, here, now?", and shows its reasons.**

## 60-second version

India is opening its low-level airspace to drones, but the people approving flights still work from static
maps. SkyOps is the missing operations desk. It watches real ADS-B traffic over India, predicts every aircraft
five minutes ahead, and flags conflicts and abnormal behaviour. A drone operator asks to fly: SkyOps checks
airport zoning, low-level traffic and predicted intrusions, returns GO, CAUTION or NO-GO with reasons, and plans
a corridor around red zones and other operators' geofences. In the air, a detector we fine-tuned finds people
and vehicles below the drone, the drone's own telemetry places them on the map, and a segmentation model rates
the ground for an emergency landing. A fleet panel predicts remaining engine life with calibrated uncertainty.
An assistant answers in plain language using the same tools. We used all five datasets in the pack, and every
model reports how it was tested, including one experiment that failed.

## 3-minute demo path (or press **Tour** and narrate)

| Time | Show | Say |
|---|---|---|
| 0:00 | Airspace tab, replay playing | "Real ADS-B over India. Each aircraft has a five-minute predicted path; median error is 125 m at one minute, measured against where the aircraft really went." |
| 0:25 | Scenario `combined` | "Real traffic rarely misbehaves on cue, so we inject clearly labelled events. Two jets converge: warning, alert, critical, before separation is lost. IGO721 squawks 7700 and descends: flagged with the reason." |
| 0:55 | Mission tab, Delhi preset | "Can I launch here? NO-GO: yellow zone, an aircraft at 1,050 ft within 9 km, three more predicted to enter. VIT Vellore: GO." |
| 1:20 | Route: Dwarka to Vasant Kunj | "The direct line crosses the IGI red zone. The planner goes around: plus 32% distance, 20 minutes, 67% battery, permission needed for the yellow zone." |
| 1:45 | Register geofence near the approach | "Now the tower watches that volume: predicted intrusions appear with time to entry." |
| 2:05 | Drone tab, play flight | "Detector fine-tuned on VisDrone. On this footage, which it never trained on, recall went from 31% to 77%. GPS, altitude and heading put each object on the map; the landing zone is scored live." |
| 2:30 | Land use, then Fleet | "A U-Net rates the ground for emergency landing. The fleet panel predicts remaining engine life with an 80% band that really covers about 80%." |
| 2:45 | Metrics, then Assistant | "Every model shows its test. The GRU did not beat physics, so we did not ship it. Ask the tower anything: it answers from the same tools." |

If the network is down: the replay, scenarios, mission brief, route planner, drone, land use, fleet and metrics
all work offline (only map tiles, LIVE mode and the assistant need internet).

## Numbers to remember

| Claim | Number | How it was measured |
|---|---|---|
| Trajectory prediction | median error 125 m at 60 s, 272 m at 120 s | against the aircraft's real later positions, 2,234 comparisons |
| Conflict logic | 9 distinct conflict pairs in a 350 s replay; RVSM altitude noise is separated into "marginal" | 5 NM / 1000 ft, 3 NM near airports, 150 ft tolerance |
| Drone detector | VisDrone val mAP50 0.47; AU-AIR recall 31% to 77% | YOLO11s, 2,500 training images, 30 epochs; AU-AIR never used for training |
| Land use | mIoU 0.65, pixel accuracy 84% (colour rules: 63%) | 12 held-out tiles of 72 |
| Fleet RUL | RMSE 15.7 to 17.8 cycles (k-NN baseline 22.2); 80% band covers 76 to 81% | NASA test sets FD001 to FD004, conformal calibration |
| Learned trajectory model | only 1 to 3% better, no gain on turns: not deployed | 3,231 samples from held-out aircraft |
| Route planner | detours in 50 to 230 ms | grid A* with line-of-sight smoothing |
| Live mode | 266 aircraft over India in one call | OpenSky REST API |

## Questions judges may ask

**What is real and what is simulated?** The traffic replay is a real OpenSky capture and LIVE mode is real. The
"scenario" events are simulated and labelled as such in the data, the map and the assistant's answers. The drone
flight is real footage with real telemetry, recorded in Denmark and relocated onto the chosen site. The land-use
tiles are real imagery of Dubai and are not georeferenced to the map. C-MAPSS is NASA's simulated engine data.

**Why not deep learning for trajectory prediction?** We tried. A GRU that corrects dead reckoning, trained on the
capture plus 25 minutes of recorded live traffic, improved mean error by 1 to 3% on held-out aircraft and did
nothing for turning aircraft. At one to five minutes, physics is a strong baseline and we do not have hours of
data. The model stays behind a flag and the evidence is in the Metrics tab.

**How are the drone zones defined?** A simplified DigitalSky-style model: red within 5 km of an airport, yellow
to 12 km, green beyond, with a 120 m ceiling. We measure from the airport reference point; a production system
would ingest the official DigitalSky airspace map with perimeter polygons and temporary restrictions.

**How do you place camera detections on the map?** A pinhole model: drone altitude, heading, pitch and roll from
the telemetry, plus camera tilt and field of view. AU-AIR does not publish exact camera geometry, so tilt and FOV
are estimates exposed as sliders. With a calibrated camera the same code is exact on flat ground.

**Does it scale?** Conflict detection is vectorised pairwise checks across all horizons: about 20 ms for 180
aircraft. A national picture of a few thousand aircraft fits the same approach; beyond that, a spatial index.

**Can the assistant make things up?** It can only answer from tool results: every figure it quotes comes from the
same functions the console uses, and the console shows which tools it called. The API key stays on the server.

**What would you build next?** Official airspace and NOTAM ingestion, Remote ID for cooperative drones, wind and
weather in the risk brief, terrain and obstacle data for the planner, and multi-drone corridor scheduling.

## Review plan for the 30 hours

| Checkpoint | Show |
|---|---|
| Review 1 (5 pm, elimination) | Replay with conflicts and anomalies, mission brief, route planner. Message: "the core decision works." |
| Review 2 (1 am, elimination) | Drone perception with geo-projection, land use, fleet health, geofence alerts, LIVE mode. Message: "all five datasets, one product." |
| Final (10 am) | Tour, assistant, metrics with the negative result. Message: "tested, honest, deployable." |

## Roles on stage

One person drives the console, one narrates, one takes questions with the Metrics tab open.
