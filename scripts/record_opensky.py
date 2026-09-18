#!/usr/bin/env python
"""Record live OpenSky state vectors over India into a CSV with the same columns as the organisers' capture.

Use it to build a fresh replay for the demo (set SKYOPS_OPENSKY_CSV to the output path) and to collect
more trajectories for the learned predictor.

  python scripts/record_opensky.py --minutes 10 --interval 20
  python scripts/record_opensky.py --minutes 60 --interval 30 --out data/raw/opensky/evening.csv

Anonymous OpenSky access allows roughly 100 calls a day over this bounding box; set
SKYOPS_OPENSKY_CLIENT_ID / SKYOPS_OPENSKY_CLIENT_SECRET in .env for ten times that.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skyops.airspace.live import INDIA_BBOX, fetch_states  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--interval", type=float, default=20.0, help="seconds between polls (OpenSky resolution is 10 s anonymous, 5 s authenticated)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else ROOT / "data" / "raw" / "opensky" / f"recorded_{datetime.now():%Y%m%d_%H%M}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    end = time.time() + a.minutes * 60
    n_snap = n_rows = 0
    last_t = None
    while time.time() < end:
        t0 = time.time()
        try:
            df = fetch_states(INDIA_BBOX)
            t = int(df.snapshot_time.iloc[0]) if len(df) else None
            if len(df) and t != last_t:
                df.to_csv(out, mode="a", header=not out.exists() or n_snap == 0, index=False)
                n_snap, n_rows, last_t = n_snap + 1, n_rows + len(df), t
                print(f"[{datetime.now():%H:%M:%S}] snapshot {n_snap}: {len(df)} aircraft (total rows {n_rows})")
        except Exception as e:  # noqa: BLE001
            print(f"[{datetime.now():%H:%M:%S}] poll failed: {e}")
            if "rate limit" in str(e).lower():
                break
        time.sleep(max(0.0, a.interval - (time.time() - t0)))
    print(f"wrote {n_snap} snapshots, {n_rows} rows -> {out}")
    print(f"replay it with:  SKYOPS_OPENSKY_CSV={out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
