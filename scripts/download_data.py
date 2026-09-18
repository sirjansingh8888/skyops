#!/usr/bin/env python
"""Fetch the SkyOps datasets from the Code Cortex 3.0 Drive folders.

Small archives (OpenSky, C-MAPSS, Dubai) are downloaded whole.  The two multi-GB archives
(VisDrone 1.96 GB, AU-AIR 2.46 GB) are read *remotely*: we fetch only the zip central directory
and then pull the files we actually need with HTTP range requests, so a laptop gets a useful
subset in minutes instead of pulling 4.4 GB.  Runs on plain Python (no third-party packages).

Examples
--------
  python scripts/download_data.py --all                      # default subsets, ~700 MB
  python scripts/download_data.py --only opensky cmapss      # tiny, seconds
  python scripts/download_data.py --only visdrone --visdrone-train 3000
  python scripts/download_data.py --only visdrone auair --full   # whole archives (GPU laptop)
  python scripts/download_data.py --list                     # show the plan, download nothing
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
UA = "Mozilla/5.0 (SkyOps data fetcher)"

# Google Drive file IDs from the organisers' "Drone Tech & Aviation" folder
# (https://drive.google.com/drive/folders/13NwAl93bcnGcnb_BZkO1Pn2_b7HpX9G5)
DATASETS = {
    "opensky": dict(id="1AH9TCUkdSP3i-uf1yDq0FtS83AWfsRH9", prefix="03_OpenSky_flight_telemetry/", size_mb=0.1),
    "cmapss": dict(id="1VB5DJTDE-cFhGwi3UXP4mjqdN7zWEsMv", prefix="01_NASA_CMAPSS_predictive_maintenance/", size_mb=12),
    "dubai": dict(id="1SakjZc7sBgFRi1R6y33A9cQuihog0rzN", prefix="02_Dubai_aerial_segmentation/", size_mb=211),
    "visdrone": dict(id="1FlqrIERu4PLA08LWZJ4G5y7-TGzmok9S", prefix="05_VisDrone_detection_tracking/", size_mb=1956),
    "auair": dict(id="19P9SSEuGHj_QooH_7rMh15fNNTroXlZ5", prefix="04_AUAIR_multimodal_uav/", size_mb=2459),
}


def drive_url(file_id: str) -> str:
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


# ----------------------------------------------------------------------------- HTTP helpers
def http_range(url: str, start: int | None = None, end: int | None = None, suffix: int | None = None,
               retries: int = 4) -> tuple[bytes, int]:
    """GET a byte range. Returns (bytes, total_size). Retries on transient errors."""
    rng = f"bytes=-{suffix}" if suffix is not None else f"bytes={start}-{end}"
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Range": rng})
            with urllib.request.urlopen(req, timeout=180) as r:
                if r.status != 206:
                    raise RuntimeError(f"server ignored the Range header (HTTP {r.status})")
                total = int(r.headers["Content-Range"].split("/")[-1])
                return r.read(), total
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"range request failed: {last_err}")


def download_whole(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
        ctype = r.headers.get("Content-Type", "")
        if "text/html" in ctype:
            raise RuntimeError("Drive returned an HTML page instead of the file (quota or permission issue)")
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        t0 = time.time()
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                rate = done / max(time.time() - t0, 1e-6) / 1e6
                print(f"\r  {dest.name}: {done/1e6:7.0f}/{total/1e6:.0f} MB  ({rate:.1f} MB/s)", end="", flush=True)
    print()
    tmp.replace(dest)
    return dest


# ----------------------------------------------------------------------------- remote zip reader
class RemoteZip:
    """Read a zip archive over HTTP using range requests (only the central directory is fetched)."""

    def __init__(self, url: str):
        self.url = url
        tail, self.size = http_range(url, suffix=65536)
        i = tail.rfind(b"PK\x05\x06")
        if i < 0:
            raise RuntimeError("end-of-central-directory record not found (not a zip?)")
        _sig, _disk, _cddisk, _n_disk, n_total, cdsize, cdoff, _clen = struct.unpack("<IHHHHIIH", tail[i:i + 22])
        if 0xFFFFFFFF in (cdoff, cdsize) or n_total == 0xFFFF:  # ZIP64
            j = tail.rfind(b"PK\x06\x06")
            n_total, cdsize, cdoff = struct.unpack("<QQQ", tail[j + 32:j + 56])
        cd, _ = http_range(url, cdoff, cdoff + cdsize - 1)
        self.entries = self._parse_central_directory(cd)

    @staticmethod
    def _parse_central_directory(cd: bytes) -> list[dict]:
        entries, p = [], 0
        while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
            (_sig, _vm, _vn, _flg, comp, _t, _d, _crc, csz, usz, fl, el, cl,
             _dn, _ia, _ea, off) = struct.unpack("<IHHHHHHIIIHHHHHII", cd[p:p + 46])
            name = cd[p + 46:p + 46 + fl].decode("utf-8", "replace")
            extra = cd[p + 46 + fl:p + 46 + fl + el]
            if 0xFFFFFFFF in (off, csz, usz):  # ZIP64 extra field
                q = 0
                while q + 4 <= len(extra):
                    hid, hsz = struct.unpack("<HH", extra[q:q + 4])
                    body = extra[q + 4:q + 4 + hsz]
                    if hid == 1:
                        vals = list(struct.unpack("<" + "Q" * (len(body) // 8), body[:8 * (len(body) // 8)]))
                        if usz == 0xFFFFFFFF:
                            usz = vals.pop(0)
                        if csz == 0xFFFFFFFF:
                            csz = vals.pop(0)
                        if off == 0xFFFFFFFF:
                            off = vals.pop(0)
                    q += 4 + hsz
            if not name.endswith("/"):
                entries.append(dict(name=name, comp=comp, csz=csz, usz=usz, off=off))
            p += 46 + fl + el + cl
        return entries

    def select(self, predicate) -> list[dict]:
        return [e for e in self.entries if predicate(e["name"])]

    def extract(self, entries: list[dict], dest_root: Path, strip_prefix: str = "",
                chunk_mb: int = 48, skip_existing: bool = True) -> int:
        """Extract entries, batching neighbouring files into one range request per ~chunk_mb."""

        def rel(e: dict) -> Path:
            n = e["name"]
            if strip_prefix and n.startswith(strip_prefix):
                n = n[len(strip_prefix):]
            return dest_root / n

        wanted = []
        for e in sorted(entries, key=lambda x: x["off"]):
            out = rel(e)
            if skip_existing and out.exists() and out.stat().st_size == e["usz"]:
                continue
            wanted.append(e)
        if not wanted:
            print(f"  nothing to do ({len(entries)} files already present)")
            return 0

        groups: list[list[dict]] = []
        cur: list[dict] = []
        for e in wanted:
            if cur:
                prev = cur[-1]
                gap = e["off"] - (prev["off"] + prev["csz"] + 30 + len(prev["name"]))
                span = e["off"] + e["csz"] - cur[0]["off"]
                if gap > (1 << 20) or span > (chunk_mb << 20):
                    groups.append(cur)
                    cur = []
            cur.append(e)
        if cur:
            groups.append(cur)

        n_done, total_bytes = 0, sum(e["csz"] for e in wanted)
        got = 0
        t0 = time.time()
        for g in groups:
            first, last = g[0], g[-1]
            start = first["off"]
            end = min(self.size - 1, last["off"] + 30 + len(last["name"]) + 65535 + last["csz"])
            blob, _ = http_range(self.url, start, end)
            for e in g:
                p = e["off"] - start
                if blob[p:p + 4] != b"PK\x03\x04":
                    raise RuntimeError(f"bad local header for {e['name']}")
                fl, el = struct.unpack("<HH", blob[p + 26:p + 30])
                data = blob[p + 30 + fl + el:p + 30 + fl + el + e["csz"]]
                if e["comp"] == 8:
                    data = zlib.decompress(data, -15)
                elif e["comp"] != 0:
                    raise RuntimeError(f"unsupported compression method {e['comp']} for {e['name']}")
                out = rel(e)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(data)
                n_done += 1
                got += e["csz"]
            rate = got / max(time.time() - t0, 1e-6) / 1e6
            print(f"\r  {n_done}/{len(wanted)} files, {got/1e6:.0f}/{total_bytes/1e6:.0f} MB ({rate:.1f} MB/s)",
                  end="", flush=True)
        print()
        return n_done


# ----------------------------------------------------------------------------- per-dataset plans
def unzip_strip(zip_path: Path, dest: Path, prefix: str) -> int:
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            name = info.filename[len(prefix):] if info.filename.startswith(prefix) else info.filename
            out = dest / name
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(out, "wb") as f:
                f.write(src.read())
            n += 1
    return n


def fetch_whole_archive(key: str, dest: Path, keep_zip: bool) -> int:
    meta = DATASETS[key]
    zpath = RAW / f"{key}.zip"
    if not zpath.exists():
        print(f"  downloading {key} archive (~{meta['size_mb']} MB)")
        download_whole(drive_url(meta["id"]), zpath)
    n = unzip_strip(zpath, dest, meta["prefix"])
    if not keep_zip:
        zpath.unlink(missing_ok=True)
    return n


def plan_visdrone(rz: RemoteZip, n_train: int, n_val: int | None, n_test: int) -> list[dict]:
    pre = DATASETS["visdrone"]["prefix"]

    def split_entries(split: str, limit: int | None) -> list[dict]:
        imgs = sorted(rz.select(lambda n: n.startswith(f"{pre}VisDrone2019-DET-{split}/images/")), key=lambda e: e["name"])
        if limit is not None:
            imgs = imgs[:limit]
        stems = {Path(e["name"]).stem for e in imgs}
        anns = rz.select(lambda n: n.startswith(f"{pre}VisDrone2019-DET-{split}/annotations/") and Path(n).stem in stems)
        return imgs + anns

    chosen = split_entries("val", n_val) + split_entries("train", n_train) + split_entries("test-dev", n_test)
    chosen += rz.select(lambda n: n == f"{pre}README.md")
    return chosen


def plan_auair(rz: RemoteZip, n_frames: int) -> list[dict]:
    pre = DATASETS["auair"]["prefix"]
    imgs = sorted(rz.select(lambda n: n.startswith(f"{pre}images/")), key=lambda e: e["name"])[:n_frames]
    meta = rz.select(lambda n: n in (f"{pre}annotations.json", f"{pre}README.md"))
    return imgs + meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="fetch every dataset (default subsets for the big ones)")
    ap.add_argument("--only", nargs="+", choices=list(DATASETS), help="fetch only these datasets")
    ap.add_argument("--full", action="store_true", help="download the whole VisDrone / AU-AIR archives")
    ap.add_argument("--keep-zip", action="store_true", help="keep downloaded .zip files")
    ap.add_argument("--visdrone-train", type=int, default=800, help="train images to fetch (subset mode)")
    ap.add_argument("--visdrone-val", type=int, default=None, help="val images to fetch (default: all 548)")
    ap.add_argument("--visdrone-test", type=int, default=0, help="test-dev images to fetch")
    ap.add_argument("--auair-frames", type=int, default=600, help="AU-AIR frames to fetch (subset mode)")
    ap.add_argument("--list", action="store_true", help="print the plan without downloading")
    args = ap.parse_args()

    keys = args.only or (list(DATASETS) if args.all else [])
    if not keys:
        ap.print_help()
        return 1

    RAW.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for key in keys:
        dest = RAW / key
        print(f"\n== {key} -> {dest}")
        t0 = time.time()
        if key in ("opensky", "cmapss", "dubai") or args.full:
            if args.list:
                print(f"  would download the whole archive (~{DATASETS[key]['size_mb']} MB)")
                continue
            n = fetch_whole_archive(key, dest, args.keep_zip)
        else:
            rz = RemoteZip(drive_url(DATASETS[key]["id"]))
            plan = plan_visdrone(rz, args.visdrone_train, args.visdrone_val, args.visdrone_test) if key == "visdrone" \
                else plan_auair(rz, args.auair_frames)
            mb = sum(e["csz"] for e in plan) / 1e6
            print(f"  archive has {len(rz.entries)} files ({rz.size/1e9:.2f} GB); plan = {len(plan)} files, {mb:.0f} MB")
            if args.list:
                continue
            n = rz.extract(plan, dest, strip_prefix=DATASETS[key]["prefix"])
        manifest[key] = dict(files_written=n, seconds=round(time.time() - t0, 1), full=bool(args.full or key in ("opensky", "cmapss", "dubai")))
        print(f"  done: {n} files in {time.time()-t0:.0f}s")

    if not args.list:
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"\nmanifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
