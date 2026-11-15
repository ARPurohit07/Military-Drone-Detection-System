"""Collect licensed bird-flock hard negatives from Wikimedia Commons.

The current detector performs well on UAV_SMID birds but can confuse external
stock-photo silhouettes with tiny synthetic drones. This collector targets that
specific domain gap, records per-image provenance/licensing, rejects duplicates,
and creates no drone boxes. The binary dataset builder imports these images as
hard negatives and keeps a deterministic held-out subset for verification.
"""

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import requests

from data.build.hashing import DuplicateIndex, dhash_file

API = "https://commons.wikimedia.org/w/api.php"
OUT = Path("dataset_sources/bird_flocks")
MANIFEST = OUT / "manifest.jsonl"
HEADERS = {
    "User-Agent": "DroneResearchDataset/1.0 (hard-negative research dataset)"
}
QUERIES = (
    "flock birds silhouette sky",
    "birds flying silhouette",
    "birds flying against clouds",
    "birds flying sunset silhouette",
    "migrating birds sky",
    "geese flying formation sky",
    "starlings flock sky",
    "swallows flock sky",
    "seagulls flock sky",
    "vultures flying sky",
    "birds circling sky",
    "black birds blue sky",
    "flock birds dramatic clouds",
    "distant birds flying sky",
)
EXCLUDED_TITLE_WORDS = {
    "drone", "uav", "quadcopter", "airplane", "aeroplane", "aircraft",
}


def text_value(metadata, key):
    return (metadata.get(key) or {}).get("value")


def search(query, limit=100):
    """Yield Commons file records for one search phrase."""
    continuation = None
    yielded = 0
    inspected = 0
    while inspected < limit:
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": 6,
            "gsrlimit": min(50, limit - inspected),
            "prop": "imageinfo",
            "iiprop": "url|mime|extmetadata",
            "iiurlwidth": 1280,
            "format": "json",
            "formatversion": 2,
        }
        if continuation:
            params["gsroffset"] = continuation
        response = None
        for attempt in range(4):
            response = requests.get(
                API, params=params, headers=HEADERS, timeout=30
            )
            if response.status_code != 429:
                break
            delay = int(response.headers.get("Retry-After", 5 * (attempt + 1)))
            print(f"  Commons rate limit; retrying in {delay}s")
            time.sleep(delay)
        response.raise_for_status()
        payload = response.json()
        pages = payload.get("query", {}).get("pages", [])
        if not pages:
            return
        inspected += len(pages)
        for page in pages:
            info = (page.get("imageinfo") or [{}])[0]
            metadata = info.get("extmetadata") or {}
            title = page.get("title", "")
            lower_title = title.casefold()
            if any(word in lower_title for word in EXCLUDED_TITLE_WORDS):
                continue
            mime = info.get("mime", "")
            if mime not in {"image/jpeg", "image/png"}:
                continue
            image_url = info.get("thumburl") or info.get("url")
            if not image_url:
                continue
            yielded += 1
            yield {
                "query": query,
                "commons_title": title,
                "source_page": info.get("descriptionurl"),
                "image_url": image_url,
                "original_image_url": info.get("url"),
                "license": text_value(metadata, "LicenseShortName"),
                "license_url": text_value(metadata, "LicenseUrl"),
                "artist": re.sub(
                    r"<[^>]+>", "", text_value(metadata, "Artist") or ""
                ).strip(),
            }
        continuation = payload.get("continue", {}).get("gsroffset")
        if continuation is None:
            return
        time.sleep(0.75)


def download(candidate):
    try:
        response = requests.get(
            candidate["image_url"], headers=HEADERS, timeout=30
        )
        response.raise_for_status()
        if "image" not in response.headers.get("Content-Type", ""):
            return candidate, None
        array = cv2.imdecode(
            np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if array is None or min(array.shape[:2]) < 256:
            return candidate, None
        return candidate, array
    except requests.RequestException:
        return candidate, None


def load_existing():
    rows = []
    if MANIFEST.exists():
        rows = [
            json.loads(line)
            for line in MANIFEST.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return rows


def collect(target=400, per_query=100, workers=10):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_existing()
    index = DuplicateIndex()
    seen_pages = set()
    for row in rows:
        path = Path(row["path"])
        if path.exists():
            image_hash = dhash_file(path)
            if image_hash is not None:
                index.add(image_hash, path.name)
        if row.get("source_page"):
            seen_pages.add(row["source_page"])

    print(f"resuming with {len(rows)} bird-flock negatives")
    for query in QUERIES:
        if len(rows) >= target:
            break
        candidates = [
            item for item in search(query, per_query)
            if item.get("source_page") not in seen_pages
        ]
        print(f"{query!r}: {len(candidates)} candidates")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(download, item) for item in candidates]
            for future in as_completed(futures):
                if len(rows) >= target:
                    break
                item, image = future.result()
                if image is None:
                    continue
                temporary = OUT / "_candidate.jpg"
                # Hash the decoded image rather than trusting source bytes.
                ok, encoded = cv2.imencode(
                    ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92]
                )
                if not ok:
                    continue
                temporary.write_bytes(encoded.tobytes())
                image_hash = dhash_file(temporary)
                temporary.unlink(missing_ok=True)
                if image_hash is None or index.find(image_hash) is not None:
                    continue
                filename = f"bird_flock_{len(rows):04d}.jpg"
                path = OUT / filename
                path.write_bytes(encoded.tobytes())
                index.add(image_hash, filename)
                seen_pages.add(item.get("source_page"))
                height, width = image.shape[:2]
                row = {
                    **item,
                    "filename": filename,
                    "path": str(path.as_posix()),
                    "width": width,
                    "height": height,
                    "dhash": image_hash.tobytes().hex(),
                    "category": "bird_flock_commons",
                    "kind": "distractor",
                    "distractor_class": "bird",
                }
                rows.append(row)
                with MANIFEST.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  kept total: {len(rows)}/{target}")

    print(f"collected {len(rows)} licensed bird-flock negatives in {OUT}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=400)
    parser.add_argument("--per-query", type=int, default=100)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    collect(args.target, args.per_query, args.workers)


if __name__ == "__main__":
    main()
