"""Scrape drone imagery per airframe, recording provenance for every file.

Search is driven by the airframe registry, one query at a time, so the country
label is a property of the query and never a per-image judgement. Everything the
downstream stages need to trust a file - which airframe query produced it, which
page it came from, its perceptual hash - is written to a JSONL manifest as the
download happens.

Two properties matter more than raw throughput:

Resumability. A scrape of 91 queries over a flaky connection will be
interrupted. The manifest is append-only and read back on startup, so rerunning
skips what is already on disk instead of redownloading it.

Dedup at the door. Image search returns the same press photo repeatedly, both
within one query and across related queries ("Shahed-136" and "Geran-2" are the
same aircraft). Candidates are hashed in memory and rejected before they are
written, which is what stops the new dataset from repeating the old one's
duplication. Cross-airframe collisions are recorded rather than silently
dropped: when one picture answers two different airframe queries, that is
exactly the label ambiguity that produced the old corpus's 39 mislabelled pairs,
and it needs to be visible.
"""

import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from data.airframes import AIRFRAMES, BY_KEY, all_queries
from data.build.hashing import DuplicateIndex, dhash_bytes

RAW_ROOT = Path("dataset_raw")
MANIFEST = RAW_ROOT / "manifest.jsonl"
REJECTS = RAW_ROOT / "rejects.jsonl"

# A browser User-Agent is required, not cosmetic: several hosts (Wikimedia among
# them) answer 403 to the default python-requests agent.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

MIN_SIDE = 160          # below this an aircraft is too small to box reliably
MAX_BYTES = 12_000_000

# Dead links dominate the wall clock: a meaningful share of image-search results
# are 403s, 404s and hosts that simply hang. At a 12s timeout and one download at
# a time, a full 64-airframe scrape projected to ~6 hours, almost all of it spent
# waiting on links that were never going to resolve. Downloads therefore run
# concurrently and time out sooner. Searches stay strictly sequential - those are
# rate-limited per IP, and a throttled search returns an empty list rather than an
# error, which would silently leave an airframe with no images.
REQUEST_TIMEOUT = 6
DOWNLOAD_WORKERS = 12
PER_QUERY_DELAY = (1.0, 2.5)   # jittered, to stay polite to the search endpoint

GENERIC_QUERY_TOKENS = {
    "drone", "uav", "uas", "ucav", "military", "aircraft", "unmanned",
    "aerial", "vehicle", "flight", "flying", "photo", "takeoff", "armed",
    "system", "systems", "force", "navy", "army", "the", "in", "of",
}
for _country in {airframe.country for airframe in AIRFRAMES}:
    GENERIC_QUERY_TOKENS.update(
        token for token in re.findall(r"[a-z0-9]+", _country.casefold())
        if len(token) > 1)
GENERIC_QUERY_TOKENS.update({
    "american", "british", "chinese", "french", "german", "indian",
    "iranian", "israeli", "italian", "korean", "pakistani", "polish",
    "russian", "serbian", "turkish", "ukrainian",
})


def _identity_tokens(query):
    """Return aircraft/manufacturer terms, excluding generic query wording."""
    return [token for token in re.findall(r"[a-z0-9-]+", query.casefold())
            if len(token) > 1 and token not in GENERIC_QUERY_TOKENS]


def _load_manifest():
    """Return (records, seen_urls). Missing or truncated lines are tolerated."""
    records, urls = [], set()
    if MANIFEST.exists():
        with MANIFEST.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue      # partial final line from an interrupted run
                records.append(rec)
                urls.add(rec["image_url"])
    return records, urls


def _append(path, rec):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def _search(query, max_results, use_commons=True):
    """Image search, returning [] rather than raising so one bad query cannot
    abort a multi-hour scrape. DDGS is primary; Bing HTML is a fallback because
    DDGS may throttle every query after the first without a distinct HTTP error.

    Bing results are filtered on the non-generic query tokens. Image search can
    otherwise return a different aircraft from the same manufacturer (for
    example TB2 for an Akinci query), which would create a confident but wrong
    country/airframe label downstream.
    """
    results = []
    try:
        from ddgs import DDGS
        hits = DDGS().images(query, max_results=max_results)
        if hits:
            for hit in hits:
                hit["source_provider"] = "DDGS"
            results.extend(hits)
    except Exception as exc:                                  # noqa: BLE001
        print(f"      ! DDGS failed: {type(exc).__name__}: {exc}; trying Bing")

    try:
        from bs4 import BeautifulSoup

        response = requests.get(
            "https://www.bing.com/images/async",
            params={"q": query, "first": 0,
                    "count": min(max_results * 2, 150), "adlt": "strict"},
            headers=HEADERS, timeout=REQUEST_TIMEOUT * 2,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        tokens = _identity_tokens(query)
        needed = max(1, (3 * len(tokens) + 4) // 5) if tokens else 0

        bing_results = []
        for anchor in soup.select("a.iusc"):
            try:
                meta = json.loads(anchor.get("m", "{}"))
            except json.JSONDecodeError:
                continue
            image_url = meta.get("murl")
            if not image_url:
                continue
            searchable = " ".join((meta.get("t") or "", meta.get("purl") or "",
                                   image_url)).lower()
            if needed and sum(token in searchable for token in tokens) < needed:
                continue
            bing_results.append({
                "image": image_url,
                "url": meta.get("purl"),
                "title": meta.get("t"),
                "thumbnail": meta.get("turl"),
                "width": meta.get("w") or 0,
                "height": meta.get("h") or 0,
                "source_provider": "Bing Images",
            })
            if len(bing_results) >= max_results:
                break
        if bing_results:
            print(f"      Bing fallback returned {len(bing_results)} relevant hits")
            results.extend(bing_results)
    except Exception as exc:                                  # noqa: BLE001
        print(f"      ! Bing fallback failed: {type(exc).__name__}: {exc}")

    # Wikimedia Commons is slower but has much stronger provenance than an
    # anonymous image-search result and excellent coverage for established
    # military airframes. Request a bounded thumbnail rather than multi-megabyte
    # originals; the source page and declared licence are retained in metadata.
    try:
        if not use_commons:
            raise RuntimeError("commons_skipped_for_query_variant")
        params = {
            "action": "query", "generator": "search", "gsrsearch": query,
            "gsrnamespace": 6, "gsrlimit": min(max_results, 50),
            "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": 1600, "format": "json", "origin": "*",
        }
        commons_headers = dict(HEADERS)
        commons_headers["User-Agent"] = (
            "DroneResearchDataset/1.0 (educational computer-vision project)"
        )
        response = requests.get("https://commons.wikimedia.org/w/api.php",
                                params=params, headers=commons_headers,
                                timeout=REQUEST_TIMEOUT * 2)
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", {}).values()
        tokens = _identity_tokens(query)
        commons_results = []
        for page in pages:
            info_list = page.get("imageinfo") or []
            if not info_list:
                continue
            info = info_list[0]
            if not (info.get("mime") or "").startswith("image/"):
                continue
            searchable = (page.get("title") or "").lower()
            # At least half of the identity tokens must occur in the title.
            # This admits "TAI Anka" / "Anka-S" variants but rejects generic
            # country and event photographs returned by full-text search.
            needed = max(1, (3 * len(tokens) + 4) // 5) if tokens else 0
            if needed and sum(token in searchable for token in tokens) < needed:
                continue
            ext = info.get("extmetadata") or {}
            commons_results.append({
                "image": info.get("thumburl") or info.get("url"),
                "url": info.get("descriptionurl"),
                "title": page.get("title"),
                "thumbnail": info.get("thumburl"),
                "width": info.get("thumbwidth") or info.get("width") or 0,
                "height": info.get("thumbheight") or info.get("height") or 0,
                "source_provider": "Wikimedia Commons",
                "license": (ext.get("LicenseShortName") or {}).get("value"),
                "license_url": (ext.get("LicenseUrl") or {}).get("value"),
                "original_image_url": info.get("url"),
            })
        if commons_results:
            print(f"      Commons returned {len(commons_results)} relevant hits")
            results.extend(commons_results)
    except RuntimeError as exc:
        if str(exc) != "commons_skipped_for_query_variant":
            raise
    except Exception as exc:                                  # noqa: BLE001
        print(f"      ! Commons fallback failed: {type(exc).__name__}: {exc}")

    # Preserve provider order but collapse duplicate URLs before download.
    unique = []
    urls = set()
    for hit in results:
        url = hit.get("image")
        if url and url not in urls:
            urls.add(url)
            unique.append(hit)
    # Each provider's limit is already bounded. Do not truncate the combined
    # list here: an earlier provider may consist entirely of URLs seen on a
    # previous run, and truncation would hide fresh Commons candidates from
    # the caller before its persistent seen-URL filter can inspect them.
    return unique


def _fetch(url):
    """Download an image, returning bytes or None."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT,
                            stream=True)
        if resp.status_code != 200:
            return None, f"http_{resp.status_code}"
        ctype = resp.headers.get("Content-Type", "")
        if "image" not in ctype:
            return None, f"content_type_{ctype or 'unknown'}"
        raw = resp.raw.read(MAX_BYTES + 1, decode_content=True)
        if len(raw) > MAX_BYTES:
            return None, "too_large"
        if len(raw) < 2000:
            return None, "too_small_bytes"
        return raw, None
    except Exception as exc:                                  # noqa: BLE001
        return None, type(exc).__name__


def scrape(target_per_airframe=120, airframe_keys=None, countries=None,
           dry_run=False):
    """Scrape up to `target_per_airframe` unique images for each airframe.

    Returns a per-airframe count of images on disk after the run.
    """
    RAW_ROOT.mkdir(parents=True, exist_ok=True)

    records, seen_urls = _load_manifest()
    index = DuplicateIndex()
    counts = {}
    for rec in records:
        counts[rec["airframe"]] = counts.get(rec["airframe"], 0) + 1
        # Rebuild the dup index from stored hashes so a resumed run keeps
        # rejecting duplicates of what earlier runs already kept.
        h = rec.get("dhash")
        if h:
            import numpy as np
            index.add(np.frombuffer(bytes.fromhex(h), dtype=np.uint8),
                      rec["filename"])
    if records:
        print(f"resuming: {len(records)} images already scraped "
              f"across {len(counts)} airframes\n")

    targets = [BY_KEY[k] for k in airframe_keys] if airframe_keys else AIRFRAMES
    if countries:
        wanted = {country.casefold() for country in countries}
        targets = [af for af in targets if af.country.casefold() in wanted]

    for af in targets:
        have = counts.get(af.key, 0)
        if have >= target_per_airframe:
            print(f"{af.key:22s} {af.country:12s} have {have}, skipping")
            continue

        out_dir = RAW_ROOT / "images" / af.key
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"{af.key:22s} {af.country:12s} have {have}, "
              f"want {target_per_airframe}")

        # Hand-written queries first, then generated variants ("<name> in
        # flight", "<name> takeoff"). The variants exist because a given query
        # string returns a largely fixed result set, so asking differently yields
        # more unique images than asking again - and the loop below stops as soon
        # as the target is met, so they cost nothing for well-documented
        # airframes that fill from their explicit queries alone.
        queries = all_queries(af)

        # Ask each query for more than the shortfall: most candidates are lost to
        # duplicates, dead links and thumbnails.
        per_query = max(30, int(2.5 * (target_per_airframe - have) / len(queries)))

        for query_number, query in enumerate(queries):
            if counts.get(af.key, 0) >= target_per_airframe:
                break
            print(f"   \"{query}\" -> asking for {per_query}")
            if dry_run:
                continue
            # Commons full-text results overlap heavily between generated query
            # variants and its API rate-limits repeated anonymous requests.
            # Query only the hand-written identity/alias queries; DDGS/Bing still
            # receive every generated in-flight/photo/takeoff variant.
            hits = _search(query, per_query,
                           use_commons=query_number < len(af.queries))

            # Filter first, then fetch the survivors concurrently. Hashing,
            # dedup and naming stay on this thread so filenames remain
            # deterministic and the duplicate index needs no locking.
            candidates = []
            for hit in hits:
                url = hit.get("image")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                w, h = hit.get("width") or 0, hit.get("height") or 0
                if w and h and min(w, h) < MIN_SIDE:
                    continue
                candidates.append((url, w, h, hit))

            kept = 0
            with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
                futures = {pool.submit(_fetch, url): (url, w, h, hit)
                           for url, w, h, hit in candidates}
                for future in as_completed(futures):
                    url, w, h, hit = futures[future]
                    if counts.get(af.key, 0) >= target_per_airframe:
                        continue      # drain the pool, keep nothing further
                    try:
                        raw, err = future.result()
                    except Exception as exc:                  # noqa: BLE001
                        raw, err = None, type(exc).__name__
                    if raw is None:
                        _append(REJECTS, {"airframe": af.key, "query": query,
                                          "image_url": url, "reason": err})
                        continue

                    hsh = dhash_bytes(raw)
                    if hsh is None:
                        _append(REJECTS, {"airframe": af.key, "query": query,
                                          "image_url": url,
                                          "reason": "undecodable"})
                        continue

                    clash = index.find(hsh)
                    if clash is not None:
                        # Cross-airframe clashes are the label-ambiguity signal;
                        # keep them in the reject log so an audit can surface
                        # them.
                        _append(REJECTS, {"airframe": af.key, "query": query,
                                          "image_url": url,
                                          "reason": "duplicate",
                                          "duplicate_of": clash})
                        continue

                    idx = counts.get(af.key, 0)
                    fname = f"{af.key}_{idx:04d}.jpg"
                    (out_dir / fname).write_bytes(raw)
                    index.add(hsh, fname)
                    counts[af.key] = idx + 1
                    kept += 1
                    _append(MANIFEST, {
                        "filename": fname,
                        "path": str((out_dir / fname).as_posix()),
                        "airframe": af.key,
                        "airframe_name": af.name,
                        "country": af.country,
                        "role": af.role,
                        "query": query,
                        "image_url": url,
                        "source_page": hit.get("url"),
                        "source_provider": hit.get("source_provider"),
                        "license": hit.get("license"),
                        "license_url": hit.get("license_url"),
                        "original_image_url": hit.get("original_image_url"),
                        "width": w,
                        "height": h,
                        "dhash": hsh.tobytes().hex(),
                    })
            print(f"      kept {kept} (total {counts.get(af.key, 0)})")
            time.sleep(random.uniform(*PER_QUERY_DELAY))

    print("\n=== scrape summary ===")
    total = 0
    for af in AIRFRAMES:
        n = counts.get(af.key, 0)
        total += n
        flag = "" if n >= target_per_airframe * 0.5 else "   <- thin"
        print(f"  {n:4d}  {af.key:22s} {af.country}{flag}")
    print(f"  {total} images across {len(counts)} airframes")
    return counts


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", type=int, default=120,
                   help="unique images to collect per airframe")
    p.add_argument("--airframes", nargs="*", default=None,
                   help="limit to these airframe keys")
    p.add_argument("--countries", nargs="*", default=None,
                   help="limit to these country names (use quotes for spaces)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the query plan without downloading")
    a = p.parse_args()
    scrape(target_per_airframe=a.target, airframe_keys=a.airframes,
           countries=a.countries,
           dry_run=a.dry_run)


if __name__ == "__main__":
    main()
