#!/usr/bin/env python3
"""Fetch the complete Phish performance dataset from phish.in (public API v2).

Downloads every show (with full track listing), plus the song, venue, and tour
catalogs, into data/raw/. Resumable: already-downloaded shows are skipped, so
re-running after a failure (or next month, for new shows) only fetches what's new.

Usage: python3 fetch.py
"""
import json
import os
import sys
import time
import urllib.request

BASE = "https://phish.in/api/v2"
UA = "phish-dataset-builder/0.1 (trevspires@gmail.com; personal research)"
ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "raw")
DELAY = 0.15  # seconds between requests — be polite, it's a fan-run site


def get(path, retries=4):
    url = f"{BASE}{path}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": UA})
            return json.load(urllib.request.urlopen(req, timeout=30))
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** (attempt + 1)
            print(f"  retry {url} in {wait}s ({e})", flush=True)
            time.sleep(wait)


def paginated(resource, key):
    """Fetch every page of a list endpoint."""
    items, page = [], 1
    while True:
        d = get(f"/{resource}?per_page=500&page={page}")
        items.extend(d[key])
        if page >= d["total_pages"]:
            return items
        page += 1


def main():
    os.makedirs(os.path.join(RAW, "shows"), exist_ok=True)

    for resource in ("songs", "venues", "tours"):
        dest = os.path.join(RAW, f"{resource}.json")
        items = paginated(resource, resource)
        json.dump(items, open(dest, "w"))
        print(f"{resource}: {len(items)}", flush=True)

    show_index = paginated("shows", "shows")
    json.dump(show_index, open(os.path.join(RAW, "show-index.json"), "w"))
    print(f"show index: {len(show_index)}", flush=True)

    done = skipped = 0
    for show in sorted(show_index, key=lambda s: s["date"]):
        date = show["date"]
        dest = os.path.join(RAW, "shows", f"{date}.json")
        if os.path.exists(dest):
            skipped += 1
            continue
        detail = get(f"/shows/{date}")
        json.dump(detail, open(dest, "w"))
        done += 1
        if done % 50 == 0:
            print(f"  fetched {done} shows (+{skipped} cached), at {date}", flush=True)
        time.sleep(DELAY)

    print(f"DONE: {done} fetched, {skipped} already cached, {len(show_index)} total", flush=True)


if __name__ == "__main__":
    sys.exit(main())
