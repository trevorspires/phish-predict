#!/usr/bin/env python3
"""Normalize raw phish.in JSON into flat, analysis-ready CSVs in data/.

Outputs:
  shows.csv        one row per show (date, tour, venue, location, era, ...)
  performances.csv one row per song performance (the core ML table):
                   date, set, position, song slug/title, original-vs-cover,
                   duration, and the song's computed gap (shows since last played)
  songs.csv        song catalog with career-wide play counts and debut date

Run after fetch.py. Pure stdlib.
"""
import csv
import json
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data")


def era(date):
    """Phish's canonical eras: 1.0 (83-00), 2.0 (02-04), 3.0 (09-20), 4.0 (21-)."""
    y = int(date[:4])
    if y <= 2000:
        return "1.0"
    if y <= 2004:
        return "2.0"
    if y <= 2020:
        return "3.0"
    return "4.0"


def main():
    show_files = sorted(os.listdir(os.path.join(RAW, "shows")))
    shows = [json.load(open(os.path.join(RAW, "shows", f))) for f in show_files]
    shows.sort(key=lambda s: s["date"])

    with open(os.path.join(OUT, "shows.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "era", "tour", "venue", "city", "state", "country",
                    "venue_lat", "venue_lon", "duration_ms", "audio_status", "n_tracks"])
        for s in shows:
            v = s.get("venue") or {}
            w.writerow([s["date"], era(s["date"]), s.get("tour_name"), v.get("name"),
                        v.get("city"), v.get("state"), v.get("country"),
                        v.get("latitude"), v.get("longitude"),
                        s.get("duration"), s.get("audio_status"), len(s.get("tracks") or [])])

    # Performances + running gap computation (shows since the song was last played,
    # counted over shows we have setlists for; 0 = played previous show too)
    last_played_idx = {}
    play_count = defaultdict(int)
    debut = {}
    rows = []
    show_idx = 0
    for s in shows:
        tracks = s.get("tracks") or []
        if not tracks:
            continue
        show_idx += 1
        for t in tracks:
            for song in t.get("songs") or []:
                slug = song["slug"]
                if slug not in last_played_idx:
                    gap = ""  # debut
                else:
                    # repeat within the same show (e.g. a Tweezer sandwich) -> 0
                    gap = max(0, show_idx - last_played_idx[slug] - 1)
                rows.append([s["date"], era(s["date"]), s.get("tour_name"),
                             (s.get("venue") or {}).get("name"),
                             t.get("set_name"), t.get("position"), slug, song.get("title"),
                             1 if song.get("original") else 0, song.get("artist") or "",
                             t.get("duration"), gap])
                last_played_idx[slug] = show_idx
                play_count[slug] += 1
                debut.setdefault(slug, s["date"])

    with open(os.path.join(OUT, "performances.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "era", "tour", "venue", "set", "position", "song_slug",
                    "song_title", "original", "cover_artist", "duration_ms", "gap"])
        w.writerows(rows)

    songs = json.load(open(os.path.join(RAW, "songs.json")))
    with open(os.path.join(OUT, "songs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slug", "title", "original", "cover_artist", "times_played", "debut"])
        for s in sorted(songs, key=lambda x: -play_count.get(x["slug"], 0)):
            w.writerow([s["slug"], s["title"], 1 if s.get("original") else 0,
                        s.get("artist") or "", play_count.get(s["slug"], 0),
                        debut.get(s["slug"], "")])

    n_with_setlist = sum(1 for s in shows if s.get("tracks"))
    print(f"shows: {len(shows)} ({n_with_setlist} with setlists)")
    print(f"performances: {len(rows)}")
    print(f"distinct songs performed: {len(play_count)}")
    print(f"date range: {shows[0]['date']} .. {shows[-1]['date']}")


if __name__ == "__main__":
    main()
