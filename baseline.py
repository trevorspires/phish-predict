#!/usr/bin/env python3
"""Gap-hazard baseline for Phish setlist prediction. Pure stdlib, fully local.

Model: for each song, P(played in the next show | shows since last played = g),
estimated from that song's own gap history, shrunk toward the global hazard curve
(so rarely-played songs borrow strength), with exponential recency weighting on
opportunities (half-life 150 shows) so songs that left the rotation decay.

Usage:
  python3 baseline.py backtest            # frozen-hazard evaluation on 2024+ shows
  python3 baseline.py predict             # rank songs for the next (unseen) show
"""
import csv
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
# phish.in pseudo-tracks that aren't songs ("Jam" stays: untitled jams are real slots)
NOT_SONGS = {"banter", "soundcheck", "narration", "interview", "intro"}
HALF_LIFE = 150      # shows; recency weight on hazard observations
SHRINK = 25.0        # pseudo-observations of the global curve per song
EVAL_START = "2024-01-01"


def bucket(g):
    """Exact for short gaps, log-ish buckets for long ones."""
    if g <= 12:
        return g
    for hi in (16, 24, 36, 54, 80, 120, 180, 270, 400):
        if g <= hi:
            return hi
    return 999


def load_shows():
    """[(date, [unique slugs in order])] for shows that have setlists."""
    by_date = defaultdict(list)
    with open(os.path.join(ROOT, "data", "performances.csv")) as f:
        for r in csv.DictReader(f):
            if r["song_slug"] in NOT_SONGS:
                continue
            if r["song_slug"] not in by_date[r["date"]]:
                by_date[r["date"]].append(r["song_slug"])
    return sorted(by_date.items())


def build_hazards(shows, upto):
    """Hazard tables from shows[:upto]. Returns (global {b:(num,den)}, per-song)."""
    glob = defaultdict(lambda: [0.0, 0.0])
    per = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    last = {}
    for idx in range(upto):
        date, songs = shows[idx]
        w = 0.5 ** ((upto - 1 - idx) / HALF_LIFE)
        played = set(songs)
        for slug, li in last.items():
            b = bucket(idx - li - 1)
            hit = 1.0 if slug in played else 0.0
            glob[b][0] += w * hit
            glob[b][1] += w
            cell = per[slug][b]
            cell[0] += w * hit
            cell[1] += w
        for slug in played:
            last[slug] = idx
    return glob, per


def score_all(last, next_idx, glob, per):
    """{slug: P(played at show next_idx)} for every debuted song."""
    out = {}
    for slug, li in last.items():
        b = bucket(next_idx - li - 1)
        gn, gd = glob.get(b, (0.0, 0.0))
        gh = gn / gd if gd else 0.0
        sn, sd = per[slug].get(b, (0.0, 0.0)) if slug in per else (0.0, 0.0)
        out[slug] = (sn + SHRINK * gh) / (sd + SHRINK)
    return out


def backtest(shows):
    cutoff = next(i for i, (d, _) in enumerate(shows) if d >= EVAL_START)
    glob, per = build_hazards(shows, cutoff)          # frozen before eval window
    freq_window = 100
    last = {}
    for idx in range(cutoff):
        for slug in shows[idx][1]:
            last[slug] = idx

    recalls = defaultdict(list)
    freq_recalls = defaultdict(list)
    for idx in range(cutoff, len(shows)):
        date, actual = shows[idx]
        actual = set(actual)
        scores = score_all(last, idx, glob, per)
        ranked = sorted(scores, key=scores.get, reverse=True)
        # naive comparison: rank by play count in the last `freq_window` shows
        counts = defaultdict(int)
        for j in range(max(0, idx - freq_window), idx):
            for slug in shows[j][1]:
                counts[slug] += 1
        freq_ranked = sorted(counts, key=counts.get, reverse=True)
        for k in (10, 20, 30):
            recalls[k].append(len(set(ranked[:k]) & actual) / len(actual))
            freq_recalls[k].append(len(set(freq_ranked[:k]) & actual) / len(actual))
        for slug in actual:
            last[slug] = idx

    n = len(recalls[10])
    avg_len = sum(len(set(s)) for _, s in shows[cutoff:]) / n
    print(f"backtest: {n} shows ({shows[cutoff][0]} .. {shows[-1][0]}), "
          f"avg {avg_len:.1f} unique songs/show")
    print(f"{'':14}{'recall@10':>10}{'recall@20':>10}{'recall@30':>10}")
    for name, r in (("gap-hazard", recalls), ("freq-100", freq_recalls)):
        print(f"{name:14}" + "".join(f"{sum(r[k])/n:>10.3f}" for k in (10, 20, 30)))


def rank_next(shows, top=100):
    """Ranked predictions for the show after shows[-1], as a list of dicts."""
    glob, per = build_hazards(shows, len(shows))
    last, plays = {}, defaultdict(int)
    for idx, (_, songs) in enumerate(shows):
        for slug in songs:
            last[slug] = idx
            plays[slug] += 1
    titles = {}
    with open(os.path.join(ROOT, "data", "songs.csv")) as f:
        for r in csv.DictReader(f):
            titles[r["slug"]] = r["title"]
    next_idx = len(shows)
    scores = score_all(last, next_idx, glob, per)
    ranked = sorted(scores, key=scores.get, reverse=True)[:top]
    return [{"rank": i, "slug": slug, "title": titles.get(slug, slug),
             "p": round(scores[slug], 4), "gap": next_idx - last[slug] - 1,
             "last_played": shows[last[slug]][0], "plays": plays[slug]}
            for i, slug in enumerate(ranked, 1)]


def predict(shows, show_date=None):
    """Print the top 40; if show_date is given, also write predictions/<date>.{json,txt}."""
    ranked = rank_next(shows)
    header = (f"prediction for {show_date or 'the show after ' + shows[-1][0]} "
              f"(history through {shows[-1][0]}, {len(shows)} shows)\n\n"
              f"{'#':>3} {'P(play)':>8} {'gap':>4} {'last played':>12} {'plays':>6}  song")
    lines = [f"{r['rank']:>3} {r['p']:>8.3f} {r['gap']:>4} {r['last_played']:>12} "
             f"{r['plays']:>6}  {r['title']}" for r in ranked[:40]]
    text = header + "\n" + "\n".join(lines)
    print(text)
    if show_date:
        pdir = os.path.join(ROOT, "predictions")
        os.makedirs(pdir, exist_ok=True)
        with open(os.path.join(pdir, f"{show_date}.txt"), "w") as f:
            f.write(text + "\n")
        with open(os.path.join(pdir, f"{show_date}.json"), "w") as f:
            json.dump({"show_date": show_date, "history_through": shows[-1][0],
                       "n_history_shows": len(shows), "ranked": ranked}, f, indent=1)
    return ranked


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "backtest"
    shows = load_shows()
    if cmd == "backtest":
        backtest(shows)
    elif cmd == "predict":
        predict(shows, sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        sys.exit(f"unknown command: {cmd}")
