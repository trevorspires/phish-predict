#!/usr/bin/env python3
"""Multi-fold paired walk-forward evaluation harness.

Four folds, each fit strictly on shows before the fold start, walked with
model params frozen. Per-show recalls are recorded so candidates compare
against the base PAIRED (per-show deltas), which cuts variance enormously
versus comparing window means.

Contract with model.py (candidates may change internals, not signatures):
  walk(shows, venue, tour, original, slots, cutoff) -> X, ys, st, groups
  fit_heads(X, ys, only=..., groups=...)            -> heads
  any_score(heads, X, haz_p)                        -> per-row score

Usage:
  uv run expeval.py --write-base    # once, from the untouched base model
  uv run expeval.py                 # candidate: per-fold + overall paired stats
"""
import json
import os
import sys

import numpy as np

import baseline
import model

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(ROOT, "base_folds.json")
FOLDS = [("2016-01-01", "2018-01-01"), ("2018-01-01", "2020-01-01"),
         ("2020-01-01", "2022-01-01"), ("2022-01-01", "2024-01-01")]


def run_fold(shows, venue, tour, original, slots, start, end):
    cut = next(i for i, (d, _) in enumerate(shows) if d >= start)
    stop = next((i for i, (d, _) in enumerate(shows) if d >= end), len(shows))
    X, ys, st, groups = model.walk(shows, venue, tour, original, slots, cut)
    heads = model.fit_heads(X, ys, only=("any",), groups=groups)
    glob, per = baseline.build_hazards(shows, cut)
    out = []
    for idx in range(cut, stop):
        date, songs = shows[idx]
        v, t = venue.get(date, ""), tour.get(date, "")
        slugs, Xn = st.rows(idx, date, v, t)
        hz = baseline.score_all(st.last, idx, glob, per)
        haz = np.fromiter((hz[s] for s in slugs), dtype=np.float64, count=len(slugs))
        order = np.argsort(-model.any_score(heads, Xn, haz))
        actual = set(songs)
        rec = {str(k): len({slugs[j] for j in order[:k]} & actual) / len(actual)
               for k in (10, 20, 30)}
        out.append({"date": date, **rec})
        st.ingest(idx, songs, v, t, slots, date)
    return out


def main():
    shows = baseline.load_shows()
    venue, tour, original, slots = model.load_meta()
    results = []
    for start, end in FOLDS:
        fold = run_fold(shows, venue, tour, original, slots, start, end)
        results.append(fold)
        r20 = np.mean([s["20"] for s in fold])
        print(f"fold {start[:4]}-{int(end[:4]) - 1}: {len(fold)} shows  "
              f"recall@20 {r20:.3f}", flush=True)

    if "--write-base" in sys.argv:
        json.dump(results, open(BASE, "w"))
        print(f"wrote {BASE}")
        return

    base = json.load(open(BASE))
    print(f"\npaired vs base ({sum(len(f) for f in base)} shows):")
    print(f"{'fold':12}{'n':>5}{'d@10':>8}{'d@20':>8}{'d@30':>8}")
    alld = {k: [] for k in ("10", "20", "30")}
    for (start, end), bf, cf in zip(FOLDS, base, results):
        bd = {s["date"]: s for s in bf}
        ds = {k: [c[k] - bd[c["date"]][k] for c in cf if c["date"] in bd]
              for k in ("10", "20", "30")}
        for k in alld:
            alld[k].extend(ds[k])
        print(f"{start[:4]}-{int(end[:4]) - 1:4}{len(ds['20']):>5}"
              + "".join(f"{np.mean(ds[k]):>+8.4f}" for k in ("10", "20", "30")))
    print(f"{'OVERALL':12}{len(alld['20']):>5}"
          + "".join(f"{np.mean(alld[k]):>+8.4f}" for k in ("10", "20", "30")))
    d = np.array(alld["20"])
    se = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / se if se else 0.0
    wins = int((d > 0).sum())
    losses = int((d < 0).sum())
    print(f"\nrecall@20 paired: mean {d.mean():+.4f}  SE {se:.4f}  t {t:+.2f}  "
          f"wins/losses {wins}/{losses}")
    print("verdict hint: |t| >= 2 with positive mean = real; otherwise noise")


if __name__ == "__main__":
    main()
