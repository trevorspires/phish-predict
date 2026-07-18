#!/usr/bin/env python3
"""Predictability-ceiling diagnostics on the final window (2024+).

Answers "how much headroom is left?" with cheating rankers:

  v3            the shipped model (reference)
  pool-oracle   told which songs appear elsewhere in the SAME TOUR (past or
                future shows, tonight excluded), ranks pool members first by
                v3 score — the upper bound on all pool/rotation modeling
  freq-oracle   ranks by total plays across the eval window incl. the future —
                the upper bound on popularity-shaped knowledge
  expected      mean over shows of (sum of v3's top-20 seed-avg probabilities)
                / setlist size — what a perfectly calibrated version of the
                current representation believes recall@20 should be

Usage: uv run oracle.py
"""
import os
from collections import defaultdict

import numpy as np

import baseline
import model

EVAL_START = "2024-01-01"


def main():
    shows = baseline.load_shows()
    venue, tour, original, slots = model.load_meta()
    cut = next(i for i, (d, _) in enumerate(shows) if d >= EVAL_START)

    # tour pool: date -> songs played in other shows of the same tour
    tour_shows = defaultdict(list)
    for idx, (date, songs) in enumerate(shows):
        t = tour.get(date, "")
        key = t if t else f"neighbors:{idx // 20}"   # fallback: 20-show blocks
        tour_shows[key].append((date, set(songs)))
    pool_of = {}
    for key, members in tour_shows.items():
        for date, _ in members:
            pool_of[date] = set().union(*(s for d, s in members if d != date)) \
                if len(members) > 1 else set()

    X, ys, st, groups = model.walk(shows, venue, tour, original, slots, cut)
    heads = model.fit_heads(X, ys, only=("any",), groups=groups)
    glob, per = baseline.build_hazards(shows, cut)

    recalls = defaultdict(lambda: defaultdict(list))
    expected20 = []
    pool_sizes = []
    for idx in range(cut, len(shows)):
        date, songs = shows[idx]
        v, t = venue.get(date, ""), tour.get(date, "")
        slugs, Xn = st.rows(idx, date, v, t)
        hz = baseline.score_all(st.last, idx, glob, per)
        haz = np.fromiter((hz[s] for s in slugs), dtype=np.float64, count=len(slugs))
        score = model.any_score(heads, Xn, haz)
        raw_p = np.mean([m.predict_proba(Xn)[:, 1] for m in heads["any"]], axis=0)
        actual = set(songs)

        pool = pool_of.get(date, set())
        pool_sizes.append(len(pool))
        in_pool = np.fromiter((s in pool for s in slugs), dtype=np.float64,
                              count=len(slugs))
        # future-inclusive play counts across the eval window
        fut = defaultdict(int)
        for _, ss in shows[cut:]:
            for s in ss:
                fut[s] += 1
        futcount = np.fromiter((fut[s] for s in slugs), dtype=np.float64,
                               count=len(slugs))

        for name, s in (("v3", score),
                        ("pool-oracle", in_pool * 1000 + score),
                        ("freq-oracle", futcount)):
            order = np.argsort(-s)
            for k in (10, 20, 30):
                recalls[name][k].append(
                    len({slugs[j] for j in order[:k]} & actual) / len(actual))

        top20 = np.argsort(-raw_p)[:20]
        expected20.append(raw_p[top20].sum() / len(actual))
        st.ingest(idx, songs, v, t, slots, date)

    n = len(recalls["v3"][20])
    print(f"{n} eval shows ({shows[cut][0]} .. {shows[-1][0]}), "
          f"median tour-pool size {int(np.median(pool_sizes))} songs\n")
    print(f"{'':14}{'recall@10':>10}{'recall@20':>10}{'recall@30':>10}")
    for name in ("v3", "pool-oracle", "freq-oracle"):
        print(f"{name:14}" + "".join(
            f"{np.mean(recalls[name][k]):>10.3f}" for k in (10, 20, 30)))
    print(f"\ncalibration: v3's own expected recall@20 = "
          f"{np.mean(expected20):.3f} vs actual {np.mean(recalls['v3'][20]):.3f}")


if __name__ == "__main__":
    main()
