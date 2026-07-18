#!/usr/bin/env python3
"""Gradient-boosted setlist model with set structure.

Four GBM heads over a shared feature matrix: "any" ranks the song pool (this is
the scored prediction); the s1/s2/enc heads organize the top picks into a
predicted Set 1 / Set 2 / Encore. Slot-propensity features (how often a song
historically lands in each set) feed all heads. v3: the "any" head is a 3-seed
ensemble (averaged probabilities) geometrically blended with the gap-hazard
baseline score (weight 0.08) — small, consistent gain on both dev and final.

Tried and rejected on validation (see git history): ranking by the union of
per-set heads (encore positives too sparse); an explicit segue-partner boost
(the model already co-ranks partners via shared features; tuned alpha=0);
recency-weighted / era-truncated training (a wash — the model wants all 40
years); calendar-time features like days-into-tour and NYE flags (dev-only
noise); stacking the hazard as a *feature* (redundant with gap features);
song co-occurrence PPMI/SVD embedding affinities (dev gain, died on holdout);
and a lower-LR/smaller-trees retune (dev-window-specific, no final gain).

Usage:
  uv run model.py backtest                  # frozen-model walk-forward eval vs baseline
  uv run model.py predict <date> [venue]    # rank next show; writes predictions/

Protocol matches baseline.py: fit strictly before 2024, walk 2024+ with model
params frozen (song states keep updating).
"""
import bisect
import csv
import datetime
import json
import os
import sys
from collections import defaultdict

import numpy as np

import baseline

ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL_START = "2024-01-01"
TRAIN_SKIP = 150             # skip earliest shows: per-song states degenerate
SLOTS = ("s1", "s2", "enc")
HEADS = ("any",) + SLOTS   # "any" ranks the pool; slot heads organize the setlist
FEATURES = [
    "gap", "log_gap", "gap_ratio", "mean_gap", "plays_25", "plays_100",
    "plays_300", "log_total_plays", "log_age", "play_rate", "played_last",
    "played_prev3", "earlier_this_tour", "earlier_this_run", "night_of_run",
    "show_in_tour", "dow", "month", "year", "original", "venue_plays",
    "p_set1", "p_set2", "p_encore",
]


def slot_of(set_name):
    s = (set_name or "").lower()
    if "encore" in s:
        return "enc"
    return "s1" if s.strip() == "set 1" else "s2"


def load_meta():
    venue, tour = {}, {}
    with open(os.path.join(ROOT, "data", "shows.csv")) as f:
        for r in csv.DictReader(f):
            venue[r["date"]] = r["venue"] or ""
            tour[r["date"]] = r["tour"] or ""
    original = {}
    with open(os.path.join(ROOT, "data", "songs.csv")) as f:
        for r in csv.DictReader(f):
            original[r["slug"]] = int(r["original"])
    slots = defaultdict(set)  # (date, slug) -> {"s1","s2","enc"}
    with open(os.path.join(ROOT, "data", "performances.csv")) as f:
        for r in csv.DictReader(f):
            if r["song_slug"] not in baseline.NOT_SONGS:
                slots[(r["date"], r["song_slug"])].add(slot_of(r["set"]))
    return venue, tour, original, slots


class State:
    """Chronological per-song state; features are computed BEFORE ingesting a show."""

    def __init__(self, original):
        self.original = original
        self.last = {}
        self.play_idxs = defaultdict(list)
        self.gap_sum = defaultdict(float)
        self.gap_n = defaultdict(int)
        self.slot_counts = defaultdict(lambda: [0, 0, 0])   # slug -> [s1,s2,enc]
        self.venue_plays = defaultdict(int)
        self.run_venue = None
        self.night_of_run = 0
        self.run_counts = defaultdict(int)
        self.cur_tour = None
        self.show_in_tour = 0
        self.tour_counts = defaultdict(int)

    def context(self, date, venue, tour):
        night = self.night_of_run + 1 if venue and venue == self.run_venue else 1
        sit = self.show_in_tour + 1 if tour and tour == self.cur_tour else 1
        d = datetime.date.fromisoformat(date)
        return night, sit, d.weekday(), d.month, d.year

    def rows(self, s_idx, date, venue, tour):
        night, sit, dow, month, year = self.context(date, venue, tour)
        run_reset = venue != self.run_venue
        tour_reset = tour != self.cur_tour
        slugs = list(self.last)
        X = np.empty((len(slugs), len(FEATURES)), dtype=np.float32)
        for i, slug in enumerate(slugs):
            li = self.last[slug]
            gap = s_idx - li - 1
            plays = self.play_idxs[slug]
            n = len(plays)
            mg = self.gap_sum[slug] / self.gap_n[slug] if self.gap_n[slug] else gap or 1.0
            age = s_idx - plays[0]
            sc = self.slot_counts[slug]
            X[i] = (
                gap, np.log1p(gap), gap / max(mg, 0.5), mg,
                n - bisect.bisect_left(plays, s_idx - 25),
                n - bisect.bisect_left(plays, s_idx - 100),
                n - bisect.bisect_left(plays, s_idx - 300),
                np.log1p(n), np.log1p(age), n / max(age, 1),
                1.0 if gap == 0 else 0.0, 1.0 if gap <= 2 else 0.0,
                0 if tour_reset else self.tour_counts[slug],
                0 if run_reset else self.run_counts[slug],
                night, sit, dow, month, year,
                self.original.get(slug, 1),
                self.venue_plays[(venue, slug)],
                sc[0] / max(n, 1), sc[1] / max(n, 1), sc[2] / max(n, 1),
            )
        return slugs, X

    def ingest(self, s_idx, songs, venue, tour, slots, date):
        if venue == self.run_venue:
            self.night_of_run += 1
        else:
            self.run_venue, self.night_of_run = venue, 1
            self.run_counts.clear()
        if tour == self.cur_tour:
            self.show_in_tour += 1
        else:
            self.cur_tour, self.show_in_tour = tour, 1
            self.tour_counts.clear()
        for slug in songs:
            if slug in self.last:
                self.gap_sum[slug] += s_idx - self.last[slug] - 1
                self.gap_n[slug] += 1
            self.last[slug] = s_idx
            self.play_idxs[slug].append(s_idx)
            self.venue_plays[(venue, slug)] += 1
            self.run_counts[slug] += 1
            self.tour_counts[slug] += 1
            for j, sl in enumerate(SLOTS):
                if sl in slots.get((date, slug), ()):
                    self.slot_counts[slug][j] += 1

SEEDS = (7, 17, 27)   # the "any" head is a seed ensemble: fit per seed, average
HAZARD_W = 0.08       # geometric blend weight on the gap-hazard baseline score


def make_model(seed=7):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
        l2_regularization=1.0, early_stopping=True, random_state=seed)


def walk(shows, venue, tour, original, slots, cutoff):
    """Training matrices from shows[TRAIN_SKIP:cutoff] for the 3 slot heads.

    Also returns `groups`: rows emitted per show, in order — listwise/ranking
    objectives need per-show query groups.
    """
    st = State(original)
    Xs, ys, groups = [], {h: [] for h in HEADS}, []
    for idx, (date, songs) in enumerate(shows[:cutoff]):
        if idx >= TRAIN_SKIP:
            slugs, X = st.rows(idx, date, venue.get(date, ""), tour.get(date, ""))
            Xs.append(X)
            groups.append(len(slugs))
            played = set(songs)
            ys["any"].append(np.fromiter((s in played for s in slugs), dtype=np.float32))
            for sl in SLOTS:
                ys[sl].append(np.fromiter(
                    (sl in slots.get((date, s), ()) for s in slugs), dtype=np.float32))
        st.ingest(idx, songs, venue.get(date, ""), tour.get(date, ""), slots, date)
    return (np.concatenate(Xs), {h: np.concatenate(ys[h]) for h in HEADS}, st,
            np.array(groups))


def fit_heads(X, ys, only=HEADS, groups=None):
    """Slot heads are single models; the "any" head is a list, one fit per seed.

    `groups` (rows per show) is unused by the GBM heads but part of the
    contract so ranking-objective variants can consume it.
    """
    heads = {}
    for h in only:
        if h == "any":
            heads[h] = [make_model(s).fit(X, ys[h]) for s in SEEDS]
        else:
            heads[h] = make_model().fit(X, ys[h])
    return heads


def any_score(heads, X, haz_p):
    """Seed-averaged "any" probability, geometrically blended with the hazard."""
    p = np.mean([m.predict_proba(X)[:, 1] for m in heads["any"]], axis=0)
    return np.clip(p, 1e-9, 1) ** (1 - HAZARD_W) * np.clip(haz_p, 1e-9, 1) ** HAZARD_W


def union_score(heads, X, haz_p):
    per_slot = {sl: heads[sl].predict_proba(X)[:, 1] for sl in SLOTS}
    return any_score(heads, X, haz_p), per_slot


def eval_window(shows, venue, tour, slots, heads, st, start, end, glob, per):
    recalls = defaultdict(list)
    for idx in range(start, end):
        date, songs = shows[idx]
        v, t = venue.get(date, ""), tour.get(date, "")
        slugs, Xn = st.rows(idx, date, v, t)
        # st.last is current through idx-1 (ingest happens after scoring), so
        # this is the same frozen-hazard walk as baseline.backtest()
        hz = baseline.score_all(st.last, idx, glob, per)
        haz_p = np.fromiter((hz[s] for s in slugs), dtype=np.float64, count=len(slugs))
        order = np.argsort(-any_score(heads, Xn, haz_p))
        actual = set(songs)
        for k in (10, 20, 30):
            recalls[k].append(len({slugs[j] for j in order[:k]} & actual) / len(actual))
        st.ingest(idx, songs, v, t, slots, date)
    return {k: sum(v) / len(v) for k, v in recalls.items()}


def backtest(eval_start=EVAL_START, eval_end="9999"):
    """Fit strictly before eval_start; walk-forward eval on [eval_start, eval_end)."""
    shows = baseline.load_shows()
    venue, tour, original, slots = load_meta()
    eval_cut = next(i for i, (d, _) in enumerate(shows) if d >= eval_start)
    eval_stop = next((i for i, (d, _) in enumerate(shows) if d >= eval_end), len(shows))
    X, ys, st, groups = walk(shows, venue, tour, original, slots, eval_cut)
    print(f"train: {X.shape[0]:,} rows x {X.shape[1]} features", flush=True)
    heads = fit_heads(X, ys, only=("any",), groups=groups)
    glob, per = baseline.build_hazards(shows, eval_cut)   # frozen at eval cutoff
    r = eval_window(shows, venue, tour, slots, heads, st, eval_cut, eval_stop, glob, per)
    n = eval_stop - eval_cut
    print(f"\nwalk-forward eval on {n} shows ({shows[eval_cut][0]} .. {shows[eval_stop - 1][0]})")
    print(f"{'':16}{'recall@10':>10}{'recall@20':>10}{'recall@30':>10}")
    print(f"{'gbm v3':16}" + "".join(f"{r[k]:>10.3f}" for k in (10, 20, 30)))
    if eval_start == EVAL_START and eval_stop == len(shows):
        print(f"{'gbm v2':16}{0.216:>10.3f}{0.379:>10.3f}{0.489:>10.3f}")
        print(f"{'gap-hazard':16}{0.158:>10.3f}{0.273:>10.3f}{0.360:>10.3f}")


def resolve_venue(name):
    if not name:
        return ""
    norm = lambda s: "".join(c for c in s.lower() if c.isalnum())
    target = norm(name)
    for v in json.load(open(os.path.join(ROOT, "data", "raw", "venues.json"))):
        names = [v["name"]] + (v.get("other_names") or [])
        if any(norm(n) in target or target in norm(n) for n in names if n):
            return v["name"]
    return name


def predict(show_date, venue_hint=""):
    shows = baseline.load_shows()
    venue, tour, original, slots = load_meta()
    X, ys, st, groups = walk(shows, venue, tour, original, slots, len(shows))
    heads = fit_heads(X, ys, groups=groups)

    v = resolve_venue(venue_hint)
    last_date = shows[-1][0]
    t = tour.get(last_date, "")
    slugs, Xn = st.rows(len(shows), show_date, v, t)
    glob, per = baseline.build_hazards(shows, len(shows))
    hz = baseline.score_all(st.last, len(shows), glob, per)
    haz_p = np.fromiter((hz[s] for s in slugs), dtype=np.float64, count=len(slugs))
    final, per_slot = union_score(heads, Xn, haz_p)
    order = np.argsort(-final)

    titles = {r["slug"]: r["title"] for r in csv.DictReader(
        open(os.path.join(ROOT, "data", "songs.csv")))}
    model_ranked = []
    for rank, j in enumerate(order[:100], 1):
        slug = slugs[j]
        model_ranked.append({
            "rank": rank, "slug": slug, "title": titles.get(slug, slug),
            "p": round(float(final[j]), 4),
            "p_slots": {sl: round(float(per_slot[sl][j]), 4) for sl in SLOTS},
            "gap": len(shows) - st.last[slug] - 1,
            "last_played": shows[st.last[slug]][0],
            "plays": len(st.play_idxs[slug])})

    # structured setlist: allocate top union picks to their strongest slot
    setlist = {"s1": [], "s2": [], "enc": []}
    want = {"s1": 8, "s2": 7, "enc": 2}
    for r in model_ranked:
        open_slots = [sl for sl in SLOTS if len(setlist[sl]) < want[sl]]
        if not open_slots:
            break
        sl = max(open_slots, key=lambda s: r["p_slots"][s])
        setlist[sl].append(r["title"])

    base_ranked = baseline.rank_next(shows)
    pdir = os.path.join(ROOT, "predictions")
    os.makedirs(pdir, exist_ok=True)
    out = {"show_date": show_date, "history_through": last_date,
           "venue": v, "n_history_shows": len(shows), "setlist": setlist,
           "rankings": {"model": model_ranked, "baseline": base_ranked}}
    with open(os.path.join(pdir, f"{show_date}.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(pdir, f"{show_date}.txt"), "w") as f:
        f.write(f"model prediction for {show_date} at {v or '?'} "
                f"(history through {last_date})\n\n")
        f.write("predicted shape:\n")
        for sl, label in (("s1", "SET 1"), ("s2", "SET 2"), ("enc", "ENCORE")):
            f.write(f"  {label}: {', '.join(setlist[sl])}\n")
        f.write(f"\n{'#':>3} {'P':>7} {'gap':>4}  song  [baseline rank]\n")
        brank = {r["slug"]: r["rank"] for r in base_ranked}
        for r in model_ranked[:40]:
            f.write(f"{r['rank']:>3} {r['p']:>7.3f} {r['gap']:>4}  {r['title']}"
                    f"  [{brank.get(r['slug'], '-')}]\n")
    print(json.dumps({"setlist": setlist, "venue": v}, indent=1))
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "backtest"
    if cmd == "backtest":
        backtest(*sys.argv[2:4])
    elif cmd == "predict":
        predict(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
    else:
        sys.exit(__doc__)
