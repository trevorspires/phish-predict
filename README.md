# phish-predict

Predicting what Phish will play tonight.

Phish is close to a worst case for setlist prediction: a catalog of ~980
performed songs, ~15 of them played per night, chosen by a band that actively
avoids repetition — songs rotate on gaps of tens to hundreds of shows, tours
reset the rotation, and every fan "knows" a bustout is due without being able
to say when. That structure (gaps, rotations, eras, multi-night runs) is
exactly what a model can learn. This repo contains the whole pipeline: dataset
builder, models, a walk-forward backtest, and a daily agent that predicts each
show the morning of and scores itself against the real setlist the next day.

## Results

Frozen walk-forward protocol: fit strictly on shows before 2024, then walk the
107 shows from 2024-02-20 to 2026-07-12 with model parameters frozen (song
states keep updating). recall@k = of the model's top-k songs, the fraction of
tonight's ~15-song setlist it called.

|  | recall@10 | recall@20 | recall@30 |
|---|---|---|---|
| **v3** (v2 + 3-seed ensemble + hazard blend) | **0.224** | **0.383** | **0.492** |
| v2 (GBM + set-slot features) | 0.216 | 0.379 | 0.489 |
| v1 (GBM) | 0.206 | 0.370 | 0.487 |
| gap-hazard baseline | 0.158 | 0.273 | 0.360 |
| naive: rank by plays in last 100 shows | 0.091 | 0.194 | 0.296 |

Random guessing from the ~500-song active pool would land near 0.02 at k=10.
The naive frequency ranker gets half the model's score — most of the edge
comes from timing (when a song is *due*), not popularity.

It runs live: the model predicts every show of the current tour the morning of
and scores itself the next day (`predictions/scores.csv`). Summer 2026 so far:
12/20, 7/20, 3/20, and 4/20 songs called in the top-20 across four shows —
the two weak nights were predicted with days-stale history, which is what the
data-supplement step (below) now fixes.

## How it works

**Dataset** (`fetch.py` → `build_dataset.py`): every Phish show from 1983 to
the present — 2,100+ shows, ~40,000 song performances — normalized into three
CSVs (shows, performances, song catalog). Performances carry date, era, tour,
venue, set, position, original-vs-cover, and gap (shows since that song was
last played).

**Baseline** (`baseline.py`): a per-song gap-hazard model — P(played tonight |
shows since last played) estimated from each song's own gap history, shrunk
toward the global hazard curve, with recency-weighted observations so songs
that leave the rotation decay. Pure stdlib, no dependencies, and surprisingly
strong: this is the model to beat.

**Model** (`model.py`): a gradient-boosted classifier (scikit-learn
HistGradientBoostingClassifier) over 24 per-(show, song) features computed
strictly from pre-show history: gap and gap-vs-typical-gap, play counts over
25/100/300-show windows, play rate over the song's lifetime, played-last-show
and played-this-run flags, tour position, night-of-run, venue play counts,
day-of-week/month/year, original-vs-cover, and the song's historical set-slot
propensities (Set 1 / Set 2 / encore). Three additional slot heads organize
the top picks into a predicted Set 1 / Set 2 / encore shape. In v3 the main
head is fit three times with different seeds and the averaged probability is
geometrically blended with the gap-hazard baseline score (weight 0.08) — a
small gain that held up on both validation windows.

**Backtest discipline**: all iteration happens on a dev window (walk 2022–23,
fit pre-2022); the untouched 2024+ window is run once per final candidate.
That gate killed most of one full experiment round — negative results, all of
which looked good on dev: recency-weighted and era-truncated training (the
model wants all 40 years of data), calendar-time features like days-into-tour
and NYE/Halloween flags, stacking the hazard probability as a GBM *feature*
(redundant with the gap features — blending it into the final *score* is what
works), song co-occurrence PPMI/SVD embedding affinities (+.005 dev, −.022
holdout), a lower-LR retune (+.015 dev@20, zero holdout gain), and — from
earlier rounds — per-set union ranking and segue-partner boosts.

## The daily agent

`daily.py` runs every morning (launchd/cron):

1. Incremental data refresh from phish.in, plus a phish.net-scraped
   supplement for shows phish.in doesn't have yet (it only adds a show once a
   recording is uploaded, which can lag days mid-tour — and a model that
   can't see last night's setlist mispredicts tonight's).
2. Scores every past prediction whose setlist has since become available →
   `predictions/scores.csv` + a notification scorecard.
3. If phish.net lists a show tonight, generates the prediction
   (`predictions/<date>.{txt,json}`) and sends the picks.

`watch_live.py <date> <setlist-url>` polls the phish.net setlist page during
a show and sends each new song as a hit/miss against tonight's prediction in
real time.

Notifications go through any executable you point `PHISH_NOTIFY` at (it gets
the message as `$1`); without one they print to stdout.

## Run it yourself

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/) (or plain
`pip install numpy scikit-learn`).

```bash
python3 fetch.py            # build the local dataset from phish.in (~80 MB raw; be patient + polite)
python3 build_dataset.py    # normalize into data/*.csv
python3 baseline.py backtest
uv run model.py backtest                         # final-window eval (2024+)
uv run model.py backtest 2022-01-01 2024-01-01   # dev-window eval
uv run model.py predict 2026-07-18 "Merriweather Post"
```

## Data & licensing

**This repo ships the builder, not the data.** Setlist data is owned by The
Mockingbird Foundation and provided under the
[phish.net API Terms of Use](https://docs.phish.net/terms-of-use), which
allow non-commercial use with attribution but prohibit republishing the data
itself. So no dataset files are committed — `fetch.py` rebuilds everything
locally from the [phish.in](https://phish.in) public API (whose setlist data
is in turn sourced from phish.net), politely rate-limited and resumable.

**Data courtesy of Phish.net and The Mockingbird Foundation.** Audio-archive
metadata via phish.in. This project is free, non-commercial, and not
affiliated with, endorsed by, or sponsored by Phish, Phish.net, The
Mockingbird Foundation, or phish.in.

Code is MIT-licensed (see LICENSE — the license covers the code only, not any
data it fetches).
