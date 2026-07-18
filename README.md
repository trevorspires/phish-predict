# 🎸 phish-predict

> A robot that tries to guess the setlist before Phish plays it. Graded in
> public, every show, the morning after. 🐟

## 🔮 What is this?

Phish has played ~**980 different songs** across 2,100+ shows since 1983, and
plays ~15 of them on any given night — chosen by a band that famously never
repeats itself. Songs disappear for 100+ shows and come roaring back
(**bustouts** ⚡). Tours have rotations. Runs have rules. Every fan walking
into the lot *knows* tonight's the night for Fluffhead. This project is that
fan — except it has actually memorized all 2,100 setlists, and it puts its
prediction in writing before every show so you can laugh at it after. 🤖📝

Every show day it:

1. ☀️ wakes up, refreshes the data, and looks at everything the band has done
   up to last night
2. 🎯 ranks all ~980 songs by tonight's play probability and sketches a full
   predicted **Set 1 / Set 2 / encore**
3. 🌙 while the band plays, a live tracker scores each song as it lands —
   hit ✅ or miss ❌
4. 📈 next morning it grades itself against the real setlist and appends the
   result to a public scorecard (`predictions/scores.csv`) — wins *and*
   faceplants, forever

## 📊 How good is it?

On an average night it calls **~6 of the ~15 songs** in its top-20 picks.
Best night so far this tour: **12 of 20** (7/12 Deer Creek 🦌🔥). The formal
number: frozen walk-forward backtest — fit strictly on pre-2024 shows, walk
all 107 shows from Feb 2024 → July 2026 with parameters frozen — measuring
recall@k (what fraction of tonight's setlist was in the model's top k):

|  | recall@10 | recall@20 | recall@30 |
|---|---|---|---|
| **v3** — GBM + 3-seed ensemble + hazard blend | **0.224** | **0.383** | **0.492** |
| v2 — GBM + set-slot features | 0.216 | 0.379 | 0.489 |
| v1 — GBM | 0.206 | 0.370 | 0.487 |
| gap-hazard baseline | 0.158 | 0.273 | 0.360 |
| naive: most-played of the last 100 shows | 0.091 | 0.194 | 0.296 |

Guessing randomly from the active pool would land near 0.02. Ranking by pure
popularity gets half the model's score — the edge isn't knowing *what* Phish
plays, it's knowing **when a song is due**. ⏰

## ⚙️ How it works

**📦 The dataset** (`fetch.py` → `build_dataset.py`): every show since 1983 —
2,100+ shows, ~40,000 individual song performances — flattened into tidy
CSVs. Each performance row knows its date, era, tour, venue, set, position,
original-vs-cover, and **gap** (shows since that song was last played).

**🎲 The baseline** (`baseline.py`): a per-song *gap hazard* — P(played
tonight | shows since last played), estimated from each song's own rotation
history, shrunk toward the global curve, recency-weighted. Pure stdlib,
no dependencies, surprisingly hard to beat. Respect the baseline. 🫡

**🧠 The model** (`model.py`): scikit-learn gradient boosting over 24
features per (show, song), all computed strictly from pre-show history — gap
vs. the song's usual cadence, play rates over 25/100/300-show windows,
tour/run position, venue history, cover-vs-original, and each song's
historical Set 1 / Set 2 / encore tendencies. Slot heads arrange the top
picks into an actual setlist shape. v3 fits the main head three times with
different seeds, averages, and blends in the hazard score (weight 0.08).

**🔬 The discipline**: all experimentation happens on a 2022–23 dev window;
the 2024+ window gets touched once per final candidate. That gate has a body
count — the 🪦 **graveyard of ideas that looked great on dev and died on the
holdout**: recency-weighted training, era truncation, days-into-tour and
NYE/Halloween features, hazard-as-a-feature, song co-occurrence embeddings
(+.005 dev, −.022 holdout, ouch), a full hyperparameter retune, segue-partner
boosts, and per-set union ranking. Negative results are results. ⚰️

## 🤖 The daily agent

`daily.py` runs every morning via launchd/cron: refresh from
[phish.in](https://phish.in), supplement from phish.net for shows phish.in
hasn't ingested yet (it waits for audio uploads, which can lag days mid-tour
— and a model that can't see last night's show will confidently predict a
repeat 🤦), score yesterday's prediction, predict tonight's show, send the
picks. `watch_live.py` polls the setlist page during the show and calls each
song hit-or-miss in real time.

Notifications pipe through whatever executable you point `PHISH_NOTIFY` at;
with nothing configured they just print.

## 🚀 Run it yourself

Python 3.12+ and [uv](https://docs.astral.sh/uv/) (or
`pip install numpy scikit-learn`):

```bash
python3 fetch.py            # build the dataset from phish.in (~80 MB raw; go get a snack 🥨)
python3 build_dataset.py    # normalize into data/*.csv
python3 baseline.py backtest
uv run model.py backtest                         # final-window eval (2024+)
uv run model.py backtest 2022-01-01 2024-01-01   # dev-window eval
uv run model.py predict 2026-07-18 "Merriweather Post"
```

## 🗺️ Roadmap

- 🌐 a tiny microsite that posts each night's prediction before the show
- 🥇 next experiment round: multi-fold paired evaluation, a proper ranking
  objective (LambdaMART), model stacking, bustout-reentry features
- 📊 dataset CSVs in the repo, if the Mockingbird Foundation grants
  permission (asked nicely 🙏)

## 📜 Data & licensing

**This repo ships the builder, not the data.** Setlist data is owned by The
Mockingbird Foundation and provided under the
[phish.net API Terms of Use](https://docs.phish.net/terms-of-use), which
allow non-commercial use with attribution but prohibit republishing the data
itself. So no dataset files are committed — `fetch.py` rebuilds everything
locally from the [phish.in](https://phish.in) public API (whose setlist data
is in turn sourced from phish.net), politely rate-limited and resumable.

**Data courtesy of Phish.net and The Mockingbird Foundation.** 🙏
Audio-archive metadata via phish.in. This project is free, non-commercial,
and not affiliated with, endorsed by, or sponsored by Phish, Phish.net, The
Mockingbird Foundation, or phish.in.

Code is MIT-licensed (see LICENSE — the license covers the code only, not
any data it fetches).
