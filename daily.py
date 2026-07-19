#!/usr/bin/env python3
"""Morning show-day agent: refresh data, score past predictions, predict tonight.

Runs daily via launchd (com.trev.phish-predict, 9:00am). All modeling is local;
network is only the phish.in refresh and a phish.net upcoming-shows check.

1. fetch.py (incremental) + build_dataset.py
2. Supplement the CSVs with phish.net-scraped setlists for shows phish.in
   doesn't have yet (it lags until a recording is uploaded)
3. Score every unscored predictions/<date>.json — from phish.in if the show
   landed, else scraped from phish.net -> scores.csv + Telegram scorecard
4. If phish.net lists a show today -> model predict -> Telegram picks
"""
import csv
import datetime
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import baseline

# notifications retired in favor of the site (phishpredict.com); set
# PHISH_NOTIFY to an executable taking the message as $1 to re-enable
NOTIFY = os.environ.get("PHISH_NOTIFY", "")
SCORES = os.path.join(ROOT, "predictions", "scores.csv")
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
UV = os.path.expanduser("~/.local/bin/uv")


def notify(msg):
    if os.path.exists(NOTIFY):
        subprocess.run([NOTIFY, msg], check=False)
    else:
        print(msg, flush=True)


def refresh():
    for script in ("fetch.py", "build_dataset.py"):
        r = subprocess.run([sys.executable, os.path.join(ROOT, script)],
                           capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            raise RuntimeError(f"{script} failed:\n{r.stderr[-2000:]}")
        print(f"{script}: {r.stdout.strip().splitlines()[-1]}", flush=True)


def actual_setlist(date):
    """Unique song slugs for a show, or None if not yet on phish.in."""
    path = os.path.join(ROOT, "data", "raw", "shows", f"{date}.json")
    if not os.path.exists(path):
        return None
    tracks = json.load(open(path)).get("tracks") or []
    slugs = []
    for t in tracks:
        for s in t.get("songs") or []:
            if s["slug"] not in baseline.NOT_SONGS and s["slug"] not in slugs:
                slugs.append(s["slug"])
    return slugs or None


def looks_complete(songs):
    """True if a scraped setlist looks finished (encore up, or clearly long).

    Guards the night runs: a west-coast show is mid-set at 00:45 ET, and a
    partial setlist must never be scored or supplemented as final.
    """
    return any("ENCORE" in s.upper() for s, _ in songs) or len(songs) >= 14


def phishnet_setlist(date, require_complete=False):
    """Fallback: song slugs scraped from the phish.net setlist page, or None.

    phish.in only adds a show once a recording is uploaded (can lag days);
    phish.net has the setlist the same night. Titles map to slugs via songs.csv;
    unknown titles (debuts) keep a pseudo-slug so they count in the denominator.
    """
    import watch_live
    try:
        songs = watch_live.fetch_setlist(f"https://phish.net/setlists/?d={date}")
    except Exception as e:
        print(f"score: phish.net fetch failed for {date}: {e}", flush=True)
        return None
    if require_complete and songs and not looks_complete(songs):
        print(f"score: {date} setlist looks partial, waiting", flush=True)
        return None
    title2slug = {watch_live.norm(r["title"]): r["slug"] for r in
                  csv.DictReader(open(os.path.join(ROOT, "data", "songs.csv")))}
    slugs = []
    for _set, title in songs:
        slug = title2slug.get(watch_live.norm(title), f"?{watch_live.norm(title)}")
        if slug not in baseline.NOT_SONGS and slug not in slugs:
            slugs.append(slug)
    return slugs or None


def supplement_recent():
    """Append phish.net-scraped setlists for recent shows phish.in lacks.

    phish.in adds a show only once a recording is uploaded (can lag days),
    which starves predictions of the freshest history mid-tour. Transient by
    design: build_dataset.py rewrites the CSVs from raw each morning, so
    supplements re-apply daily until phish.in catches up.
    """
    import watch_live
    shows_csv = os.path.join(ROOT, "data", "shows.csv")
    perf_csv = os.path.join(ROOT, "data", "performances.csv")
    with open(shows_csv) as f:
        show_rows = list(csv.DictReader(f))
    last = max(show_rows, key=lambda r: r["date"])
    songs_cat = list(csv.DictReader(open(os.path.join(ROOT, "data", "songs.csv"))))
    title2slug = {watch_live.norm(r["title"]): r["slug"] for r in songs_cat}
    slug2orig = {r["slug"]: r["original"] for r in songs_cat}
    venues = json.load(open(os.path.join(ROOT, "data", "raw", "venues.json")))
    norm = watch_live.norm

    def resolve_venue(name):
        target = norm(name)
        for v in venues:
            for n in [v["name"]] + (v.get("other_names") or []):
                if n and (norm(n) in target or target in norm(n)):
                    return v["name"]
        return name   # phish.net's own venue name — already clean

    d = datetime.date.fromisoformat(last["date"]) + datetime.timedelta(days=1)
    today = datetime.date.today()
    while d < today:
        date, dd = d.isoformat(), d
        d += datetime.timedelta(days=1)
        try:
            req = urllib.request.Request(f"https://phish.net/setlists/?d={date}",
                                         headers={"User-Agent": UA})
            resp = urllib.request.urlopen(req, timeout=30)
            final_url = resp.geturl()
            page = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code != 404:      # 404 = no show that night
                print(f"supplement: {date} fetch failed: {e}", flush=True)
            continue
        except Exception as e:
            print(f"supplement: {date} fetch failed: {e}", flush=True)
            continue
        # no-show dates redirect to a listing page, not a dated permalink
        stamp = f"-{dd.strftime('%B').lower()}-{dd.day}-{dd.year}-"
        if stamp not in final_url:
            continue
        songs = watch_live.parse_setlist(page)
        if not songs or not looks_complete(songs):
            if songs:
                print(f"supplement: {date} setlist looks partial, waiting", flush=True)
            continue
        m = re.search(r'href="/venue/\d+/([^"]+)"', page)
        pn_venue = urllib.parse.unquote(m.group(1)).replace("_", " ").strip() if m else ""
        if not pn_venue:
            m = re.search(rf"setlists/phish{stamp}([a-z0-9-]+)\.html", final_url)
            pn_venue = (m.group(1) if m else "").replace("-", " ").title()
        venue = resolve_venue(pn_venue)
        with open(shows_csv, "a", newline="") as f:
            csv.writer(f).writerow([date, last["era"], last["tour"], venue,
                                    "", "", "", "", "", "", "supplemented", len(songs)])
        with open(perf_csv, "a", newline="") as f:
            w = csv.writer(f)
            for pos, (set_label, title) in enumerate(songs, 1):
                slug = title2slug.get(norm(title), norm(title))
                w.writerow([date, last["era"], last["tour"], venue,
                            set_label.title() if set_label != "?" else "Set 1",
                            pos, slug, title, slug2orig.get(slug, "1"), "", 0, ""])
        print(f"supplement: added {date} at {venue} ({len(songs)} songs) from phish.net",
              flush=True)


def rankings_of(pred):
    """{name: ranked list} from a prediction json (v2 or legacy v1)."""
    if "rankings" in pred:
        return pred["rankings"]
    return {"baseline": pred["ranked"]}  # legacy single-model file


def score_pending():
    scored = set()
    if os.path.exists(SCORES):
        scored = {r["date"] for r in csv.DictReader(open(SCORES))}
    pdir = os.path.join(ROOT, "predictions")
    pending = sorted(f[:-5] for f in os.listdir(pdir)
                     if f.endswith(".json") and f[:-5] not in scored
                     and re.fullmatch(r"\d{4}-\d{2}-\d{2}", f[:-5]))
    today = datetime.date.today().isoformat()
    for date in pending:
        actual = actual_setlist(date)
        if not actual and date < today:
            actual = phishnet_setlist(date, require_complete=True)
        if not actual:
            print(f"score: {date} not on phish.in yet", flush=True)
            continue
        pred = json.load(open(os.path.join(pdir, f"{date}.json")))
        aset = set(actual)
        hits, called = {}, {}
        for name, ranked in rankings_of(pred).items():
            slugs = [r["slug"] for r in ranked]
            for k in (10, 20, 30):
                hits[(name, k)] = len(set(slugs[:k]) & aset)
            called[name] = [r["title"] for r in ranked[:20] if r["slug"] in aset]

        new = not os.path.exists(SCORES)
        with open(SCORES, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["date", "n_songs",
                            "model_h10", "model_h20", "model_h30",
                            "base_h10", "base_h20", "base_h30", "model_called"])
            g = lambda n, k: hits.get((n, k), "")
            w.writerow([date, len(aset),
                        g("model", 10), g("model", 20), g("model", 30),
                        g("baseline", 10), g("baseline", 20), g("baseline", 30),
                        "; ".join(called.get("model", called.get("baseline", [])))])

        d = datetime.date.fromisoformat(date).strftime("%b %-d")
        lines = [f"🎯 Phish scorecard · {d}", f"🎸 {len(aset)} songs played"]
        if ("model", 20) in hits:
            lines.append(f"🤖 Model: {g('model',10)}/10 · {g('model',20)}/20 · "
                         f"{g('model',30)}/30 (backtest avg 6.6 of 20)")
        if ("baseline", 20) in hits:
            lines.append(f"📉 Baseline: {g('baseline',10)}/10 · {g('baseline',20)}/20 · "
                         f"{g('baseline',30)}/30 (avg 4.9 of 20)")
        best = called.get("model", called.get("baseline", []))
        lines.append(f"✅ Called: {', '.join(best) if best else 'none in top 20'}")
        notify("\n".join(lines))
        print(f"score: {date} model={g('model',20)}/20 base={g('baseline',20)}/20", flush=True)


def todays_show():
    """(date, venue) if phish.net lists a show today, else None."""
    today = datetime.date.today().isoformat()
    req = urllib.request.Request("https://phish.net/upcoming",
                                 headers={"User-Agent": UA})
    html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
    pat = rf'{today}" class="anchor" href="/setlists/phish-[a-z]+-\d+-\d+-([a-z0-9-]+)\.html'
    m = re.search(pat, html)
    if not m:
        return None
    venue = m.group(1).replace("-", " ").title()
    return today, venue


def run_model_predict(date, venue):
    r = subprocess.run([UV, "run", "model.py", "predict", date, venue],
                       cwd=ROOT, capture_output=True, text=True, timeout=1200)
    if r.returncode != 0:
        raise RuntimeError(f"model.py predict failed:\n{r.stderr[-2000:]}")
    return json.load(open(os.path.join(ROOT, "predictions", f"{date}.json")))


def predict_today():
    show = todays_show()
    if not show:
        print("no show today", flush=True)
        return
    date, venue = show
    latest = max(r["date"] for r in
                 csv.DictReader(open(os.path.join(ROOT, "data", "shows.csv"))))
    pj = os.path.join(ROOT, "predictions", f"{date}.json")
    updated = False
    if os.path.exists(pj):
        if json.load(open(pj)).get("history_through", "") >= latest:
            print(f"already predicted {date} (fresh through {latest})", flush=True)
            return
        updated = True   # new setlist data landed since — re-predict
    pred = run_model_predict(date, venue)
    ranked = pred["rankings"]["model"]
    sl = pred.get("setlist", {})
    d = datetime.date.fromisoformat(date).strftime("%b %-d")
    watch = " · ".join(f"{r['title']} (gap {r['gap']})"
                       for r in ranked[:25] if r["gap"] >= 40)[:200]
    tag = " (updated)" if updated else ""
    msg = f"🎸 Phish tonight{tag} · {pred.get('venue') or venue} · {d}\n"
    if sl:
        msg += (f"1️⃣ {' · '.join(sl['s1'])}\n"
                f"2️⃣ {' · '.join(sl['s2'])}\n"
                f"🎤 {' · '.join(sl['enc'])}\n")
    else:
        msg += f"🔮 {' · '.join(r['title'] for r in ranked[:10])}\n"
    if watch:
        msg += f"👀 Bustout watch: {watch}\n"
    msg += "👉 full top 40 in phish-predict/predictions/" + date + ".txt"
    notify(msg)


def publish_site():
    """Rebuild data.json and push the microsite. Non-fatal on failure."""
    r = subprocess.run([UV, "run", os.path.join("site", "publish_site.py")],
                       cwd=ROOT, capture_output=True, text=True, timeout=600)
    if r.returncode == 0:
        print(f"site: {r.stdout.strip().splitlines()[-1]}", flush=True)
    else:
        print(f"site publish failed:\n{r.stderr[-1000:]}", flush=True)


def main():
    refresh()
    supplement_recent()
    score_pending()
    predict_today()
    publish_site()


if __name__ == "__main__":
    main()
