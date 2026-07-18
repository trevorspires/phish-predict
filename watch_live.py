#!/usr/bin/env python3
"""Live show tracker: poll the phish.net setlist page during a show and
Telegram each new song as a hit/miss against tonight's prediction.

Usage: python3 watch_live.py <date> <phish.net setlist url>

Polls every 3 minutes. State in predictions/.live-<date>.json so restarts
don't re-send. Exits when 75 min pass with no new songs (show over) or at
2am. Sends a final tally on exit.
"""
import datetime
import html
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
# any executable taking the message as $1; falls back to stdout if absent
NOTIFY = os.environ.get("PHISH_NOTIFY", os.path.expanduser("~/.claude/lib/notify.sh"))
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
POLL = 180
IDLE_STOP = 75 * 60


def norm(t):
    return "".join(c for c in t.lower() if c.isalnum())


def notify(msg):
    if os.path.exists(NOTIFY):
        subprocess.run([NOTIFY, msg], check=False)
    else:
        print(msg, flush=True)


def fetch_setlist(url):
    """Ordered [(set_label, title)] fetched from a phish.net setlist page."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    h = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
    return parse_setlist(h)


def parse_setlist(h):
    """Ordered [(set_label, title)] parsed from phish.net setlist page HTML."""
    m = re.search(r'class=.setlist-body.(.*?)(?:</p>\s*)?</div>', h, re.S)
    if not m:
        return []
    body = m.group(1)
    out, current = [], "?"
    # walk through set markers and song links in document order
    for tag, text in re.findall(r"<(a|span)[^>]*>(.*?)</\1>", body, re.S):
        text = html.unescape(re.sub("<[^>]+>", "", text)).strip()
        if not text:
            continue
        if re.match(r"(?i)^(set\s*\d|encore)", text):
            current = text.upper().rstrip(":").strip()
        elif tag == "a" and len(text) > 1 and not text.startswith("["):
            out.append((current, text))
    return out


def main():
    date, url = sys.argv[1], sys.argv[2]
    pred = json.load(open(os.path.join(ROOT, "predictions", f"{date}.json")))
    where = pred.get("venue") or date
    ranks = {}
    for name, ranked in pred["rankings"].items():
        for r in ranked:
            ranks.setdefault(norm(r["title"]), {})[name] = r["rank"]

    state_path = os.path.join(ROOT, "predictions", f".live-{date}.json")
    seen = json.load(open(state_path)) if os.path.exists(state_path) else []
    last_new = time.time()

    while True:
        try:
            setlist = fetch_setlist(url)
        except Exception as e:
            print(f"fetch error: {e}", flush=True)
            setlist = []
        new = [(s, t) for s, t in setlist if [s, t] not in seen and (s, t) not in
               [tuple(x) for x in seen]]
        if new:
            last_new = time.time()
            lines = []
            for st_label, title in new:
                r = ranks.get(norm(title))
                if r and "model" in r and r["model"] <= 30:
                    lines.append(f"✅ {title} · model #{r['model']}"
                                 + (f" · base #{r['baseline']}" if r.get("baseline", 999) <= 30 else ""))
                elif r:
                    lines.append(f"➖ {title} · model #{r.get('model', '−')}")
                else:
                    lines.append(f"❌ {title} · unranked")
            played = seen + [list(x) for x in new]
            hits20 = sum(1 for _, t in [tuple(x) for x in played]
                         if ranks.get(norm(t), {}).get("model", 999) <= 20)
            msg = (f"🎸 {where} live · {new[0][0].title()}\n" + "\n".join(lines)
                   + f"\n📊 {hits20} of top-20 called · {len(played)} songs in")
            notify(msg)
            print(msg, flush=True)
            seen = played
            json.dump(seen, open(state_path, "w"))
        now = datetime.datetime.now()
        if time.time() - last_new > IDLE_STOP or now.hour >= 2 and now.hour < 6:
            break
        time.sleep(POLL)

    n = len(seen)
    uniq = {norm(t) for _, t in [tuple(x) for x in seen]}
    m20 = sum(1 for t in uniq if ranks.get(t, {}).get("model", 999) <= 20)
    b20 = sum(1 for t in uniq if ranks.get(t, {}).get("baseline", 999) <= 20)
    notify(f"🏁 {where} final · {n} songs\n"
           f"🤖 Model: {m20} of top 20 hit\n"
           f"📉 Baseline: {b20} of top 20 hit\n"
           f"👉 official scorecard tomorrow 9am")
    print("done", flush=True)


if __name__ == "__main__":
    main()
