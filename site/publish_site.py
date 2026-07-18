#!/usr/bin/env python3
"""Build data.json from predictions/ + scores.csv and publish the microsite.

Uploads site/index.html + data.json to s3://phish-trevorspires-com-site via the
scoped [phish-site] profile, then invalidates CloudFront. Called by daily.py
after every prediction/score cycle; safe to run any time.
"""
import csv
import datetime
import json
import os
import re

import boto3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.path.dirname(os.path.abspath(__file__))
BUCKET = "phish-trevorspires-com-site"
DIST = "EZ9RORMFAP452"


def clean_venue(v):
    """Trim the city/state/country tail that supplement's slug fallback leaves
    ("Enmarket Arena Savannah Ga Usa" -> "Enmarket Arena")."""
    v = re.sub(r"\s+\w+\s+[A-Z][a-z]\s+Usa$", "", v or "")
    return re.sub(r"\s+[A-Z][a-z]\s+Usa$", "", v)


def build_data():
    pdir = os.path.join(ROOT, "predictions")
    dates = sorted(f[:-5] for f in os.listdir(pdir)
                   if f.endswith(".json") and re.fullmatch(r"\d{4}-\d{2}-\d{2}", f[:-5]))
    preds = {d: json.load(open(os.path.join(pdir, f"{d}.json"))) for d in dates}

    latest = dates[-1]
    p = preds[latest]
    ranked = p["rankings"]["model"] if "rankings" in p else p["ranked"]
    nxt = {
        "date": latest,
        "venue": clean_venue(p.get("venue", "")),
        "history_through": p.get("history_through", ""),
        "setlist": p.get("setlist", {}),
        "top": [{"rank": r["rank"], "title": r["title"], "p": r["p"],
                 "gap": r["gap"], "last_played": r["last_played"]}
                for r in ranked[:40]],
    }

    history = []
    spath = os.path.join(ROOT, "predictions", "scores.csv")
    if os.path.exists(spath):
        for r in csv.DictReader(open(spath)):
            pr = preds.get(r["date"], {})
            history.append({
                "date": r["date"], "venue": clean_venue(pr.get("venue", "")),
                "n_songs": int(r["n_songs"]),
                "h10": int(r["model_h10"] or 0), "h20": int(r["model_h20"] or 0),
                "h30": int(r["model_h30"] or 0),
                "b20": int(r["base_h20"] or 0) if r.get("base_h20") else None,
                "called": [s for s in (r.get("model_called") or "").split("; ") if s],
            })
    history.sort(key=lambda h: h["date"], reverse=True)

    return {
        "updated": datetime.datetime.now(datetime.timezone.utc)
                   .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "backtest": {"r10": 0.224, "r20": 0.383, "r30": 0.491,
                     "window": "107 shows, Feb 2024 – Jul 2026"},
        "next": nxt,
        "history": history,
    }


def main():
    data = build_data()
    dpath = os.path.join(SITE, "data.json")
    json.dump(data, open(dpath, "w"))

    s3 = boto3.Session(profile_name="phish-site").client("s3")
    s3.upload_file(os.path.join(SITE, "index.html"), BUCKET, "index.html",
                   ExtraArgs={"ContentType": "text/html; charset=utf-8",
                              "CacheControl": "max-age=300"})
    s3.upload_file(dpath, BUCKET, "data.json",
                   ExtraArgs={"ContentType": "application/json",
                              "CacheControl": "max-age=120"})
    cf = boto3.Session(profile_name="phish-site").client("cloudfront")
    cf.create_invalidation(DistributionId=DIST, InvalidationBatch={
        "CallerReference": data["updated"] + "-pub",
        "Paths": {"Quantity": 2, "Items": ["/index.html", "/data.json"]}})
    print(f"published: next={data['next']['date']} "
          f"history={len(data['history'])} shows")


if __name__ == "__main__":
    main()
