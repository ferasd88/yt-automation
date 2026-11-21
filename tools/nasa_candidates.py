"""Search the NASA Image and Video Library (public domain) and build contact sheets for review.

Usage: python tools/nasa_candidates.py space/queries.json out/candidates
Writes <out>/<key>.jpg (numbered contact sheet) and <out>/candidates.json (nasa_id, title, center, size).
"""
import io
import json
import os
import sys
import urllib.parse
import urllib.request

from PIL import Image, ImageDraw, ImageFont

API = "https://images-api.nasa.gov"
queries = json.load(open(sys.argv[1], encoding="utf-8"))
out = sys.argv[2]
os.makedirs(out, exist_ok=True)
PER = 8


def get(url, timeout=40):
    req = urllib.request.Request(url, headers={"User-Agent": "yt-automation/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def font(size):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", os.path.expanduser("~/.fonts/IBMPlexSansArabic-Bold.ttf")]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


result = {}
for key, q in queries.items():
    url = f"{API}/search?" + urllib.parse.urlencode({"q": q, "media_type": "image"})
    try:
        items = json.loads(get(url))["collection"]["items"]
    except Exception as e:
        print(f"[{key}] search failed: {e}", flush=True)
        continue
    picks = []
    for it in items:
        if len(picks) >= PER:
            break
        d = it["data"][0]
        thumb = next((l["href"] for l in it.get("links", []) if l.get("rel") == "preview"), None)
        if not thumb:
            continue
        try:
            im = Image.open(io.BytesIO(get(thumb.replace(" ", "%20")))).convert("RGB")
        except Exception:
            continue
        # full-resolution size from the asset manifest
        size = None
        try:
            hrefs = [x["href"] for x in json.loads(get(f"{API}/asset/{urllib.parse.quote(d['nasa_id'])}"))["collection"]["items"]]
            orig = next((h for h in hrefs if "~orig." in h and h.lower().endswith((".jpg", ".jpeg", ".png", ".tif"))), None)
            large = next((h for h in hrefs if "~large." in h), None)
        except Exception:
            orig = large = None
        picks.append({"n": len(picks) + 1, "nasa_id": d["nasa_id"], "title": d.get("title", ""), "center": d.get("center", ""),
                      "date": d.get("date_created", "")[:10], "orig": orig, "large": large, "_im": im})
    if not picks:
        print(f"[{key}] no results", flush=True)
        continue
    # contact sheet: 4 x 2 tiles of 480x300 with number label
    W, H, cols = 480, 300, 4
    rows = (len(picks) + cols - 1) // cols
    sheet = Image.new("RGB", (W * cols, H * rows), (20, 20, 20))
    dr = ImageDraw.Draw(sheet)
    f = font(34)
    for p in picks:
        im = p.pop("_im")
        im.thumbnail((W - 8, H - 8))
        x, y = ((p["n"] - 1) % cols) * W, ((p["n"] - 1) // cols) * H
        sheet.paste(im, (x + (W - im.width) // 2, y + (H - im.height) // 2))
        dr.rectangle([x + 4, y + 4, x + 54, y + 48], fill=(242, 181, 68))
        dr.text((x + 14, y + 6), str(p["n"]), fill=(0, 0, 0), font=f)
    sheet.save(f"{out}/{key}.jpg", quality=85)
    result[key] = picks
    print(f"[{key}] {len(picks)} candidates", flush=True)

json.dump(result, open(f"{out}/candidates.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
