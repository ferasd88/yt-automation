"""Search Wikimedia Commons (and optionally the NASA library) for freely licensed images; build numbered contact sheets.

Usage: python tools/commons_candidates.py <queries.json> <outdir>
queries.json: {"key": {"commons": "search text"} | {"nasa": "search text"}}
Only keeps Public Domain / CC0 / CC BY / CC BY-SA files. Records license + author for attribution.
"""
import io
import json
import os
import re
import sys
import urllib.parse
import urllib.request

from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None
UA = {"User-Agent": "yt-automation/1.0 (educational shorts; contact via GitHub ferasd88)"}
queries = json.load(open(sys.argv[1], encoding="utf-8"))
out = sys.argv[2]
os.makedirs(out, exist_ok=True)
PER = 8
OK_LIC = re.compile(r"^(public domain|pd|cc0|cc by(-sa)? ?\d|cc-by(-sa)?-\d)", re.I)


def get(url, timeout=60):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def strip_html(s):
    return re.sub(r"<[^>]+>", "", s or "").strip()


def commons(q):
    p = {"action": "query", "format": "json", "generator": "search", "gsrsearch": f"filetype:bitmap {q}", "gsrnamespace": 6,
         "gsrlimit": 25, "prop": "imageinfo", "iiprop": "url|size|extmetadata", "iiurlwidth": 600}
    data = json.loads(get("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(p)))
    pages = sorted(data.get("query", {}).get("pages", {}).values(), key=lambda x: x.get("index", 0))
    res = []
    for pg in pages:
        ii = pg.get("imageinfo", [{}])[0]
        em = ii.get("extmetadata", {})
        lic = strip_html(em.get("LicenseShortName", {}).get("value", ""))
        if not OK_LIC.match(lic):
            continue
        if ii.get("width", 0) < 1000:
            continue
        res.append({"source": "commons", "title": pg["title"], "url": ii["url"], "thumb": ii.get("thumburl"),
                    "w": ii["width"], "h": ii["height"], "license": lic,
                    "artist": strip_html(em.get("Artist", {}).get("value", ""))[:120]})
    return res


def nasa(q):
    data = json.loads(get("https://images-api.nasa.gov/search?" + urllib.parse.urlencode({"q": q, "media_type": "image"})))
    res = []
    for it in data["collection"]["items"][:25]:
        d = it["data"][0]
        th = next((l["href"] for l in it.get("links", []) if l.get("rel") == "preview"), None)
        if th:
            res.append({"source": "nasa", "title": d.get("title", ""), "nasa_id": d["nasa_id"], "thumb": th,
                        "license": "Public domain (NASA)", "artist": d.get("center", "")})
    return res


def font(size):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


allres = {}
for key, q in queries.items():
    try:
        items = commons(q["commons"]) if "commons" in q else nasa(q["nasa"])
    except Exception as e:
        print(f"[{key}] failed: {e}", flush=True)
        continue
    picks = []
    for it in items:
        if len(picks) >= PER:
            break
        try:
            im = Image.open(io.BytesIO(get(it["thumb"].replace(" ", "%20")))).convert("RGB")
        except Exception:
            continue
        it["n"] = len(picks) + 1
        it["_im"] = im
        picks.append(it)
    if not picks:
        print(f"[{key}] none", flush=True)
        continue
    W, H, cols = 480, 360, 4
    rows = (len(picks) + cols - 1) // cols
    sheet = Image.new("RGB", (W * cols, H * rows), (20, 20, 20))
    dr = ImageDraw.Draw(sheet)
    f = font(32)
    for p in picks:
        im = p.pop("_im")
        im.thumbnail((W - 8, H - 8))
        x, y = ((p["n"] - 1) % cols) * W, ((p["n"] - 1) // cols) * H
        sheet.paste(im, (x + (W - im.width) // 2, y + (H - im.height) // 2))
        dr.rectangle([x + 4, y + 4, x + 50, y + 44], fill=(242, 181, 68))
        dr.text((x + 12, y + 6), str(p["n"]), fill=(0, 0, 0), font=f)
    sheet.save(f"{out}/{key}.jpg", quality=82)
    allres[key] = picks
    print(f"[{key}] {len(picks)}", flush=True)

json.dump(allres, open(f"{out}/candidates.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
