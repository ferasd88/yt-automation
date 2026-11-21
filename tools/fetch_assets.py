"""Download chosen NASA Image Library assets (public domain) and prepare them for rendering.

Usage: python tools/fetch_assets.py space/assets.json out/ep/assets
"""
import io
import json
import os
import sys
import urllib.parse
import urllib.error
import urllib.request

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
API = "https://images-api.nasa.gov"
spec = json.load(open(sys.argv[1], encoding="utf-8"))
out = sys.argv[2]
os.makedirs(out, exist_ok=True)


def get(url, timeout=180, tries=6):
    import time
    req = urllib.request.Request(url.replace(" ", "%20"), headers={"User-Agent": "yt-automation/1.0 (educational shorts; github ferasd88)"})
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            time.sleep(1.5)  # be polite to Wikimedia / NASA
            return data
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and k < tries - 1:
                wait = 10 * (k + 1)
                print(f"  HTTP {e.code}, retrying in {wait}s", flush=True)
                time.sleep(wait)
                continue
            raise


meta = {}
def nasa_src(s):
    hrefs = [x["href"] for x in json.loads(get(f"{API}/asset/{urllib.parse.quote(s['nasa_id'])}"))["collection"]["items"]]
    imgs = [h for h in hrefs if h.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))]
    order = ["~orig.", "~large.", "~medium."]
    src = next((h for tag in order for h in imgs if tag in h), None)
    return src


def commons_thumbs(url):
    # upload.wikimedia.org/wikipedia/commons/a/ab/Name.jpg -> standard-size thumbnails (originals get 429 from CI IPs)
    import re as _re
    m = _re.match(r"(https://upload\.wikimedia\.org/wikipedia/commons)/([0-9a-f])/([0-9a-f]{2})/(.+)$", url)
    if not m:
        return []
    base, a, ab, name = m.groups()
    return [f"{base}/thumb/{a}/{ab}/{name}/{w}px-{name}" for w in (3840, 2560, 1920)]


for key, s in spec.items():
    if os.path.exists(f"{out}/{key}.jpg"):
        print(f"[{key}] cached", flush=True)
        with Image.open(f"{out}/{key}.jpg") as im0:
            meta[key] = {"cached": True, "size": list(im0.size)}
        continue
    src, data = None, None
    if s.get("commons_title"):
        # Wikimedia serves bots only standard thumbnail sizes, and only below the original width
        try:
            q = urllib.parse.urlencode({"action": "query", "format": "json", "titles": s["commons_title"], "prop": "imageinfo", "iiprop": "size"})
            ow = next(iter(json.loads(get("https://commons.wikimedia.org/w/api.php?" + q))["query"]["pages"].values()))["imageinfo"][0]["width"]
        except Exception:
            ow = 99999
        for w in [x for x in (3840, 1920, 1280, 960) if x < ow]:
            try:
                q = urllib.parse.urlencode({"action": "query", "format": "json", "titles": s["commons_title"], "prop": "imageinfo",
                                            "iiprop": "url|size", "iiurlwidth": w})
                pg = next(iter(json.loads(get("https://commons.wikimedia.org/w/api.php?" + q))["query"]["pages"].values()))
                th = pg["imageinfo"][0].get("thumburl")
                if not th:
                    continue
                data, src = get(th, timeout=120, tries=3), th
                break
            except Exception as e:
                print(f"[{key}] api thumb {w} failed: {e}", flush=True)
    if data is None and s.get("url"):  # higher-resolution source (e.g. ESA/Webb CC BY 4.0), NASA library as fallback
        try:
            data, src = get(s["url"], timeout=300), s["url"]
        except Exception as e:
            print(f"[{key}] url failed ({e}), falling back to NASA library", flush=True)
    if data is None:
        if not s.get("nasa_id"):
            sys.exit(f"[{key}] download failed and no NASA fallback")
        src = nasa_src(s)
        if not src:
            sys.exit(f"[{key}] no image file for {s['nasa_id']}")
        data = get(src)
    im = Image.open(io.BytesIO(data))
    im = im.convert("RGB")
    if s.get("crop"):  # fractions [left, top, right, bottom] to remove (e.g. caption strips)
        l, t, r, b = s["crop"]
        W0, H0 = im.size
        im = im.crop((int(W0 * l), int(H0 * t), int(W0 * (1 - r)), int(H0 * (1 - b))))
    w0, h0 = im.size
    mx = s["width"]
    if max(im.size) > mx:
        r = mx / max(im.size)
        im = im.resize((round(im.width * r), round(im.height * r)), Image.LANCZOS)
    im.save(f"{out}/{key}.jpg", quality=93)
    meta[key] = {"nasa_id": s.get("nasa_id"), "src": src, "orig_size": [w0, h0], "size": list(im.size)}
    print(f"[{key}] {s.get('nasa_id') or src[-60:]} {w0}x{h0} -> {im.size[0]}x{im.size[1]}", flush=True)

json.dump(meta, open(f"{out}/assets_meta.json", "w"), indent=1)
