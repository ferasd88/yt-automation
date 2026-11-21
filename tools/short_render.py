"""Render a vertical Short (1080x1920, 30 fps) from plan.json + assets + mix.wav.

Usage: python tools/short_render.py <outdir> <assets_dir> [--stills-only]
"""
import json
import os
import subprocess
import sys

from PIL import Image
from playwright.sync_api import sync_playwright

FPS, W, H = 30, 1080, 1920
out, adir = sys.argv[1], sys.argv[2]
stills_only = "--stills-only" in sys.argv
plan = json.load(open(f"{out}/plan.json", encoding="utf-8"))
assets = {}
for f in sorted(os.listdir(adir)):
    if f.endswith(".jpg"):
        with Image.open(f"{adir}/{f}") as im:
            assets[f[:-4]] = {"src": "file://" + os.path.abspath(f"{adir}/{f}"), "w": im.width, "h": im.height}
n = int(plan["total"] * FPS)
os.makedirs(f"{out}/stills", exist_ok=True)


def launch(p):
    try:
        return p.chromium.launch(channel="chrome")
    except Exception:
        return p.chromium.launch()


with sync_playwright() as p:
    b = launch(p)
    pg = b.new_page(viewport={"width": W, "height": H})
    pg.goto("file://" + os.path.abspath("tools/short_scene.html"))
    logo_b64 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "brand", "lamh", "logo_src.b64")
    logo = open(logo_b64).read() if os.path.exists(logo_b64) else None
    pg.evaluate("d => init(d)", {"plan": plan, "assets": assets, "logo": logo})
    pg.evaluate("document.fonts.ready.then(() => Promise.all([...document.images].map(i => i.decode().catch(() => null))))")
    for i, sh in enumerate(plan["shots"]):
        for frac, tag in ((0.25, "a"), (0.8, "b")):
            t = sh["start"] + (sh["end"] - sh["start"]) * frac
            pg.evaluate(f"render({t})")
            pg.screenshot(path=f"{out}/stills/{i:02d}{tag}.jpg", type="jpeg", quality=85)
    if stills_only:
        b.close()
        sys.exit(0)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(FPS), "-c:v", "mjpeg", "-i", "-",
                           "-i", f"{out}/mix.wav", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                           "-profile:v", "high", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", f"{out}/short.mp4"],
                          stdin=subprocess.PIPE)
    for f in range(n):
        pg.evaluate(f"render({f / FPS})")
        ff.stdin.write(pg.screenshot(type="jpeg", quality=95))
        if f % 300 == 0:
            print(f"frame {f}/{n}", flush=True)
    b.close()
ff.stdin.close()
sys.exit(ff.wait())
