"""Render an episode: Chromium draws each frame of tools/scene.html, FFmpeg encodes it with the mixed audio.

Usage: python tools/render.py space/script.json out/ep [--stills-only]
Needs out/ep/timings.json, out/ep/mix.wav, out/ep/assets/*.jpg
"""
import json
import os
import subprocess
import sys

from PIL import Image
from playwright.sync_api import sync_playwright

FPS, W, H = 30, 1920, 1080
script = json.load(open(sys.argv[1], encoding="utf-8"))
ep = sys.argv[2]
stills_only = "--stills-only" in sys.argv
timings = json.load(open(f"{ep}/timings.json"))
assets = {}
for f in sorted(os.listdir(f"{ep}/assets")):
    if f.endswith(".jpg"):
        with Image.open(f"{ep}/assets/{f}") as im:
            assets[f[:-4]] = {"src": "file://" + os.path.abspath(f"{ep}/assets/{f}"), "w": im.width, "h": im.height}

total = timings["total"]
n = int(total * FPS)
os.makedirs(f"{ep}/stills", exist_ok=True)
still_times = [round((s["start"] + s["end"]) / 2 + 0.3, 2) for s in timings["scenes"]]


def launch(p):
    try:
        return p.chromium.launch(channel="chrome")
    except Exception:
        return p.chromium.launch()


with sync_playwright() as p:
    browser = launch(p)
    page = browser.new_page(viewport={"width": W, "height": H})
    page.goto("file://" + os.path.abspath("tools/scene.html"))
    page.evaluate("d => init(d)", {"script": script, "timings": timings, "assets": assets})
    page.evaluate("document.fonts.ready.then(() => Promise.all([...document.images].map(i => i.decode().catch(() => null))))")

    for i, t in enumerate(still_times):
        page.evaluate(f"render({t})")
        page.screenshot(path=f"{ep}/stills/s{i:02d}.jpg", type="jpeg", quality=88)
    if stills_only:
        browser.close()
        sys.exit(0)

    out = f"{ep}/video.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error",
                           "-f", "image2pipe", "-framerate", str(FPS), "-c:v", "mjpeg", "-i", "-",
                           "-i", f"{ep}/mix.wav",
                           "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-ar", "48000",
                           "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
                           "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out],
                          stdin=subprocess.PIPE)
    for f in range(n):
        page.evaluate(f"render({f / FPS})")
        ff.stdin.write(page.screenshot(type="jpeg", quality=94))
        if f % 300 == 0:
            print(f"frame {f}/{n}", flush=True)
    browser.close()
ff.stdin.close()
if ff.wait() != 0:
    sys.exit("ffmpeg failed")
print(f"wrote {out}", flush=True)
