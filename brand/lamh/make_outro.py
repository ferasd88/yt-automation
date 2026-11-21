"""Render the «لمح» outro: 1080x1920 @30fps, duration D, with the synthesized sound logo.
Usage: python make_outro.py <layers_dir> <out_dir> [duration] [fps]
Writes outro.mp4 (H.264 + AAC), outro.wav and contact stills.
"""
import os
import subprocess
import sys

from playwright.sync_api import sync_playwright

ld, od = sys.argv[1], sys.argv[2]
D = float(sys.argv[3]) if len(sys.argv) > 3 else 1.3
FPS = int(sys.argv[4]) if len(sys.argv) > 4 else 30
os.makedirs(od, exist_ok=True)
here = os.path.dirname(os.path.abspath(__file__))


def b64(n):
    return open(os.path.join(ld, f"{n}.b64")).read()


data = {"letters": b64("L_letters"), "stop": b64("L_slash_top"), "sbot": b64("L_slash_bot"), "tri": b64("L_tri"),
        "orig": b64("logo_src"), "duration": D}
subprocess.run([sys.executable, os.path.join(here, "sound_logo.py"), f"{od}/outro.wav", str(D)], check=True)

n = round(D * FPS)
with sync_playwright() as p:
    try:
        b = p.chromium.launch(channel="chrome")
    except Exception:
        b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1080, "height": 1920})
    pg.goto("file://" + os.path.join(here, "outro.html"))
    pg.evaluate("d => init(d)", data)
    pg.evaluate("Promise.all([...document.images].map(i => i.decode().catch(() => null)))")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(FPS), "-c:v", "png", "-i", "-",
                           "-i", f"{od}/outro.wav", "-c:v", "libx264", "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p",
                           "-r", str(FPS), "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest", "-movflags", "+faststart",
                           f"{od}/outro.mp4"], stdin=subprocess.PIPE)
    for f in range(n):
        t = f / FPS
        pg.evaluate(f"render({t})")
        png = pg.screenshot(type="png")
        ff.stdin.write(png)
        if f in (0, 3, 6, 9, 12, 15, 18, 24, 30, 36) and f < n:
            open(f"{od}/still_{f:02d}.png", "wb").write(png)
    b.close()
ff.stdin.close()
sys.exit(ff.wait())
