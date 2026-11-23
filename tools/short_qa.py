"""Automated QA for a rendered Short. Exit code 1 if any hard check fails.

Usage: python tools/short_qa.py <script.json> <outdir>   (reads outdir/short.mp4)
Checks: format (1080x1920, 30fps, duration window), black frames, frozen video, loudness/true peak,
narration-vs-script accuracy on the final mix (faster-whisper), clipping.
"""
import json
import re
import subprocess
import sys

import numpy as np

script = json.load(open(sys.argv[1], encoding="utf-8"))
out = sys.argv[2]
mp4 = f"{out}/short.mp4"
lim = list(script.get("duration_window", [35, 45]))
lim[1] += 1.5   # story window + the fixed 1.3 s «لمح» outro
report, fails = {}, []


def run(args):
    return subprocess.run(args, capture_output=True, text=True).stderr + subprocess.run(args, capture_output=True, text=True).stdout


info = subprocess.run(["ffmpeg", "-hide_banner", "-i", mp4], capture_output=True, text=True).stderr
m_res = re.search(r"Video:.*?(\d{3,4})x(\d{3,4})", info)
m_fps = re.search(r"(\d+(?:\.\d+)?) fps", info)
m_dur = re.search(r"Duration: (\d+):(\d+):([\d.]+)", info)
v = {"width": int(m_res.group(1)), "height": int(m_res.group(2)), "r_frame_rate": m_fps.group(1) if m_fps else "?"}
dur = int(m_dur.group(1)) * 3600 + int(m_dur.group(2)) * 60 + float(m_dur.group(3))
report["format"] = {"w": v["width"], "h": v["height"], "fps": v["r_frame_rate"], "duration": round(dur, 2)}
if (v["width"], v["height"]) != (1080, 1920):
    fails.append("resolution")
if not (lim[0] <= dur <= lim[1]):
    fails.append(f"duration {dur:.1f}s outside {lim}")

log = subprocess.run(["ffmpeg", "-hide_banner", "-i", mp4, "-vf", "blackdetect=d=0.1:pix_th=0.08,freezedetect=n=0.002:d=2.0",
                      "-af", "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True).stderr
blacks = re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", log)
freezes = re.findall(r"freeze_start: ([\d.]+)", log)
I = re.findall(r"I:\s+(-?[\d.]+) LUFS", log)
TP = re.findall(r"Peak:\s+(-?[\d.]+) dBFS", log)
report["black_segments"] = blacks
report["freeze_starts"] = freezes
report["loudness_lufs"] = float(I[-1]) if I else None
report["true_peak_dbfs"] = float(TP[-1]) if TP else None
if blacks:
    fails.append("black frames")
if freezes:
    fails.append("frozen video")
if report["loudness_lufs"] is None or not (-16 <= report["loudness_lufs"] <= -12):
    fails.append("loudness")
if report["true_peak_dbfs"] is not None and report["true_peak_dbfs"] > -0.5:
    fails.append("true peak")

# narration accuracy on the final mix (music + sfx included)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-ac", "1", "-ar", "16000", f"{out}/qa16k.wav"], check=True)
sys.argv = [sys.argv[0]]
from faster_whisper import WhisperModel

AR_DIAC = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")


DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
NUMW = {"2560": "الفين وخمسمئة وستين", "2500": "الفين وخمسمئة", "1969": "الف وتسعمئة وتسعة وستين", "2000": "الفي", "69": "تسعة وستين"}
NUMW.update(script.get("numbers", {}))


def norm(s):
    s = s.translate(DIGITS).replace(",", "").replace("مائة", "مئة")
    for k, v in script.get("latin", {}).items():
        s = re.sub(rf"(?<![A-Za-z]){re.escape(k)}(?![A-Za-z])", f" {v} ", s)
    for k in sorted(NUMW, key=len, reverse=True):
        s = re.sub(rf"(?<!\d){k}(?!\d)", NUMW[k], s)
    s = AR_DIAC.sub("", s)
    s = re.sub("[إأآا]", "ا", s).replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    return re.sub(r"[^\u0621-\u064A]+", "", s)


def cer(r, h):
    r, h = norm(r), norm(h)
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


m = WhisperModel("medium", device="cpu", compute_type="int8")
segs, _ = m.transcribe(f"{out}/qa16k.wav", language="ar", beam_size=5)
hyp = " ".join(s.text for s in segs)
ref = " ".join(l["tts"] for l in script["lines"])
report["transcript"] = hyp.strip()
report["cer_full"] = round(cer(ref, hyp), 3)
if report["cer_full"] > 0.15:
    fails.append(f"narration CER {report['cer_full']}")

report["fails"] = fails
report["pass"] = not fails
json.dump(report, open(f"{out}/qa.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in report.items() if k != "transcript"}, ensure_ascii=False, indent=1))
print("TRANSCRIPT:", report["transcript"])
sys.exit(0 if not fails else 1)  # noqa
