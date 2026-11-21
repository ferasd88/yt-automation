"""Objective voice QA: transcription accuracy (faster-whisper), pitch, pitch variability, pace, clipping.

Usage: python tools/voice_qa.py <script.json> <dir-with-variant-subdirs> [whisper_model]
Writes <dir>/qa.json and prints a ranking table.
"""
import json
import os
import re
import sys

import numpy as np
import soundfile as sf

script = json.load(open(sys.argv[1], encoding="utf-8"))
root = sys.argv[2]
wm = sys.argv[3] if len(sys.argv) > 3 else "medium"

import librosa
from faster_whisper import WhisperModel

asr = WhisperModel(wm, device="cpu", compute_type="int8")

AR_DIAC = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
NUMS = {"2560": "الفين وخمسمئه وستين", "69": "تسعه وستين", "2500": "الفين وخمسمئه", "1969": "الف وتسعمئه وتسعه وستين",
        "2000": "الفي"}


def norm(s):
    s = AR_DIAC.sub("", s)
    s = re.sub("[إأآا]", "ا", s).replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    s = s.replace(",", "")
    for k, v in NUMS.items():
        s = s.replace(k, v)
    s = re.sub(r"[^ء-ي ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def cer(ref, hyp):
    r, h = norm(ref).replace(" ", ""), norm(hyp).replace(" ", "")
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


results = {}
for v in sorted(os.listdir(root)):
    vd = os.path.join(root, v)
    if not os.path.isdir(vd):
        continue
    per, f0s, dur_total, chars = [], [], 0.0, 0
    for ln in script["lines"]:
        fn = f"{vd}/{ln['id']}.wav"
        if not os.path.exists(fn):
            continue
        y, sr = sf.read(fn, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        idx = np.where(np.abs(y) > 0.01)[0]
        speech = (idx[-1] - idx[0]) / sr if len(idx) else 0
        dur_total += speech
        chars += len(norm(ln["tts"]).replace(" ", ""))
        y16 = librosa.resample(y, orig_sr=sr, target_sr=16000)
        segs, _ = asr.transcribe(y16, language="ar", beam_size=5)
        hyp = " ".join(s.text for s in segs)
        c = cer(ln["tts"], hyp)
        f0, vf, _ = librosa.pyin(librosa.resample(y, orig_sr=sr, target_sr=16000), fmin=60, fmax=400, sr=16000)
        f0 = f0[~np.isnan(f0)]
        f0s.extend(f0.tolist())
        per.append({"id": ln["id"], "cer": round(c, 3), "hyp": hyp.strip(), "speech_sec": round(speech, 2),
                    "clip": bool(np.abs(y).max() >= 0.999)})
    if not per:
        continue
    f0s = np.array(f0s)
    st = 12 * np.log2(f0s / np.median(f0s)) if len(f0s) else np.array([0])
    results[v] = {
        "mean_cer": round(float(np.mean([p["cer"] for p in per])), 3),
        "max_cer": round(float(np.max([p["cer"] for p in per])), 3),
        "median_f0_hz": round(float(np.median(f0s)), 1) if len(f0s) else 0,
        "f0_std_semitones": round(float(np.std(st)), 2),
        "chars_per_sec": round(chars / max(dur_total, 1e-6), 2),
        "speech_total_sec": round(dur_total, 1),
        "lines": per,
    }
    r = results[v]
    print(f"{v:40s} CER {r['mean_cer']:.3f} (max {r['max_cer']:.3f})  F0 {r['median_f0_hz']:5.0f}Hz  "
          f"var {r['f0_std_semitones']:.2f}st  pace {r['chars_per_sec']:.1f}c/s  total {r['speech_total_sec']}s", flush=True)

json.dump(results, open(f"{root}/qa.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
