"""Arabic narration with SILMA TTS, one clip per scene, with a content-hash cache.

Usage: python tools/tts.py space/script.json out/ep [voice_cache_dir]
Writes out/ep/voice.wav (24 kHz mono) and out/ep/timings.json.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

SR = 24000
LEAD, GAP, TAIL = 0.9, 0.55, 3.0
SEED = 7

script = json.load(open(sys.argv[1], encoding="utf-8"))
outdir = sys.argv[2]
cache = sys.argv[3] if len(sys.argv) > 3 else "voice_cache"
os.makedirs(outdir, exist_ok=True)
os.makedirs(cache, exist_ok=True)

REF = os.path.expanduser("~/silma-tts/src/silma_tts/infer/ref_audio_samples/ar.ref.24k.wav")
REF_TEXT = "ويدقق النظر في القرآن الكريم وسائر الكتب السماوية ويتبع مسالك الرسل العظام عليهم الصلاة والسلام."
_model = None


def model():
    global _model
    if _model is None:
        # torchaudio.load/save route through torchcodec, whose PyPI build needs CUDA libs on CPU runners.
        import torch
        import torchaudio

        def _load(path, *a, **k):
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
            return torch.from_numpy(data.T.copy()), sr

        def _save(path, wav, sr, *a, **k):
            x = wav.detach().cpu().numpy() if hasattr(wav, "detach") else np.asarray(wav)
            sf.write(str(path), x.T if x.ndim == 2 else x, sr)

        torchaudio.load, torchaudio.save = _load, _save
        from silma_tts.api import SilmaTTS
        _model = SilmaTTS()
    return _model


def load_mono(fn):
    a, sr = sf.read(fn, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != SR:
        x = np.linspace(0, len(a) - 1, int(len(a) * SR / sr))
        a = np.interp(x, np.arange(len(a)), a).astype(np.float32)
    return a


def trim(a, thr=0.012, pad=0.06):
    idx = np.where(np.abs(a) > thr)[0]
    if len(idx) == 0:
        return a
    return a[max(0, idx[0] - int(pad * SR)): min(len(a), idx[-1] + int(pad * SR))]


def fade(a, ms=12):
    n = int(SR * ms / 1000)
    if len(a) > 2 * n:
        a[:n] *= np.linspace(0, 1, n)
        a[-n:] *= np.linspace(1, 0, n)
    return a


# Shard mode: SHARD="i/N" only synthesises scenes with index % N == i into the cache, then exits.
shard = os.environ.get("SHARD")
if shard:
    si, sn = map(int, shard.split("/"))
    for i, sc in enumerate(script["scenes"]):
        if i % sn != si:
            continue
        key = hashlib.sha1(f"silma|{SEED}|{sc['tts']}".encode()).hexdigest()[:16]
        fn = f"{cache}/{key}.wav"
        if os.path.exists(fn):
            print(f"scene {i:02d}: cached", flush=True)
            continue
        t0 = time.time()
        model().infer(ref_file=REF, ref_text=REF_TEXT, gen_text=sc["tts"], file_wave=fn, seed=SEED, speed=1)
        print(f"scene {i:02d}: {len(sf.read(fn)[0]) / SR:.2f}s audio in {time.time() - t0:.1f}s", flush=True)
    sys.exit(0)

pieces = [np.zeros(int(LEAD * SR), np.float32)]
t = LEAD
timings, report = [], []
scenes = script["scenes"]
for i, sc in enumerate(scenes):
    key = hashlib.sha1(f"silma|{SEED}|{sc['tts']}".encode()).hexdigest()[:16]
    fn = f"{cache}/{key}.wav"
    t0 = time.time()
    hit = os.path.exists(fn)
    if not hit:
        model().infer(ref_file=REF, ref_text=REF_TEXT, gen_text=sc["tts"], file_wave=fn, seed=SEED, speed=1)
    a = fade(trim(load_mono(fn)))
    d = len(a) / SR
    report.append({"scene": i, "audio_sec": round(d, 2), "gen_sec": round(time.time() - t0, 1), "cached": hit})
    print(f"scene {i:02d}: {d:5.2f}s audio  {'cache' if hit else f'{time.time() - t0:.1f}s gen'}", flush=True)
    timings.append({"start": round(t, 3), "end": round(t + d, 3)})
    pieces.append(a)
    t += d
    gap = GAP if i < len(scenes) - 1 else TAIL
    pieces.append(np.zeros(int(gap * SR), np.float32))
    t += gap

voice = np.concatenate(pieces)
peak = np.abs(voice).max()
if peak > 0:
    voice = voice / peak * 0.9
sf.write(f"{outdir}/voice.wav", voice, SR)
json.dump({"total": round(len(voice) / SR, 3), "scenes": timings, "report": report},
          open(f"{outdir}/timings.json", "w"), indent=1)
print(f"total {len(voice) / SR:.1f}s")
