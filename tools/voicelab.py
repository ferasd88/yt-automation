"""Synthesize every script line with one engine/variant.

Usage: python tools/voicelab.py <script.json> <engine> <variant> <outdir>
engines: silma (variant = ref wav filename | speed), chatterbox (variant = exaggeration:cfg[:prompt])
Writes <outdir>/<variant_slug>/<line_id>.wav and gen_report.json
"""
import json
import os
import re
import sys
import time

import numpy as np
import soundfile as sf

script = json.load(open(sys.argv[1], encoding="utf-8"))
engine, variant, out = sys.argv[2], sys.argv[3], sys.argv[4]
slug = f"{engine}-" + re.sub(r"[^A-Za-z0-9.]+", "_", variant)
od = os.path.join(out, slug)
os.makedirs(od, exist_ok=True)

SILMA_REF_DIR = os.path.expanduser("~/silma-tts/src/silma_tts/infer/ref_audio_samples")
SILMA_REF_TEXT = "ويدقق النظر في القرآن الكريم وسائر الكتب السماوية ويتبع مسالك الرسل العظام عليهم الصلاة والسلام."


def patch_torchaudio():
    import torch
    import torchaudio

    def _load(path, *a, **k):
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T.copy()), sr

    def _save(path, wav, sr, *a, **k):
        x = wav.detach().cpu().numpy() if hasattr(wav, "detach") else np.asarray(wav)
        sf.write(str(path), x.T if x.ndim == 2 else x, sr)

    torchaudio.load, torchaudio.save = _load, _save


report = {"engine": engine, "variant": variant, "lines": []}
t_all = time.time()
if engine == "silma":
    patch_torchaudio()
    from silma_tts.api import SilmaTTS
    m = SilmaTTS()
    ref_name, speed = (variant.split("|") + ["1.0"])[:2]
    ref = os.path.join(SILMA_REF_DIR, ref_name)
    ref_text = SILMA_REF_TEXT if ref_name.startswith("ar.") else None
    for ln in script["lines"]:
        t0 = time.time()
        fn = f"{od}/{ln['id']}.wav"
        m.infer(ref_file=ref, ref_text=ref_text, gen_text=ln["tts"], file_wave=fn, seed=7, speed=float(speed))
        report["lines"].append({"id": ln["id"], "gen_sec": round(time.time() - t0, 1)})
        print(f"{slug} {ln['id']} {time.time() - t0:.1f}s", flush=True)
elif engine == "chatterbox":
    patch_torchaudio()
    import torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    m = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
    parts = variant.split(":")
    exag, cfg = float(parts[0]), float(parts[1])
    prompt = parts[2] if len(parts) > 2 else ""
    prompt_path = os.path.join(SILMA_REF_DIR, prompt) if prompt else None
    for ln in script["lines"]:
        t0 = time.time()
        torch.manual_seed(7)
        kw = dict(language_id="ar", exaggeration=exag, cfg_weight=cfg)
        if prompt_path:
            kw["audio_prompt_path"] = prompt_path
        wav = m.generate(ln["tts"], **kw)
        x = wav.detach().cpu().numpy()
        sf.write(f"{od}/{ln['id']}.wav", x.T if x.ndim == 2 else x, m.sr)
        report["lines"].append({"id": ln["id"], "gen_sec": round(time.time() - t0, 1)})
        print(f"{slug} {ln['id']} {time.time() - t0:.1f}s", flush=True)
else:
    sys.exit("unknown engine")
report["total_sec"] = round(time.time() - t_all, 1)
json.dump(report, open(f"{od}/gen_report.json", "w"), indent=1)
