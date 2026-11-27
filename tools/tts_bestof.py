"""Best-of-N narration for one script line with Chatterbox Multilingual (MIT), verified by faster-whisper.

Usage: python tools/tts_bestof.py <script.json> <line_index> <outdir>
For each take: synthesize -> transcribe with word timestamps -> cut any trailing hallucinated words after the
last expected word -> score (CER vs script + required keywords). Keeps the best take as <outdir>/<id>.wav.
"""
import json
import os
import re
import sys
import time

import numpy as np
import soundfile as sf

script = json.load(open(sys.argv[1], encoding="utf-8"))
li = int(sys.argv[2])
od = sys.argv[3]
os.makedirs(od, exist_ok=True)
ln = script["lines"][li]
cfgv = script.get("voice", {})
TAKES = ln.get("takes", cfgv.get("takes", 3))

AR_DIAC = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
NUMW = {"2560": "الفين وخمسمئة وستين", "2500": "الفين وخمسمئة", "1969": "الف وتسعمئة وتسعة وستين", "2000": "الفي", "69": "تسعة وستين"}


NUMW.update(script.get("numbers", {}))


def norm(s, keep_space=True):
    s = s.translate(DIGITS).replace(",", "").replace("٬", "")
    s = s.replace("مائة", "مئة").replace("مائه", "مئه")
    for k, v in script.get("latin", {}).items():  # e.g. Whisper writes the brand "كيو" as "Q"
        s = re.sub(rf"(?<![A-Za-z]){re.escape(k)}(?![A-Za-z])", f" {v} ", s)
    for k in sorted(NUMW, key=len, reverse=True):
        s = re.sub(rf"(?<!\d){k}(?!\d)", NUMW[k], s)
    s = AR_DIAC.sub("", s)
    s = re.sub("[إأآا]", "ا", s).replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    s = re.sub(r"[^ء-ي ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if keep_space else s.replace(" ", "")


def cer(r, h):
    r, h = norm(r, False), norm(h, False)
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


import torch
import torchaudio


def _load(path, *a, **k):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr


torchaudio.load = _load
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from faster_whisper import WhisperModel
import librosa

tts = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
asr = WhisperModel("medium", device="cpu", compute_type="int8")

# ---- cloned voice: reference clip per mood ------------------------------------------------------------
# voice.refs      {mood: path}                    explicit mapping (paths relative to the script folder)
# voice.refs_dir  folder of unlabeled clips + voice.ref_texts {mood: text read in the clip}:
#                 each clip is transcribed and matched to the text it reads (best overall assignment).
# voice.moods     {mood: {exaggeration, cfg}}     per-mood delivery; line["mood"] picks the clip.
import itertools
sdir = os.path.dirname(os.path.abspath(sys.argv[1]))
refs = {k: os.path.join(sdir, v) for k, v in cfgv.get("refs", {}).items()}
if not refs and cfgv.get("refs_dir"):
    rdir = os.path.join(sdir, cfgv["refs_dir"])
    clips = sorted(f for f in os.listdir(rdir) if f.endswith(".wav"))
    texts = cfgv["ref_texts"]
    heard = {}
    for c in clips:
        a, _ = librosa.load(os.path.join(rdir, c), sr=16000)
        heard[c] = " ".join(x.text for x in asr.transcribe(a, language="ar", beam_size=5)[0])
    moods = list(texts)
    best_perm, best_cost = None, 9e9
    for perm in itertools.permutations(clips, len(moods)):
        cost = sum(cer(texts[m], heard[c]) for m, c in zip(moods, perm))
        if cost < best_cost:
            best_perm, best_cost = perm, cost
    refs = {m: os.path.join(rdir, c) for m, c in zip(moods, best_perm)}
    for m, c in zip(moods, best_perm):
        print(f"REF {m} <- {c} cer {cer(texts[m], heard[c]):.3f} | {heard[c]}", flush=True)
mood = ln.get("mood", cfgv.get("default_mood"))
ref = refs.get(mood) if mood else None
mcfg = cfgv.get("moods", {}).get(mood, {}) if mood else {}
EXAG = mcfg.get("exaggeration", cfgv.get("exaggeration", 0.65))
CFG = mcfg.get("cfg", cfgv.get("cfg", 0.4))
if ref:
    print(f"line {ln['id']} mood {mood} ref {os.path.basename(ref)} exaggeration {EXAG} cfg {CFG}", flush=True)
ref_chars = len(norm(ln["tts"], False))
musts = [norm(m, False) for m in ln.get("must", [])]

takes = []
for k in range(TAKES):
    seed = 101 + 17 * k
    torch.manual_seed(seed)
    t0 = time.time()
    wav = tts.generate(ln["tts"], language_id="ar", audio_prompt_path=ref, exaggeration=EXAG, cfg_weight=CFG)
    y = wav.detach().cpu().numpy().squeeze().astype(np.float32)
    sr = tts.sr
    y16 = librosa.resample(y, orig_sr=sr, target_sr=16000)
    segs, _ = asr.transcribe(y16, language="ar", beam_size=5, word_timestamps=True)
    words = [w for s in segs for w in (s.words or [])]
    full = " ".join(w.word for w in words)
    # cut right after the last expected word (drops trailing babble); fuzzy match on the script's final word
    import difflib
    last_ref = norm(ln["tts"]).split()[-1]
    cut_t, acc = None, 0
    for w in words:
        nw = norm(w.word, False)
        acc += len(nw)
        if acc >= 0.85 * ref_chars and nw and difflib.SequenceMatcher(None, nw, last_ref).ratio() >= 0.75:
            cut_t = w.end + 0.15
            break
    kept = full
    if cut_t is not None and cut_t * sr < len(y) - 0.25 * sr:
        y = y[: int(cut_t * sr)]
        kept = " ".join(w.word for w in words if w.end <= cut_t)
    n = int(0.03 * sr)
    y[-n:] *= np.linspace(1, 0, n)
    c = cer(ln["tts"], kept)
    nh = norm(kept, False)
    miss = [m for m in musts if m not in nh]
    score = c + 0.25 * len(miss)
    fn = f"{od}/{ln['id']}.take{k}.wav"
    sf.write(fn, y, sr)
    takes.append({"take": k, "seed": seed, "cer": round(c, 3), "missing": miss, "score": round(score, 3), "dur": round(len(y) / sr, 2),
                  "trimmed": cut_t is not None, "hyp": kept, "raw_hyp": full, "gen_sec": round(time.time() - t0, 1), "file": fn})
    print(f"{ln['id']} take {k}: cer {c:.3f} miss {miss} dur {len(y) / sr:.2f}s | {kept}", flush=True)

best = min(takes, key=lambda x: (x["score"], x["dur"]))
os.replace(best["file"], f"{od}/{ln['id']}.wav")
for t in takes:
    if os.path.exists(t["file"]):
        os.remove(t["file"])
json.dump({"line": ln["id"], "mood": mood, "ref": os.path.basename(ref) if ref else None, "best": best["take"], "takes": takes}, open(f"{od}/{ln['id']}.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(f"BEST {ln['id']}: take {best['take']} cer {best['cer']} missing {best['missing']}")
