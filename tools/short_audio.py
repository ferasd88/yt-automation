"""Assemble a Short's audio: narration timing, shot plan, original music, synthesized SFX, final mix.

Usage: python tools/short_audio.py <script.json> <voice_dir> <outdir>
voice_dir holds <line_id>.wav. Writes outdir/plan.json, outdir/voice.wav, outdir/mix.wav (48 kHz stereo).
"""
import json
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

SR = 48000
script = json.load(open(sys.argv[1], encoding="utf-8"))
vdir, out = sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(5)

LEAD = 0.12          # hook starts almost immediately
GAP = 0.16           # tight pacing between lines
GAP_LONG = {"age": 0.32, "salawat": 0.22, "gap": 0.3, "payoff": 0.45}   # breathing room after key beats
TAIL = 0.55          # the «لمح» outro follows right after the story


TEMPO = float(script.get("voice", {}).get("tempo", 1.0))


def load(fn):
    if abs(TEMPO - 1.0) > 1e-3:   # pitch-preserving speed-up for a livelier pace
        tmp = f"{out}/_tempo.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", fn, "-af", f"atempo={TEMPO}", tmp], check=True)
        fn = tmp
    y, sr = sf.read(fn, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        x = np.linspace(0, len(y) - 1, int(len(y) * SR / sr))
        y = np.interp(x, np.arange(len(y)), y).astype(np.float32)
    idx = np.where(np.abs(y) > 0.012)[0]
    if len(idx):
        y = y[max(0, idx[0] - int(0.03 * SR)): idx[-1] + int(0.08 * SR)]
    n = int(0.01 * SR)
    y[:n] *= np.linspace(0, 1, n)
    y[-n:] *= np.linspace(1, 0, n)
    return y


# --- narration + plan
t = LEAD
pieces = [np.zeros(int(LEAD * SR), np.float32)]
lines = []
for ln in script["lines"]:
    y = load(f"{vdir}/{ln['id']}.wav")
    d = len(y) / SR
    lines.append({"id": ln["id"], "start": round(t, 3), "end": round(t + d, 3), "caption": ln["caption"]})
    pieces.append(y)
    t += d
    g = GAP_LONG.get(ln["id"], GAP)
    if ln is script["lines"][-1]:
        g = TAIL
    pieces.append(np.zeros(int(g * SR), np.float32))
    t += g
voice = np.concatenate(pieces)
voice /= max(1e-6, np.abs(voice).max()) / 0.9
total = len(voice) / SR

# shots: split each line's window (from its start to next line's start) by shot count
shots, events = [], []
for i, ln in enumerate(script["lines"]):
    a = lines[i]["start"] - (0.08 if i else LEAD)
    b = lines[i + 1]["start"] - 0.08 if i + 1 < len(lines) else total
    k = len(ln["shots"])
    for j, sh in enumerate(ln["shots"]):
        s0 = a + (b - a) * j / k
        s1 = a + (b - a) * (j + 1) / k
        shots.append({**sh, "line": ln["id"], "start": round(max(0, s0), 3), "end": round(s1, 3)})
        events.append({"t": max(0.0, s0 - 0.12), "sfx": "whoosh"})
        if sh.get("flash"):
            events.append({"t": s0, "sfx": "impact"})
        if sh.get("year"):
            events.append({"t": s0 + 0.35, "sfx": "hit"})
        if sh.get("counter"):
            events.append({"t": s0 + 0.3, "sfx": "ticks", "dur": 1.2})
        if sh["type"] == "salawat":
            events.append({"t": s0 + (s1 - s0) * 0.45, "sfx": "click"})
            events.append({"t": s0 + (s1 - s0) * 0.68, "sfx": "pop"})
        if sh["type"] in ("timeline", "map", "aerosol"):
            events.append({"t": s0 + 0.4, "sfx": "rise"})
        if sh["type"] == "title":
            events.append({"t": s0 + 0.1, "sfx": "impact"})
events = [e for e in events if e["t"] > 0.05 or e["sfx"] != "whoosh"]
json.dump({"total": round(total, 3), "lines": lines, "shots": shots, "events": events},
          open(f"{out}/plan.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
sf.write(f"{out}/voice.wav", voice, SR)

# --- SFX (synthesized, original)
N = len(voice)
sfx = np.zeros(N, np.float32)


def env(n, a, r):
    e = np.ones(n, np.float32)
    na, nr = int(a * SR), int(r * SR)
    e[:na] = np.linspace(0, 1, na)
    e[-nr:] *= np.linspace(1, 0, nr) ** 2
    return e


def bandnoise(n, f_lo, f_hi):
    x = rng.standard_normal(n).astype(np.float32)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    X[(f < f_lo) | (f > f_hi)] = 0
    return np.fft.irfft(X, n).astype(np.float32)


def make(kind, dur=None):
    if kind == "whoosh":
        n = int(0.42 * SR)
        x = rng.standard_normal(n).astype(np.float32)
        # sweeping one-pole lowpass
        cut = np.linspace(0.02, 0.22, n) ** 1.5
        y = np.zeros(n, np.float32)
        acc = 0.0
        for i in range(n):
            acc += cut[i] * (x[i] - acc)
            y[i] = acc
        return 0.2 * y / (np.abs(y).max() + 1e-9) * np.sin(np.linspace(0, np.pi, n)) ** 1.5  # softer cut whoosh (~-9 dB)
    if kind in ("hit", "impact"):
        n = int((0.9 if kind == "impact" else 0.6) * SR)
        tt = np.arange(n) / SR
        f = 55 + 70 * np.exp(-tt * 18)
        boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * (5 if kind == "impact" else 7))
        click = bandnoise(n, 1500, 6000) * np.exp(-tt * 60) * 0.35
        return (0.9 if kind == "impact" else 0.65) * (boom + click)
    if kind == "click":
        n = int(0.08 * SR)
        tt = np.arange(n) / SR
        return 0.5 * bandnoise(n, 2000, 8000) * np.exp(-tt * 90)
    if kind == "pop":
        n = int(0.18 * SR)
        tt = np.arange(n) / SR
        f = 900 * np.exp(-tt * 8) + 300
        return 0.45 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 22)
    if kind == "ticks":
        n = int((dur or 1.0) * SR)
        y = np.zeros(n, np.float32)
        for k in np.arange(0, n, int(0.07 * SR)):
            m = min(n - k, int(0.02 * SR))
            y[k:k + m] += bandnoise(m, 3000, 9000)[:m] * np.exp(-np.arange(m) / (0.003 * SR)) * 0.35
        return y
    if kind == "rise":
        n = int(0.9 * SR)
        tt = np.arange(n) / SR
        f = 200 + 500 * (tt / tt[-1]) ** 2
        return 0.18 * np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.6, 0.15) + 0.12 * bandnoise(n, 2000, 7000) * (tt / tt[-1]) ** 2
    return np.zeros(1, np.float32)


for e in events:
    s = make(e["sfx"], e.get("dur"))
    i = int(e["t"] * SR)
    j = min(N, i + len(s))
    if i < N:
        sfx[i:j] += s[: j - i].astype(np.float32)

# --- music: driving minor pulse, 100 BPM (original)
bpm = 100
beat = 60 / bpm
T = np.arange(N) / SR
music = np.zeros((N, 2), np.float32)
prog = [57, 53, 48, 55]       # A, F, C, G roots (MIDI)
bar = 4 * beat


def hz(m):
    return 440 * 2 ** ((m - 69) / 12)


for b_i in range(int(total / bar) + 2):
    root = prog[b_i % 4]
    for step in range(8):                       # eighth-note pluck arpeggio
        st = b_i * bar + step * beat / 2
        i = int(st * SR)
        if i >= N:
            break
        n = min(N - i, int(0.45 * SR))
        tt = np.arange(n) / SR
        note = root + [12, 19, 15, 24, 12, 19, 15, 22][step]
        tone = (np.sign(np.sin(2 * np.pi * hz(note) * tt)) * 0.25 + np.sin(2 * np.pi * hz(note) * tt)) * np.exp(-tt * 9)
        pan = 0.35 if step % 2 else -0.35
        music[i:i + n, 0] += 0.07 * tone * (1 - pan)
        music[i:i + n, 1] += 0.07 * tone * (1 + pan)
    for q in range(4):                          # soft kick on each beat + sub bass
        st = b_i * bar + q * beat
        i = int(st * SR)
        if i >= N:
            break
        n = min(N - i, int(0.35 * SR))
        tt = np.arange(n) / SR
        kick = np.sin(2 * np.pi * np.cumsum(45 + 90 * np.exp(-tt * 30)) / SR) * np.exp(-tt * 12)
        sub = np.sin(2 * np.pi * hz(root - 24) * tt) * np.exp(-tt * 3)
        music[i:i + n] += (0.22 * kick + 0.12 * sub)[:, None]
music = np.tanh(music * 1.4) / 1.4
fin, fout = int(0.25 * SR), int(0.6 * SR)
music[:fin] *= np.linspace(0, 1, fin)[:, None]
music[-fout:] *= np.linspace(1, 0, fout)[:, None]
# duck under voice
e_ = np.convolve(np.abs(voice), np.ones(int(SR * 0.08)) / int(SR * 0.08), mode="same")
spk = np.clip(e_ / (e_.max() * 0.06 + 1e-9), 0, 1)
spk = np.convolve(spk, np.ones(int(SR * 0.25)) / int(SR * 0.25), mode="same")
music *= (1 - 0.6 * spk)[:, None]
music /= np.abs(music).max() + 1e-9

sf.write(f"{out}/music.wav", (music * 0.3).astype(np.float32), SR)
sf.write(f"{out}/sfx.wav", sfx, SR)

# --- voice polish + final mix with ffmpeg (presence EQ, gentle compression), then loudness to -14 LUFS
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{out}/voice.wav",
                "-af", "highpass=f=75,equalizer=f=3200:t=q:w=1.2:g=3,equalizer=f=220:t=q:w=1:g=-2,"
                       "acompressor=threshold=-20dB:ratio=3:attack=5:release=80:makeup=3dB",
                "-ar", str(SR), f"{out}/voice_fx.wav"], check=True)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{out}/voice_fx.wav", "-i", f"{out}/music.wav", "-i", f"{out}/sfx.wav",
                "-filter_complex",
                "[0:a]aformat=channel_layouts=stereo,volume=1.0[v];[1:a]volume=0.55[m];[2:a]aformat=channel_layouts=stereo,volume=0.6[s];"
                "[v][m][s]amix=inputs=3:normalize=0,alimiter=limit=0.95,loudnorm=I=-14:TP=-1.5:LRA=9[o]",
                "-map", "[o]", "-ar", str(SR), f"{out}/mix.wav"], check=True)
print(f"total {total:.2f}s, {len(shots)} shots, {len(events)} sfx events")
