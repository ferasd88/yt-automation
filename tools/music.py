"""Generate an original ambient space pad (no third-party music, so no licensing risk) and mix it under the voice.

Usage: python tools/music.py out/ep   (reads voice.wav, writes mix.wav at 48 kHz stereo)
"""
import sys

import numpy as np
import soundfile as sf

SR = 48000
ep = sys.argv[1]
voice, vsr = sf.read(f"{ep}/voice.wav", dtype="float32")
if voice.ndim > 1:
    voice = voice.mean(axis=1)
# resample voice 24k -> 48k
x = np.linspace(0, len(voice) - 1, int(len(voice) * SR / vsr))
voice = np.interp(x, np.arange(len(voice)), voice).astype(np.float32)
N = len(voice)
T = np.arange(N) / SR
rng = np.random.default_rng(11)


def hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


# A minor progression, 10 s per chord: Am - Fmaj7 - C - G(add9)
chords = [[45, 57, 60, 64, 69], [41, 57, 60, 64, 65], [48, 55, 60, 64, 67], [43, 55, 59, 62, 69]]
CH = 10.0
pad = np.zeros((N, 2), np.float32)
for ci in range(int(np.ceil(N / SR / CH)) + 1):
    notes = chords[ci % len(chords)]
    s0 = int((ci * CH - 2.5) * SR)
    length = int((CH + 5.0) * SR)
    a, b = max(0, s0), min(N, s0 + length)
    if a >= b:
        continue
    tt = (np.arange(a, b) - s0) / SR
    env = np.clip(tt / 3.5, 0, 1) * np.clip((CH + 5.0 - tt) / 3.5, 0, 1)
    env = env * env * (3 - 2 * env)  # smoothstep crossfade
    for j, m in enumerate(notes):
        f = hz(m)
        amp = 0.16 if j == 0 else 0.09
        for side, det in ((0, -0.18), (1, 0.18)):  # slight detune -> stereo width
            ph = rng.uniform(0, 2 * np.pi)
            w = 2 * np.pi * (f + det) * tt + ph
            tone = np.sin(w) + 0.22 * np.sin(2 * w) + 0.08 * np.sin(3 * w)
            pad[a:b, side] += (amp * env * tone).astype(np.float32)

# slow breathing + gentle high shimmer
pad *= (0.8 + 0.2 * np.sin(2 * np.pi * T / 17.0))[:, None]
noise = rng.standard_normal((N, 2)).astype(np.float32)
k = int(SR * 0.004)
noise = np.cumsum(noise, axis=0)
noise -= np.convolve(noise[:, 0], np.ones(k) / k, mode="same")[:, None]
pad += 0.004 * noise * (0.5 + 0.5 * np.sin(2 * np.pi * T / 23.0))[:, None]


# simple FFT reverb: exponentially decaying noise impulse, 4 s
def reverb(sig, seconds=4.0, mix=0.45):
    L = int(SR * seconds)
    ir = rng.standard_normal(L).astype(np.float32) * np.exp(-np.arange(L) / (SR * 1.1))
    ir /= np.sqrt((ir ** 2).sum())
    n = 1 << int(np.ceil(np.log2(len(sig) + L)))
    out = np.fft.irfft(np.fft.rfft(sig, n) * np.fft.rfft(ir, n), n)[: len(sig)]
    return (1 - mix) * sig + mix * out.astype(np.float32)


pad = np.stack([reverb(pad[:, 0]), reverb(pad[:, 1])], axis=1)
# low-pass by moving average (soften)
k = 6
pad = np.stack([np.convolve(pad[:, c], np.ones(k) / k, mode="same") for c in range(2)], axis=1)

# fade in/out
fin, fout = int(2.5 * SR), int(4.0 * SR)
pad[:fin] *= np.linspace(0, 1, fin)[:, None]
pad[-fout:] *= np.linspace(1, 0, fout)[:, None]

# duck music under the voice (smoothed envelope, ~ -7 dB while speaking)
env = np.abs(voice)
win = int(SR * 0.25)
env = np.convolve(env, np.ones(win) / win, mode="same")
speaking = np.clip(env / (env.max() * 0.08 + 1e-9), 0, 1)
smooth = int(SR * 0.6)
speaking = np.convolve(speaking, np.ones(smooth) / smooth, mode="same")
duck = 1.0 - 0.55 * speaking

pad /= np.abs(pad).max() + 1e-9
music_level = 0.24
mix = voice[:, None] * 0.92 + pad * (music_level * duck)[:, None]
mix /= max(1.0, np.abs(mix).max() / 0.97)
sf.write(f"{ep}/mix.wav", mix.astype(np.float32), SR)
sf.write(f"{ep}/music_only.wav", (pad * music_level).astype(np.float32), SR)
print(f"mix {N / SR:.1f}s")
