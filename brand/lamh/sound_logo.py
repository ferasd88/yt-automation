"""Original «لمح» sound logo (synthesized, no third-party audio): airy swish -> two-note mallet motif -> soft shimmer.
Usage: python sound_logo.py out.wav [duration]
"""
import sys

import numpy as np
from scipy.io import wavfile

SR = 48000
D = float(sys.argv[2]) if len(sys.argv) > 2 else 1.3
N = int(D * SR)
rng = np.random.default_rng(3)
out = np.zeros((N, 2), np.float32)
t = np.arange(N) / SR


def place(sig, at, pan=0.0, gain=1.0):
    i = int(at * SR)
    j = min(N, i + len(sig))
    l, r = np.sqrt((1 - pan) / 2), np.sqrt((1 + pan) / 2)
    out[i:j, 0] += gain * l * sig[: j - i]
    out[i:j, 1] += gain * r * sig[: j - i]


def lowpass(x, a):
    y = np.zeros_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc += a[i] * (x[i] - acc)
        y[i] = acc
    return y


# 1) swish (0.00-0.34s): band-limited noise with rising cutoff, swept right->left like the slashes
n = int(0.36 * SR)
sw = lowpass(rng.standard_normal(n).astype(np.float32), np.linspace(0.015, 0.22, n) ** 1.4)
sw *= np.sin(np.linspace(0, np.pi, n)) ** 2
sw /= np.abs(sw).max() + 1e-9
place(sw * 0.55, 0.0, pan=0.5)
place(sw * 0.55, 0.03, pan=-0.5)


# 2) two-note mallet motif  «لَمْ - ح» : B4 (soft) then E5 (accent) on the triangle pop
def mallet(f, dur, bright=1.0):
    m = int(dur * SR)
    tt = np.arange(m) / SR
    s = (np.sin(2 * np.pi * f * tt) * np.exp(-tt * 6)
         + 0.35 * bright * np.sin(2 * np.pi * f * 2.76 * tt) * np.exp(-tt * 18)     # marimba-like overtone
         + 0.12 * bright * np.sin(2 * np.pi * f * 5.4 * tt) * np.exp(-tt * 35))
    att = int(0.004 * SR)
    s[:att] *= np.linspace(0, 1, att)
    return s.astype(np.float32)


place(mallet(493.88, 0.9, 0.8), 0.30, pan=0.15, gain=0.32)       # B4
place(mallet(659.25, 0.95, 1.0), 0.44, pan=-0.1, gain=0.5)        # E5 (accent)
place(mallet(329.63, 0.9, 0.5), 0.44, pan=0.0, gain=0.22)         # E4 body under the accent

# soft sub thump on the pop
m = int(0.35 * SR)
tt = np.arange(m) / SR
thump = np.sin(2 * np.pi * np.cumsum(48 + 60 * np.exp(-tt * 25)) / SR) * np.exp(-tt * 11)
place(thump.astype(np.float32), 0.44, gain=0.55)

# 3) shimmer on the sheen (0.55s): very quiet high airy noise
m = int(0.6 * SR)
sh = rng.standard_normal(m).astype(np.float32)
sh = sh - lowpass(sh, np.full(m, 0.35))           # crude high-pass
sh *= np.exp(-np.arange(m) / (0.18 * SR)) * np.minimum(1, np.arange(m) / (0.05 * SR))
place(sh * 0.05, 0.55, pan=-0.3)

# short room reverb (FFT convolution), then fit inside the duration
L = int(0.9 * SR)
ir = rng.standard_normal(L).astype(np.float32) * np.exp(-np.arange(L) / (0.22 * SR))
ir /= np.sqrt((ir ** 2).sum())
nfft = 1 << int(np.ceil(np.log2(N + L)))
for c in range(2):
    wet = np.fft.irfft(np.fft.rfft(out[:, c], nfft) * np.fft.rfft(ir, nfft), nfft)[:N]
    out[:, c] = 0.82 * out[:, c] + 0.28 * wet
fo = int(0.14 * SR)
out[-fo:] *= np.linspace(1, 0, fo)[:, None] ** 1.5
out /= np.abs(out).max() / 0.7           # peak about -3 dBFS
wavfile.write(sys.argv[1], SR, out.astype(np.float32))
print(f"sound logo {D}s")
