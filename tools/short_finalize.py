"""Append the fixed «لمح» outro to a rendered Short with a quick, smooth handoff.

Usage: python tools/short_finalize.py <story.mp4> <outro.mp4> <final.mp4>
The story already ends ~0.55 s after its last word; a 0.15 s cross-dissolve (video) and crossfade (audio)
starts at the very end, so the outro begins right after the story without covering it.
"""
import re
import subprocess
import sys

story, outro, final = sys.argv[1:4]
XF = 0.15


def duration(p):
    info = subprocess.run(["ffmpeg", "-hide_banner", "-i", p], capture_output=True, text=True).stderr
    h, m, s = re.search(r"Duration: (\d+):(\d+):([\d.]+)", info).groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


d = duration(story)
off = max(0.0, d - XF)
fc = (f"[0:v]fps=30,format=yuv420p,settb=AVTB[v0];[1:v]fps=30,format=yuv420p,settb=AVTB[v1];"
      f"[v0][v1]xfade=transition=fade:duration={XF}:offset={off:.3f}[v];"
      f"[0:a]aresample=48000,aformat=channel_layouts=stereo[a0];[1:a]aresample=48000,aformat=channel_layouts=stereo[a1];"
      f"[a0][a1]acrossfade=d={XF}:c1=tri:c2=tri,loudnorm=I=-14:TP=-2:LRA=11,aresample=48000,alimiter=limit=0.75:level=disabled:attack=1:release=40[a]")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", story, "-i", outro, "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-profile:v", "high",
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", final], check=True)
print(f"story {d:.2f}s + outro -> {duration(final):.2f}s")
