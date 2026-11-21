# yt-automation

Production pipeline for short vertical videos, run on GitHub Actions.

- `tools/` — narration (Chatterbox, best-of-N with Whisper scoring), licensed-image fetching, HTML/Chromium frame rendering, synthesized music/SFX, FFmpeg mix and automated QA.
- `brand/` — channel brand assets and outro.
- `content/` — private content, encrypted (AES-256). Outputs are published encrypted to `out-<id>` branches.
