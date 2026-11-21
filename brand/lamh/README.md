# «لمح» outro (v1, pending approval)

- `logo_src.png`: original logo (transparent PNG, unmodified). The final frames show this file as-is.
- `L_*.png`: the same logo split into layers by colour/region for animation (letters, top slash, bottom slash, play triangle).
- `outro.html`: deterministic motion (render(t)); `sound_logo.py`: original synthesized sound logo; `make_outro.py`: renders 1080x1920@30 MP4.
- Timing (1.3 s): slashes fly in 0.00-0.34 · letters wipe right->left 0.12-0.50 · triangle pop + 2nd note 0.42 · sheen 0.55-1.00 · exact original logo from 0.95.

Not yet wired into the Shorts pipeline (step 2 after Firas approves).
