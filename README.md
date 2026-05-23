# 3D Ray Caster

A small Python ray-traced demo with a sample scene, checkerboard floor, lighting, and keyboard camera controls.

## Run

```bash
pip install -r requirements.txt
copy .env.example .env
python 3dRender.py
```

## Controls

- `W/S/A/D` move
- `Q/E` move up/down
- Arrow keys look around

## Environment

Copy `.env.example` to `.env` to change resolution, camera position, and frame rate limits.

Set `QUALITY=low|medium|high|ultra` to scale the internal render resolution. If `QUALITY` is set, it takes priority over `RENDER_WIDTH` / `RENDER_HEIGHT`.
