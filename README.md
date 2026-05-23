# 3D Ray Caster

A small Python ray-traced demo with a sample scene, checkerboard floor, lighting, and keyboard camera controls.
The sample scene mixes matte, glossy, and metallic materials.

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

`RENDER_BACKEND=auto` will use CUDA when CuPy is installed, otherwise CPU. `ENABLE_SHADOWS=0` is the fastest mode; turn it on for nicer lighting.

Set `AA_SAMPLES=1` for no anti-aliasing, or `2` / `4` for smoother edges at higher cost.

For best speed, keep `QUALITY=medium` or lower, `ENABLE_SHADOWS=0`, and `AA_SAMPLES=1`.
