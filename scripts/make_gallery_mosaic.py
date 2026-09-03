"""Composite several environment clips into one mosaic GIF.

A row of separate GIFs looks ragged for two reasons that no amount of HTML
fixes: their aspect ratios run from 3.0:1 to 1.5:1, so nothing lines up, and
their frame counts differ (90 to 160), so they drift out of phase and the grid
shimmers. One pre-rendered mosaic solves both -- uniform tiles, a single loop,
and one request instead of six.

    uv run python scripts/make_gallery_mosaic.py

Writes ``videos/gallery_mosaic.gif``.
"""

from __future__ import annotations

import pathlib

from PIL import Image, ImageDraw, ImageSequence

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_WEBP = ROOT / "videos" / "gallery_mosaic.webp"
OUT_GIF = ROOT / "videos" / "gallery_mosaic.gif"

# Six plants that between them show the range: an aircraft flying a 3D path, a
# furnace, a drum boiler, a reactor, a coupled tank rig and a turbine.
TILES = [
    ("videos/plane3d_figure8/pid_output.gif", "Aircraft - figure-8"),
    ("videos/glass_furnace/pid_output_short.gif", "Glass furnace"),
    ("videos/boiler_drum/pid_output_short.gif", "Boiler drum"),
    ("videos/reactor/pid_output_short.gif", "Nuclear reactor"),
    ("videos/four_tank/pid_output_short.gif", "Four-tank"),
    ("videos/wind_turbine/pid_output_short.gif", "Wind turbine"),
]

COLS, ROWS = 3, 2
TILE_W, TILE_H = (
    646,
    346,
)  # the sources' own size: no downscaling, no mush  # 1.85:1, the aspect most of the clips already are
PAD = 6
LABEL_H = 22
BG = (255, 255, 255)
LABEL_BG = (33, 33, 33)
LABEL_FG = (255, 255, 255)
N_FRAMES = 40  # enough to read the motion; the file is a hero image, not a demo
DURATION_MS = 100
COLORS = 64  # the palette scripts/shorten_gifs.py already uses for the gallery
QUALITY = 72


def _frames(path: pathlib.Path, n: int) -> list[Image.Image]:
    """``n`` frames from a clip, letterboxed onto a uniform tile.

    Sources are cycled rather than stretched: a clip with 150 frames is sampled
    ``i % 150`` so every tile advances at the same rate and the mosaic loops as
    one animation.
    """
    src = Image.open(path)
    raw = [f.convert("RGB") for f in ImageSequence.Iterator(src)]
    out = []
    for i in range(n):
        frame = raw[i % len(raw)]
        scale = min(TILE_W / frame.width, (TILE_H - LABEL_H) / frame.height)
        size = (max(int(frame.width * scale), 1), max(int(frame.height * scale), 1))
        tile = Image.new("RGB", (TILE_W, TILE_H), BG)
        tile.paste(
            frame.resize(size, Image.LANCZOS),
            ((TILE_W - size[0]) // 2, (TILE_H - LABEL_H - size[1]) // 2),
        )
        out.append(tile)
    return out


def main() -> int:
    clips, labels = [], []
    for rel, label in TILES:
        path = ROOT / rel
        if not path.exists():
            print(f"  missing, skipped: {rel}")
            continue
        clips.append(_frames(path, N_FRAMES))
        labels.append(label)

    if not clips:
        print("no clips found")
        return 1

    width = COLS * TILE_W + (COLS + 1) * PAD
    height = ROWS * TILE_H + (ROWS + 1) * PAD
    frames = []
    for i in range(N_FRAMES):
        canvas = Image.new("RGB", (width, height), BG)
        draw = ImageDraw.Draw(canvas)
        for k, (clip, label) in enumerate(zip(clips, labels)):
            col, row = k % COLS, k // COLS
            x = PAD + col * (TILE_W + PAD)
            y = PAD + row * (TILE_H + PAD)
            canvas.paste(clip[i], (x, y))
            draw.rectangle(
                [x, y + TILE_H - LABEL_H, x + TILE_W, y + TILE_H], fill=LABEL_BG
            )
            draw.text((x + 6, y + TILE_H - LABEL_H + 4), label, fill=LABEL_FG)
        frames.append(canvas)

    OUT_WEBP.parent.mkdir(parents=True, exist_ok=True)
    # Animated WebP, not GIF. GIF caps at 256 colours and compresses these
    # instrument panels badly, so fitting one under a few megabytes meant
    # downscaling the tiles to 400 px and losing the text on every gauge. WebP
    # keeps the sources at their own 646x346 and still comes out smaller.
    frames[0].save(
        OUT_WEBP,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=DURATION_MS,
        loop=0,
        quality=QUALITY,
        method=6,
    )
    mb = OUT_WEBP.stat().st_size / 1e6
    print(
        f"  wrote {OUT_WEBP.relative_to(ROOT)}  {width}x{height}  {len(frames)} frames  {mb:.2f} MB"
    )

    # A GIF alongside it, downscaled, for anywhere WebP is not rendered.
    small = [f.resize((width // 2, height // 2), Image.LANCZOS) for f in frames[::2]]
    palette = small[0].quantize(colors=COLORS, method=Image.MEDIANCUT)
    quantised = [
        f.quantize(palette=palette, dither=Image.FLOYDSTEINBERG) for f in small
    ]
    quantised[0].save(
        OUT_GIF,
        save_all=True,
        append_images=quantised[1:],
        duration=DURATION_MS * 2,
        loop=0,
        optimize=True,
    )
    print(
        f"  wrote {OUT_GIF.relative_to(ROOT)}  fallback  "
        f"{OUT_GIF.stat().st_size / 1e6:.2f} MB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
