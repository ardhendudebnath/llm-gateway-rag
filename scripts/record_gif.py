"""Render the demo walkthrough as an animated GIF for the README, from a real run.

    python scripts/record_gif.py -- --admin-token <token> --pace 1.2      # the Kubernetes stack
    python scripts/record_gif.py -- --base-url http://localhost:7860 --key ng_...

Everything after ``--`` goes to scripts/demo.py. This runs it, timestamps every line it prints and
replays them as terminal frames at roughly the speed they appeared. Nothing is staged: the GIF
shows exactly what the run printed. Credentials passed here never appear in it; only the command
given by --command is drawn. Needs Pillow (installed with the ``embeddings`` extra).
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]

BG, BAR, FG, DIM = "#1a1b26", "#16161e", "#c0caf5", "#6b7089"
ANSI = {"32": "#9ece6a", "33": "#e0af68", "36": "#7dcfff"}
PROMPT = "#7aa2f7"
FONTS = [  # regular, bold
    ("C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/consolab.ttf"),
    (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    ),
    ("/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Menlo.ttc"),
]
SIZE, LINE_H, PAD, BAR_H = 16, 22, 18, 30
MERGE_S, MIN_S, MAX_S, HOLD_S = 0.05, 0.1, 1.6, 6.0

Span = tuple[str, str, bool]  # text, colour, bold


def load_fonts() -> tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
    for regular, bold in FONTS:
        if Path(regular).exists():
            return ImageFont.truetype(regular, SIZE), ImageFont.truetype(bold, SIZE)
    font = ImageFont.load_default(size=SIZE)
    return font, font


def parse_ansi(line: str) -> list[Span]:
    spans, colour, bold, dim = [], FG, False, False
    for i, part in enumerate(re.split(r"\x1b\[([0-9;]*)m", line)):
        if i % 2 == 0:
            if part:
                spans.append((part, DIM if dim and colour == FG else colour, bold))
            continue
        for code in part.split(";") or ["0"]:
            if code in ("", "0"):
                colour, bold, dim = FG, False, False
            elif code == "1":
                bold = True
            elif code == "2":
                dim = True
            elif code in ANSI:
                colour = ANSI[code]
    return spans


def capture(demo_args: list[str]) -> list[tuple[float, str]]:
    """Run the walkthrough and return (seconds since start, line) for each printed line."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    cmd = [sys.executable, str(ROOT / "scripts" / "demo.py"), *demo_args]
    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", env=env
    ) as proc:
        start, lines = time.monotonic(), []
        for line in proc.stdout:
            lines.append((time.monotonic() - start, line.rstrip("\n")))
    if proc.returncode != 0:
        sys.exit(
            "the walkthrough failed, so there is nothing to record:\n"
            + "\n".join(text for _, text in lines)
        )
    return lines


class Terminal:
    def __init__(self, cols: int, rows: int, title: str):
        self.regular, self.bold = load_fonts()
        self.char_w = self.regular.getlength("M")
        self.cols = cols
        self.size = (int(PAD * 2 + cols * self.char_w), BAR_H + PAD * 2 + rows * LINE_H)
        self.title = title

    def frame(self, rows: list[list[Span]]) -> Image.Image:
        img = Image.new("RGB", self.size, BG)
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, self.size[0], BAR_H), fill=BAR)
        for i, dot in enumerate(("#f7768e", "#e0af68", "#9ece6a")):
            x = 16 + i * 20
            draw.ellipse((x, 10, x + 11, 21), fill=dot)
        width = self.regular.getlength(self.title)
        draw.text(((self.size[0] - width) / 2, 7), self.title, font=self.regular, fill=DIM)
        for r, spans in enumerate(rows):
            x, y, used = PAD, BAR_H + PAD + r * LINE_H, 0
            for text, colour, bold in spans:
                text = text[: max(0, self.cols - used)]
                draw.text((x, y), text, font=self.bold if bold else self.regular, fill=colour)
                x += len(text) * self.char_w
                used += len(text)
        return img


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "demo.gif")
    parser.add_argument("--command", default="python scripts/demo.py", help="what the GIF shows")
    parser.add_argument("--cols", type=int, default=88)
    parser.add_argument("demo_args", nargs=argparse.REMAINDER, help="-- then scripts/demo.py args")
    args = parser.parse_args()
    demo_args = args.demo_args[1:] if args.demo_args[:1] == ["--"] else args.demo_args

    lines = capture(demo_args)
    prompt: list[Span] = [("$ ", PROMPT, True)]
    term = Terminal(args.cols, len(lines) + 2, "nexusgate: demo walkthrough")
    frames: list[tuple[Image.Image, float]] = []

    for n in range(0, len(args.command) + 1, 3):  # type the command
        frames.append((term.frame([[*prompt, (args.command[:n], FG, False)]]), 0.06))
    typed = [*prompt, (args.command, FG, False)]
    frames.append((term.frame([typed]), 0.5))

    # One frame per burst of output; a GIF frame shorter than ~20 ms plays as 100 ms in browsers.
    shown = 0
    while shown < len(lines):
        end = shown + 1
        while end < len(lines) and lines[end][0] - lines[end - 1][0] < MERGE_S:
            end += 1
        gap = lines[end][0] - lines[end - 1][0] if end < len(lines) else HOLD_S
        rows = [typed, *(parse_ansi(text) for _, text in lines[:end])]
        frames.append(
            (term.frame(rows), HOLD_S if end == len(lines) else min(MAX_S, max(MIN_S, gap)))
        )
        shown = end

    palette = frames[-1][0].quantize(colors=64)
    images = [img.quantize(palette=palette, dither=Image.Dither.NONE) for img, _ in frames]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        args.out,
        save_all=True,
        append_images=images[1:],
        duration=[round(seconds * 1000) for _, seconds in frames],
        loop=0,
        optimize=True,
    )
    seconds = sum(s for _, s in frames)
    kib = args.out.stat().st_size / 1024
    print(f"{args.out}: {len(images)} frames, {seconds:.0f} s, {kib:.0f} KiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
