"""Shared pieces of the two PDF guides: page styles, code highlighting and a small
helper for drawing box-and-arrow diagrams as inline SVG."""

from __future__ import annotations

import html
import textwrap
from typing import Dict, List, Optional, Tuple

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name

INK = "#1b2733"
MUTED = "#5a6878"
LINE = "#c9d2dc"
BLUE = "#1f5fa8"
BLUE_SOFT = "#e6eef8"
GREEN = "#1d6b55"
GREEN_SOFT = "#e2f0eb"
AMBER = "#a8641a"
AMBER_SOFT = "#f7ebdb"
ROSE = "#a6245f"
ROSE_SOFT = "#f8e6ee"
GREY_SOFT = "#f1f4f7"

_formatter = HtmlFormatter(nowrap=True, style="friendly")

CSS = """
@page { size: A4; margin: 16mm 15mm 18mm 15mm;
        @bottom-center { content: counter(page); font: 9pt "Fira Sans"; color: #5a6878; } }
* { box-sizing: border-box; }
html { font-family: "Fira Sans", "DejaVu Sans", sans-serif; font-size: 10.2pt;
       line-height: 1.5; color: #1b2733; }
body { margin: 0; }
h1 { font-size: 27pt; line-height: 1.12; margin: 0 0 6pt; font-weight: 700;
     letter-spacing: -0.01em; }
h2 { font-size: 16pt; line-height: 1.2; margin: 22pt 0 8pt; padding-top: 6pt;
     border-top: 2.5pt solid #1f5fa8; break-after: avoid; }
h2 .num { color: #1f5fa8; margin-right: 6pt; }
h3 { font-size: 11.6pt; margin: 14pt 0 4pt; break-after: avoid; }
p { margin: 0 0 7pt; }
section { break-before: auto; }
.lead { font-size: 12pt; color: #3a4755; max-width: 150mm; margin-bottom: 12pt; }
.meta { color: #5a6878; font-size: 9.2pt; margin-bottom: 16pt; }
ul, ol { margin: 0 0 8pt; padding-left: 16pt; }
li { margin-bottom: 2.5pt; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8.6pt;
       background: #f1f4f7; padding: 0.5pt 3pt; border-radius: 2pt; }
pre { font-family: "DejaVu Sans Mono", monospace; font-size: 7.9pt; line-height: 1.45;
      background: #f6f8fa; border: 0.6pt solid #c9d2dc; border-left: 2.5pt solid #1f5fa8;
      border-radius: 2pt; padding: 7pt 9pt; margin: 4pt 0 10pt; white-space: pre-wrap;
      break-inside: avoid; }
pre.out { border-left-color: #5a6878; background: #fbfbfc; color: #3a4755; }
pre code { background: none; padding: 0; font-size: inherit; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 11pt; font-size: 9.2pt; }
tr { break-inside: avoid; }
th { text-align: left; font-weight: 600; border-bottom: 1.2pt solid #1b2733;
     padding: 3pt 8pt 3pt 0; vertical-align: bottom; }
td { border-bottom: 0.6pt solid #c9d2dc; padding: 3.5pt 8pt 3.5pt 0; vertical-align: top; }
figure { margin: 8pt 0 12pt; break-inside: avoid; }
figure svg, figure img { display: block; width: 100%; height: auto; }
figure img { border: 0.6pt solid #c9d2dc; }
figcaption { font-size: 8.8pt; color: #5a6878; margin-top: 4pt; }
.note { background: #f7ebdb; border-left: 2.5pt solid #a8641a; padding: 6pt 9pt;
        margin: 6pt 0 10pt; font-size: 9.4pt; break-inside: avoid; }
.note.info { background: #e6eef8; border-left-color: #1f5fa8; }
.toc { columns: 2; column-gap: 14mm; margin: 0 0 6pt; padding: 0; list-style: none;
       font-size: 10pt; }
.toc li { break-inside: avoid; margin-bottom: 3pt; }
.toc .num { display: inline-block; width: 16pt; color: #1f5fa8; font-weight: 600; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 12pt; }
.steps { counter-reset: step; list-style: none; padding: 0; }
.steps > li { counter-increment: step; position: relative; padding-left: 22pt;
              margin-bottom: 6pt; }
.steps > li::before { content: counter(step); position: absolute; left: 0; top: 0;
    width: 15pt; height: 15pt; border-radius: 50%; background: #1f5fa8; color: #fff;
    font-size: 8.6pt; font-weight: 600; text-align: center; line-height: 15pt; }
pre .err { border: none !important; }
svg text { font-family: "Fira Sans", "DejaVu Sans", sans-serif; }
svg .mono { font-family: "DejaVu Sans Mono", monospace; }
""" + _formatter.get_style_defs("pre")


def code(source: str, lang: str = "python") -> str:
    """A highlighted code block."""
    source = textwrap.dedent(source).strip("\n")
    return f"<pre>{highlight(source, get_lexer_by_name(lang), _formatter)}</pre>"


def out(text: str) -> str:
    """Program output, not highlighted."""
    return f'<pre class="out">{html.escape(textwrap.dedent(text).strip(chr(10)))}</pre>'


def fig(svg: str, caption: str) -> str:
    return f"<figure>{svg}<figcaption>{caption}</figcaption></figure>"


def table(
    header: List[str], rows: List[List[str]], widths: Optional[List[str]] = None
) -> str:
    cols = ""
    if widths:
        cols = (
            "<colgroup>"
            + "".join(f'<col style="width:{w}">' for w in widths)
            + "</colgroup>"
        )
    head = "".join(f"<th>{h}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table>{cols}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def page(title: str, body: str) -> str:
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
        f"<body>{body}</body></html>"
    )


KINDS = {
    "plain": ("#ffffff", INK),
    "soft": (GREY_SOFT, MUTED),
    "blue": (BLUE_SOFT, BLUE),
    "green": (GREEN_SOFT, GREEN),
    "amber": (AMBER_SOFT, AMBER),
    "rose": (ROSE_SOFT, ROSE),
}


class Diagram:
    """Boxes with a title and optional lines of small text, joined by arrows."""

    def __init__(self, width: int, height: int):
        self.w, self.h = width, height
        self.boxes: Dict[str, Tuple[float, float, float, float]] = {}
        self.parts: List[str] = []

    def box(
        self,
        key: str,
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        sub: str = "",
        kind: str = "plain",
        mono: bool = False,
        db: bool = False,
    ) -> None:
        fill, stroke = KINDS[kind]
        self.boxes[key] = (x, y, w, h)
        if db:  # a database: a cylinder
            r = 7
            self.parts.append(
                f'<path d="M{x},{y + r} a{w / 2},{r} 0 0 1 {w},0 v{h - 2 * r} '
                f'a{w / 2},{r} 0 0 1 {-w},0 z" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>'
                f'<path d="M{x},{y + r} a{w / 2},{r} 0 0 0 {w},0" fill="none" '
                f'stroke="{stroke}" stroke-width="1.2"/>'
            )
        else:
            self.parts.append(
                f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{fill}" '
                f'stroke="{stroke}" stroke-width="1.2"/>'
            )
        lines = sub.split("\n") if sub else []
        total = 13 + 12 * len(lines)
        ty = y + (h - total) / 2 + 11 + (4 if db else 0)
        self.parts.append(
            f'<text x="{x + w / 2}" y="{ty}" text-anchor="middle" font-size="11.3" '
            f'font-weight="600" fill="{INK}">{html.escape(title)}</text>'
        )
        cls = ' class="mono"' if mono else ""
        size = 8.4 if mono else 9.5
        for i, line in enumerate(lines):
            self.parts.append(
                f'<text{cls} x="{x + w / 2}" y="{ty + 13.5 + 12 * i}" text-anchor="middle" '
                f'font-size="{size}" fill="{MUTED}">{html.escape(line)}</text>'
            )

    def _anchor(self, key: str, side: str) -> Tuple[float, float]:
        x, y, w, h = self.boxes[key]
        return {
            "l": (x, y + h / 2),
            "r": (x + w, y + h / 2),
            "t": (x + w / 2, y),
            "b": (x + w / 2, y + h),
        }[side]

    def arrow(
        self,
        a: str,
        b: str,
        label: str = "",
        sides: str = "",
        dashed: bool = False,
        color: str = INK,
        label_dy: float = -4,
        label_dx: float = 0,
    ) -> None:
        ax, ay, aw, ah = self.boxes[a]
        bx, by, bw, bh = self.boxes[b]
        if not sides:
            dx, dy = (bx + bw / 2) - (ax + aw / 2), (by + bh / 2) - (ay + ah / 2)
            if abs(dx) >= abs(dy):
                sides = "rl" if dx > 0 else "lr"
            else:
                sides = "bt" if dy > 0 else "tb"
        (x1, y1), (x2, y2) = self._anchor(a, sides[0]), self._anchor(b, sides[1])
        self.line(
            x1,
            y1,
            x2,
            y2,
            label,
            dashed,
            color,
            label_dy,
            label_dx,
            elbow=sides[0] in "lr",
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        label: str = "",
        dashed: bool = False,
        color: str = INK,
        label_dy: float = -4,
        label_dx: float = 0,
        elbow: bool = True,
        head: bool = True,
    ) -> None:
        if abs(x1 - x2) < 1 or abs(y1 - y2) < 1:
            d = f"M{x1},{y1} L{x2},{y2}"
        elif elbow:  # horizontal, vertical, horizontal
            mx = (x1 + x2) / 2
            d = f"M{x1},{y1} L{mx},{y1} L{mx},{y2} L{x2},{y2}"
        else:  # vertical, horizontal, vertical
            my = (y1 + y2) / 2
            d = f"M{x1},{y1} L{x1},{my} L{x2},{my} L{x2},{y2}"
        dash = ' stroke-dasharray="4 3"' if dashed else ""
        marker = f' marker-end="url(#arrow-{color[1:]})"' if head else ""
        self.parts.append(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.2"{dash}{marker}/>'
        )
        if label:
            self.text(
                (x1 + x2) / 2 + label_dx,
                (y1 + y2) / 2 + label_dy,
                label,
                size=8.4,
                color=MUTED,
                anchor="middle",
                halo=True,
            )

    def text(
        self,
        x: float,
        y: float,
        text: str,
        size: float = 9,
        color: str = INK,
        anchor: str = "start",
        weight: int = 400,
        mono: bool = False,
        halo: bool = False,
    ) -> None:
        cls = ' class="mono"' if mono else ""
        stroke = (
            ' paint-order="stroke" stroke="#ffffff" stroke-width="3"' if halo else ""
        )
        for i, line in enumerate(text.split("\n")):
            self.parts.append(
                f'<text{cls} xml:space="preserve" x="{x}" y="{y + i * size * 1.3}" text-anchor="{anchor}" '
                f'font-size="{size}" font-weight="{weight}" fill="{color}"{stroke}>'
                f"{html.escape(line)}</text>"
            )

    def circle(
        self,
        x: float,
        y: float,
        r: float,
        fill: str,
        stroke: str = "none",
        width: float = 1.2,
        label: str = "",
        label_color: str = "#ffffff",
    ) -> None:
        self.parts.append(
            f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{width}"/>'
        )
        if label:
            self.parts.append(
                f'<text x="{x}" y="{y + 3}" text-anchor="middle" font-size="8.4" '
                f'font-weight="600" fill="{label_color}">{html.escape(label)}</text>'
            )

    def edge(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str = LINE,
        width: float = 1.2,
        dashed: bool = False,
    ) -> None:
        dash = ' stroke-dasharray="3 3"' if dashed else ""
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{width}"{dash}/>'
        )

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str = "none",
        stroke: str = LINE,
        dashed: bool = False,
    ) -> None:
        dash = ' stroke-dasharray="4 3"' if dashed else ""
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="1"{dash}/>'
        )

    def svg(self) -> str:
        markers = "".join(
            f'<marker id="arrow-{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
            for c in (INK, BLUE, GREEN, AMBER, ROSE, MUTED)
        )
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}">'
            f"<defs>{markers}</defs>{''.join(self.parts)}</svg>"
        )
