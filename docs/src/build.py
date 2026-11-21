#!/usr/bin/env python
"""Builds the two PDF guides in docs/ from the sources in docs/src/.

    python docs/src/build.py

Needs pygments and Google Chrome (or Chromium) for printing HTML to PDF.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import doc_explained  # noqa: E402
import doc_getting_started  # noqa: E402
from common import page  # noqa: E402

GUIDES = {
    "fastgraphml-explained.pdf": doc_explained,
    "fastgraphml-getting-started.pdf": doc_getting_started,
}


def main() -> None:
    chrome = next(
        (
            c
            for c in ("google-chrome", "chromium", "chromium-browser")
            if shutil.which(c)
        ),
        None,
    )
    if chrome is None:
        sys.exit("Google Chrome or Chromium is needed to print the guides to PDF.")
    for name, module in GUIDES.items():
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "guide.html"
            source.write_text(page(module.TITLE, module.build()), encoding="utf-8")
            target = HERE.parent / name
            subprocess.run(
                [
                    chrome,
                    "--headless=new",
                    "--no-sandbox",
                    "--disable-gpu",
                    "--no-pdf-header-footer",
                    f"--print-to-pdf={target}",
                    str(source),
                ],
                check=True,
                capture_output=True,
            )
            print(f"wrote {target}")


if __name__ == "__main__":
    main()
