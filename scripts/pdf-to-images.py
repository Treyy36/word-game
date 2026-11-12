"""Render the menu PDFs in assets/ to PNGs for the itinerary screen.

Browsers render PDFs inconsistently (iOS Safari in particular opens them in a
separate viewer), so the itinerary shows images instead. Re-run this if the
PDFs change.

    python scripts/pdf-to-images.py
"""

import pathlib
import sys

try:
    import pymupdf
except ImportError:
    sys.exit("Missing dependency: python -m pip install pymupdf")

DPI = 150

JOBS = [
    ("assets/website+menu.pdf", "assets/menu"),
    ("assets/HH+2020.pdf", "assets/happy-hour"),
]


def render(pdf_path: str, out_prefix: str, dpi: int = DPI) -> None:
    source = pathlib.Path(pdf_path)
    if not source.exists():
        print(f"skip (missing): {source}")
        return

    doc = pymupdf.open(source)
    print(f"{source}: {doc.page_count} page(s)")

    for index, page in enumerate(doc, start=1):
        pixmap = page.get_pixmap(dpi=dpi)
        out = pathlib.Path(f"{out_prefix}-{index}.png")
        pixmap.save(out)
        print(f"  wrote {out} ({pixmap.width}x{pixmap.height})")

    doc.close()


if __name__ == "__main__":
    for pdf, prefix in JOBS:
        render(pdf, prefix)
