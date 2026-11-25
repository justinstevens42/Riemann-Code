"""Regenerate the slides layout for finding_zeta_zeros.py from its cell names.

marimo's slides layout stores one entry per cell, in notebook order, so it has
to be rebuilt whenever cells are added or removed. The convention:

- a cell named ``frag_*`` is revealed step by step on the slide before it;
- every other cell with output starts a new slide;
- cells with no output (the hidden definitions) are skipped automatically.

Run from the repo root:

    uv run python make_slides_layout.py
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).parent
NOTEBOOK = HERE / "finding_zeta_zeros.py"
LAYOUT = HERE / "layouts" / "finding_zeta_zeros.slides.json"


def main():
    source = NOTEBOOK.read_text(encoding="utf-8")
    names = re.findall(r"^@app\.cell(?:\([^)]*\))?\s*\ndef (\w+)\(", source, flags=re.M)
    cells = [{"type": "fragment"} if name.startswith("frag_") else {} for name in names]
    layout = {"type": "slides",
              "data": {"cells": cells, "deck": {"transition": "slide", "verticalAlign": "top"}}}
    LAYOUT.parent.mkdir(exist_ok=True)
    LAYOUT.write_text(json.dumps(layout, indent=2) + "\n", encoding="utf-8")
    fragments = sum(1 for c in cells if c)
    print(f"{LAYOUT.relative_to(HERE)}: {len(cells)} cells, {fragments} fragments")


if __name__ == "__main__":
    main()
