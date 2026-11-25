"""Store the slide layout inside finding_zeta_zeros.py, built from its cell names.

marimo's slides layout lists one entry per cell, in notebook order, so it has to
be rebuilt whenever cells are added or removed. The convention:

- a cell named ``frag_*`` is revealed step by step on the slide before it;
- every other cell with output starts a new slide;
- cells with no output (the hidden definitions) are skipped automatically.

The layout is written into the notebook itself, as a data URI in its
``layout_file=`` argument, so the deck opens as slides wherever the single file
goes (molab, a download, any folder).

Run from the repository folder:

    uv run python make_slides_layout.py
"""

import base64
import json
import re
from pathlib import Path

NOTEBOOK = Path(__file__).parent / "finding_zeta_zeros.py"
CELL_NAME = re.compile(r"^@app\.cell(?:\([^)]*\))?\s*\n(?:async\s+)?def (\w+)\(", re.M)
LAYOUT_ARG = re.compile(r'layout_file=(?:"[^"]*"|\((?:\s*"[^"]*")+\s*\))')


def main():
    source = NOTEBOOK.read_text(encoding="utf-8")
    names = CELL_NAME.findall(source)
    cells = [{"type": "fragment"} if name.startswith("frag_") else {} for name in names]
    layout = {"type": "slides",
              "data": {"cells": cells, "deck": {"transition": "slide", "verticalAlign": "top"}}}
    encoded = base64.b64encode(json.dumps(layout, separators=(",", ":")).encode()).decode()
    chunks = "".join(f'        "{encoded[i:i + 64]}"\n' for i in range(0, len(encoded), 64))
    argument = f'layout_file=(\n        "data:application/json;base64,"\n{chunks}    )'
    updated, count = LAYOUT_ARG.subn(lambda _: argument, source)
    if count != 1:
        raise SystemExit(f"expected one layout_file= argument in {NOTEBOOK.name}, found {count}")
    NOTEBOOK.write_text(updated, encoding="utf-8", newline="\n")
    fragments = sum(1 for cell in cells if cell)
    print(f"{NOTEBOOK.name}: layout for {len(cells)} cells ({fragments} fragments) stored inline")


if __name__ == "__main__":
    main()
