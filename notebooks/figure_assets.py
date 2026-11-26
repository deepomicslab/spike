"""Verify a bundled manuscript image and display a notebook-sized preview."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

from IPython.display import Image as NotebookImage, display
from PIL import Image

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "manuscript"


def show_figure(filename):
    expected = json.loads((ASSETS / "sha256.json").read_text())[filename]
    raw = (ASSETS / filename).read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError(f"Image checksum mismatch for {filename}")
    with Image.open(BytesIO(raw)) as image:
        image.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
        preview = BytesIO()
        image.save(preview, format="PNG")
    display(NotebookImage(data=preview.getvalue(), format="png"))
