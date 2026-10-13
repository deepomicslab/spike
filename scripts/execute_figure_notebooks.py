"""Execute figure notebooks from fresh kernels and save their inline previews in place."""
from pathlib import Path
import argparse
import os
import tempfile

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def write_atomic(notebook, destination):
    """Replace a notebook only after its complete JSON has been written."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".ipynb",
            dir=destination.parent, delete=False,
        ) as stream:
            temporary = Path(stream.name)
            nbformat.write(notebook, stream)
        if destination.exists():
            temporary.chmod(destination.stat().st_mode & 0o777)
        else:
            temporary.chmod(0o644)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure", type=int, choices=[2, 3, 4, 5], action="append",
        help="Repeat for several figures; default: all",
    )
    args = parser.parse_args()
    os.environ["SPIKE_CODE_ROOT"] = str(ROOT)
    directory = ROOT / "outputs/executed_notebooks"
    directory.mkdir(parents=True, exist_ok=True)
    for number in (args.figure or [2, 3, 4, 5]):
        source = ROOT / f"notebooks/figure{number}.ipynb"
        original = source.read_bytes()
        notebook = nbformat.reads(original.decode("utf-8"), as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
        client = NotebookClient(
            notebook, timeout=600, kernel_name="python3",
            resources={"metadata": {"path": str(ROOT / "notebooks")}},
        )
        client.execute()
        if source.read_bytes() != original:
            raise RuntimeError(f"Notebook changed during execution; refusing to overwrite: {source}")
        write_atomic(notebook, directory / source.name)
        write_atomic(notebook, source)
        cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        images = sum("image/png" in output.get("data", {}) for cell in cells for output in cell.outputs)
        print(f"PASS: Figure {number}; {len(cells)} code cells; {images} inline images; {source}", flush=True)


if __name__ == "__main__":
    main()
