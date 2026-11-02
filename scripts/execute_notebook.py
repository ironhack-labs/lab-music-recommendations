"""Run and save the deliverable with its optional Pinecone section disabled."""
import asyncio
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("IPYTHONDIR", str(ROOT / ".cache/ipython"))
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(ROOT / ".cache/jupyter"))
for variable in ("IPYTHONDIR", "JUPYTER_RUNTIME_DIR"):
    Path(os.environ[variable]).mkdir(parents=True, exist_ok=True)
if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
path = ROOT / "lab-music-recommendations.ipynb"
notebook = nbformat.read(path, as_version=4)
NotebookClient(notebook, timeout=180, kernel_name="python3",
               resources={"metadata": {"path": str(ROOT)}}).execute()
nbformat.write(notebook, path)
print(f"Executed and saved {path.name} ({len(notebook.cells)} cells)")
