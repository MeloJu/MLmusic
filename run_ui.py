"""Launch the Streamlit UI in a Windows-friendly way.

On Windows it's very common to have multiple Pythons (Conda base, system Python,
project .venv). If Streamlit runs under the wrong interpreter, imports like
torch/chromadb can fail.

This launcher detects the project `.venv` and (when needed) re-executes itself
using `./.venv/Scripts/python.exe`.

Run:
    python run_ui.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parent
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"

    # If we're not running inside the venv, restart using venv python.
    # Prevent infinite recursion via env var.
    already_reexec = os.environ.get("MLMUSICS_UI_REEXEC") == "1"
    if venv_python.exists() and not already_reexec:
        current = Path(sys.executable).resolve()
        if current != venv_python.resolve():
            env = dict(os.environ)
            env["MLMUSICS_UI_REEXEC"] = "1"
            return subprocess.call([str(venv_python), str(project_root / "run_ui.py")], env=env)

    try:
        from streamlit.web import cli as stcli
    except Exception as exc:
        raise SystemExit(
            "streamlit is not installed in this Python environment. "
            "Install requirements.txt in the project's venv."
        ) from exc

    sys.argv = ["streamlit", "run", "app.py"]
    return stcli.main()


if __name__ == "__main__":
    raise SystemExit(main())
