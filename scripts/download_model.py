"""Descarga el modelo FaceLandmarker de MediaPipe en models/."""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alertdrive.config import DEFAULT_MODEL_PATH, MODEL_URL  # noqa: E402


def main() -> None:
    if DEFAULT_MODEL_PATH.exists():
        print(f"El modelo ya existe: {DEFAULT_MODEL_PATH}")
        return
    DEFAULT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando {MODEL_URL} ...")
    urllib.request.urlretrieve(MODEL_URL, DEFAULT_MODEL_PATH)
    print(f"Listo: {DEFAULT_MODEL_PATH} ({DEFAULT_MODEL_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
