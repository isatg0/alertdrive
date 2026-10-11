"""Manifiestos de datasets: una fila por video con sujeto y etiqueta.

El manifiesto desacopla el formato de cada dataset del resto del pipeline.
Guardar el ``subject_id`` es indispensable: la partición train/test de la
Fase 2 se hace por sujeto para que el modelo nunca se evalúe sobre una
persona que ya vio durante el entrenamiento.

Uso::

    python -m alertdrive.data.manifest --dataset uta-rldd --root /datos/UTA-RLDD --out manifests/uta.csv
    python -m alertdrive.data.manifest --dataset yawdd --root /datos/YawDD --out manifests/yawdd.csv

IMPORTANTE: los patrones de nombres se basan en la estructura publicada de cada
dataset. Verifica el resumen que imprime el comando (conteo por etiqueta y
sujetos) contra la documentación del dataset que descargues.
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".m4v", ".mkv", ".wmv"}
MANIFEST_FIELDS = ("video_path", "subject_id", "label", "dataset", "frame_labels_path")


@dataclass(frozen=True)
class ManifestEntry:
    video_path: str
    subject_id: str
    label: str
    dataset: str
    frame_labels_path: str = ""


def _videos(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)


# ---------------------------------------------------------------- UTA-RLDD
# Estructura: <Fold>/<sujeto>/{0,5,10}.<ext>; 0 = alerta, 5 = baja vigilancia, 10 = somnoliento.
UTA_LABELS = {"0": "alert", "5": "low_vigilant", "10": "drowsy"}


def scan_uta_rldd(root: Path) -> list[ManifestEntry]:
    entries = []
    for video in _videos(root):
        label = UTA_LABELS.get(video.stem)
        if label is None:
            logger.warning("UTA-RLDD: se ignora %s (nombre no reconocido)", video)
            continue
        entries.append(ManifestEntry(str(video), f"uta_{video.parent.name}", label, "uta-rldd"))
    return entries


# ---------------------------------------------------------------- YawDD
# Carpeta "Mirror": <n>-<Male|Female><NoGlasses|Glasses|SunGlasses>-<Normal|Talking|Yawning>.avi
# La numeración se reinicia por género, por eso el sujeto combina género y número.
# Los videos de la carpeta "Dash" no traen etiqueta en el nombre y se ignoran.
_YAWDD_RE = re.compile(r"^(\d+)-(Male|Female)[A-Za-z]*-(Normal|Talking|Yawning)", re.IGNORECASE)


def scan_yawdd(root: Path) -> list[ManifestEntry]:
    entries = []
    for video in _videos(root):
        m = _YAWDD_RE.match(video.stem)
        if m is None:
            logger.warning("YawDD: se ignora %s (sin etiqueta en el nombre)", video.name)
            continue
        num, gender, label = m.groups()
        entries.append(ManifestEntry(str(video), f"yawdd_{gender.lower()}{num}", label.lower(), "yawdd"))
    return entries


SCANNERS = {"uta-rldd": scan_uta_rldd, "yawdd": scan_yawdd}


def write_manifest(entries: list[ManifestEntry], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for e in entries:
            writer.writerow(asdict(e))


def read_manifest(path: Path) -> list[ManifestEntry]:
    """Lee un manifiesto CSV. Admite manifiestos hechos a mano (p. ej. NTHU-DDD con
    ``frame_labels_path`` apuntando al archivo de anotación por frame)."""
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    missing = {"video_path", "subject_id", "label"} - set(rows[0].keys() if rows else ())
    if missing:
        raise ValueError(f"Faltan columnas en el manifiesto: {sorted(missing)}")
    return [
        ManifestEntry(
            video_path=r["video_path"],
            subject_id=r["subject_id"],
            label=r["label"],
            dataset=r.get("dataset") or "custom",
            frame_labels_path=r.get("frame_labels_path") or "",
        )
        for r in rows
    ]


def summarize(entries: list[ManifestEntry]) -> str:
    labels = Counter(e.label for e in entries)
    subjects = {e.subject_id for e in entries}
    return f"{len(entries)} videos, {len(subjects)} sujetos, etiquetas: {dict(labels)}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera un manifiesto CSV a partir de un dataset.")
    parser.add_argument("--dataset", choices=sorted(SCANNERS), required=True)
    parser.add_argument("--root", type=Path, required=True, help="Carpeta raíz del dataset descargado")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    entries = SCANNERS[args.dataset](args.root)
    if not entries:
        raise SystemExit(f"No se encontraron videos válidos en {args.root}")
    write_manifest(entries, args.out)
    print(f"Manifiesto escrito en {args.out}: {summarize(entries)}")


if __name__ == "__main__":
    main()
