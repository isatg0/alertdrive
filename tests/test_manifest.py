from pathlib import Path

from alertdrive.data.manifest import read_manifest, scan_uta_rldd, scan_yawdd, write_manifest


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def test_uta_rldd(tmp_path):
    for subj in ("01", "02"):
        for stem in ("0", "5", "10"):
            _touch(tmp_path / "Fold1_part1" / subj / f"{stem}.mov")
    _touch(tmp_path / "Fold1_part1" / "01" / "extra.mov")
    entries = scan_uta_rldd(tmp_path)
    assert len(entries) == 6
    assert {e.label for e in entries} == {"alert", "low_vigilant", "drowsy"}
    assert {e.subject_id for e in entries} == {"uta_01", "uta_02"}


def test_yawdd_and_round_trip(tmp_path):
    for name in ("1-FemaleNoGlasses-Normal.avi", "1-FemaleNoGlasses-Yawning.avi",
                 "1-MaleGlasses-Talking.avi", "3-FemaleSunGlasses.avi"):
        _touch(tmp_path / "Mirror" / name)
    entries = scan_yawdd(tmp_path)
    assert len(entries) == 3  # el video sin etiqueta se ignora
    assert {e.subject_id for e in entries} == {"yawdd_female1", "yawdd_male1"}
    out = tmp_path / "m.csv"
    write_manifest(entries, out)
    assert read_manifest(out) == entries
