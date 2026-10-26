"""Verify that every class in configs/class_map.yaml exists in both datasets and report counts.

Example:
    python scripts/check_class_map.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdshift.config import load_class_map, load_config  # noqa: E402
from pdshift.data import IMAGE_EXTENSIONS  # noqa: E402

logger = logging.getLogger("check_class_map")
MIN_TEST_IMAGES = 5


def count(folder: Path) -> int | None:
    if not folder.is_dir():
        return None
    return sum(1 for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cfg = load_config(args.config)
    paths = cfg["paths"]
    roots = {"PV": (Path(paths["plantvillage_dir"]), "plantvillage"),
             "PD-train": (Path(paths["plantdoc_train_dir"]), "plantdoc"),
             "PD-test": (Path(paths["plantdoc_test_dir"]), "plantdoc")}
    problems = 0
    logger.info("%-26s %7s %9s %8s", "class", "PV", "PD-train", "PD-test")
    for entry in load_class_map(paths["class_map"]):
        counts = {k: count(root / entry[src]) for k, (root, src) in roots.items()}
        flag = ""
        if any(v is None for v in counts.values()):
            flag, problems = "  MISSING FOLDER", problems + 1
        elif counts["PD-test"] < MIN_TEST_IMAGES:
            flag = f"  <{MIN_TEST_IMAGES} test images: per-class recall will be unreliable"
        logger.info("%-26s %7s %9s %8s%s", entry["name"], *(counts[k] if counts[k] is not None else "-"
                                                          for k in roots), flag)
    if problems:
        raise SystemExit(f"{problems} class(es) have missing folders; fix configs/class_map.yaml")
    logger.info("Class map OK")


if __name__ == "__main__":
    main()
