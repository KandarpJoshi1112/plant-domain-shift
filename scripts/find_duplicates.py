"""Group near-duplicate PlantVillage images with perceptual hashing.

Images whose pHash Hamming distance is <= threshold are linked; connected
components become ``group_id`` values used for leakage-free splitting.

Example:
    python scripts/find_duplicates.py --root data/raw/plantvillage/color \
        --out results/plantvillage_duplicate_groups.csv --threshold 4
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger("find_duplicates")
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def hash_folder(folder: Path) -> tuple[list[str], np.ndarray]:
    paths, hashes = [], []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in EXTS:
            continue
        try:
            with Image.open(p) as img:
                hashes.append(imagehash.phash(img).hash.flatten())
            paths.append(str(p))
        except OSError:
            logger.warning("Skipping unreadable image %s", p)
    return paths, np.array(hashes, dtype=bool)


def group_class(paths: list[str], hashes: np.ndarray, threshold: int) -> list[int]:
    """Union images within one class whose hash distance is within threshold (O(n^2) per class)."""
    uf = UnionFind(len(paths))
    for i in range(len(paths)):
        dists = np.count_nonzero(hashes[i + 1:] != hashes[i], axis=1)
        for j in np.nonzero(dists <= threshold)[0]:
            uf.union(i, i + 1 + int(j))
    return [uf.find(i) for i in range(len(paths))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--threshold", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not args.root.is_dir():
        raise SystemExit(f"Not a directory: {args.root}")

    rows = []
    for folder in sorted(p for p in args.root.iterdir() if p.is_dir()):
        paths, hashes = hash_folder(folder)
        if not paths:
            continue
        roots = group_class(paths, hashes, args.threshold)
        rows.extend({"path": p, "group_id": f"{folder.name}::{r}"} for p, r in zip(paths, roots))
        n_groups = len(set(roots))
        logger.info("%-55s %6d images -> %6d groups", folder.name, len(paths), n_groups)
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    dup_share = 1 - df["group_id"].nunique() / len(df)
    logger.info("Wrote %s; %.1f%% of images share a group with another image", args.out, 100 * dup_share)


if __name__ == "__main__":
    main()
