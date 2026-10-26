"""Datasets, leakage-aware splitting and transforms.

PlantVillage contains several photos of the same physical leaf, so a random
image-level split can place near-identical images in train and test. When a
duplicate-group file from ``scripts/find_duplicates.py`` is available, splits are
made at group level instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Sample:
    path: str
    label: int


def list_images(root: Path, class_map: list[dict[str, str]], source: str) -> pd.DataFrame:
    """List images for every mapped class of one source.

    Args:
        root: Dataset root containing one folder per class.
        class_map: Shared label space from ``configs/class_map.yaml``.
        source: ``"plantvillage"`` or ``"plantdoc"``.

    Returns:
        DataFrame with columns ``path``, ``label``, ``class_name``.

    Raises:
        FileNotFoundError: If the root or any mapped class folder is missing.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"{source} directory not found: {root}")
    rows: list[dict[str, object]] = []
    for label, entry in enumerate(class_map):
        folder = root / entry[source]
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing {source} class folder: {folder}")
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not files:
            logger.warning("No images in %s", folder)
        rows.extend({"path": str(p), "label": label, "class_name": entry["name"]} for p in files)
    df = pd.DataFrame(rows, columns=["path", "label", "class_name"])
    logger.info("%s: %d images across %d classes", source, len(df), df["label"].nunique())
    return df


def split_plantvillage(
    df: pd.DataFrame,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    groups_csv: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Stratified train/val/test split, grouped by duplicate cluster when available.

    Whole groups are assigned to one split per class, so near-duplicates never
    straddle train and test.
    """
    if not 0 < val_fraction + test_fraction < 1:
        raise ValueError("val_fraction + test_fraction must be in (0, 1)")
    df = df.copy()
    if groups_csv is not None and groups_csv.is_file():
        groups = pd.read_csv(groups_csv)[["path", "group_id"]]
        df = df.merge(groups, on="path", how="left")
        missing = df["group_id"].isna()
        df.loc[missing, "group_id"] = "single_" + df.loc[missing, "path"]
        logger.info("Group-aware split using %s (%d groups)", groups_csv, df["group_id"].nunique())
    else:
        logger.warning("No duplicate-group file found; falling back to an image-level split. "
                       "Run scripts/find_duplicates.py first to avoid leakage.")
        df["group_id"] = df["path"]

    rng = np.random.default_rng(seed)
    assignment: dict[str, str] = {}
    for _, cls_df in df.groupby("label"):
        group_ids = cls_df["group_id"].unique()
        rng.shuffle(group_ids)
        n = len(group_ids)
        n_test = max(1, round(n * test_fraction))
        n_val = max(1, round(n * val_fraction))
        for i, gid in enumerate(group_ids):
            assignment[gid] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    df["split"] = df["group_id"].map(assignment)
    return {name: part.drop(columns="split").reset_index(drop=True) for name, part in df.groupby("split")}


def sample_fewshot(df: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """Draw ``k`` images per class (or all if fewer exist)."""
    if k <= 0:
        return df.iloc[0:0]
    parts = [g.sample(n=min(k, len(g)), random_state=seed) for _, g in df.groupby("label")]
    return pd.concat(parts).reset_index(drop=True)


def build_transforms(image_size: int, augmentation: str, train: bool) -> transforms.Compose:
    """Build torchvision transforms.

    ``strong`` adds crops, colour and blur perturbations as a cheap proxy for
    field conditions (lighting, focus, clutter).
    """
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if not train:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])
    if augmentation == "basic":
        ops = [transforms.Resize((image_size, image_size)), transforms.RandomHorizontalFlip()]
    elif augmentation == "strong":
        ops = [
            transforms.RandomResizedCrop(image_size, scale=(0.35, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(25),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.05),
            transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.3),
        ]
    else:
        raise ValueError(f"Unknown augmentation '{augmentation}'")
    return transforms.Compose([*ops, transforms.ToTensor(), normalize,
                               transforms.RandomErasing(p=0.25 if augmentation == "strong" else 0.0)])


class ImageDataset(Dataset):
    """Image dataset backed by a DataFrame with ``path`` and ``label`` columns."""

    def __init__(self, df: pd.DataFrame, transform: transforms.Compose) -> None:
        if df.empty:
            raise ValueError("Empty dataset")
        self.paths = df["path"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        try:
            with Image.open(self.paths[idx]) as img:
                image = img.convert("RGB")
        except OSError as exc:
            raise OSError(f"Cannot read image {self.paths[idx]}") from exc
        return self.transform(image), self.labels[idx]
