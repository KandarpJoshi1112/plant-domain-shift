"""Train on PlantVillage (lab), evaluate on PlantVillage test and PlantDoc test (field).

Example:
    python -m pdshift.train --config configs/base.yaml \
        --override configs/experiments/e2_resnet18.yaml --seed 0
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from pdshift.config import load_class_map, load_config
from pdshift.data import (ImageDataset, build_transforms, list_images, sample_fewshot,
                          split_plantvillage)
from pdshift.metrics import classification_report
from pdshift.models import build_model

logger = logging.getLogger("pdshift")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _loader(df: pd.DataFrame, cfg: dict[str, Any], train: bool) -> DataLoader:
    tf = build_transforms(cfg["data"]["image_size"], cfg["train"]["augmentation"], train)
    return DataLoader(ImageDataset(df, tf), batch_size=cfg["data"]["batch_size"], shuffle=train,
                      num_workers=cfg["data"]["num_workers"], pin_memory=torch.cuda.is_available())


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Return raw logits and labels for a loader."""
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).cpu())
        labels.append(y)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit a single temperature on validation logits by minimising NLL (Guo et al., 2017)."""
    t_logits = torch.tensor(logits, dtype=torch.float32)
    t_labels = torch.tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = nn.functional.cross_entropy(t_logits / log_t.exp(), t_labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_one(cfg: dict[str, Any], seed: int, device: torch.device) -> dict[str, Any]:
    """Run one full experiment for one seed and write metrics and predictions to disk."""
    set_seed(seed)
    paths = cfg["paths"]
    class_map = load_class_map(paths["class_map"])
    class_names = [c["name"] for c in class_map]
    run_dir = Path(paths["output_dir"]) / cfg["experiment_name"] / f"seed{seed}"
    if cfg["train"]["fewshot_k"]:
        run_dir = run_dir.with_name(f"seed{seed}_k{cfg['train']['fewshot_k']}")
    run_dir.mkdir(parents=True, exist_ok=True)

    pv = list_images(Path(paths["plantvillage_dir"]), class_map, "plantvillage")
    splits = split_plantvillage(pv, cfg["data"]["val_fraction"], cfg["data"]["test_fraction"], seed,
                                Path(paths["duplicate_groups"]))
    pd_test = list_images(Path(paths["plantdoc_test_dir"]), class_map, "plantdoc")
    train_df = splits["train"]
    k = int(cfg["train"]["fewshot_k"])
    if k > 0:
        pd_train = list_images(Path(paths["plantdoc_train_dir"]), class_map, "plantdoc")
        shots = sample_fewshot(pd_train, k, seed)
        train_df = pd.concat([train_df, shots], ignore_index=True)
        logger.info("Added %d PlantDoc few-shot images (k=%d)", len(shots), k)
    for name, df in {"train": train_df, **{s: splits[s] for s in ("val", "test")}, "plantdoc_test": pd_test}.items():
        df.to_csv(run_dir / f"split_{name}.csv", index=False)

    model = build_model(cfg["model"]["name"], len(class_names), cfg["model"]["pretrained"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["train"]["epochs"])
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg["train"]["label_smoothing"])
    train_loader, val_loader = _loader(train_df, cfg, True), _loader(splits["val"], cfg, False)

    best_f1, patience, history = -1.0, 0, []
    ckpt = run_dir / "best.pt"
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        t0, running = time.time(), 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * len(y)
        sched.step()
        v_logits, v_labels = predict_logits(model, val_loader, device)
        val_f1 = f1_score(v_labels, v_logits.argmax(1), average="macro", zero_division=0)
        history.append({"epoch": epoch, "train_loss": running / len(train_loader.dataset), "val_macro_f1": val_f1})
        logger.info("epoch %d  loss %.4f  val macro-F1 %.4f  (%.0fs)", epoch, history[-1]["train_loss"], val_f1,
                    time.time() - t0)
        if val_f1 > best_f1:
            best_f1, patience = val_f1, 0
            torch.save(model.state_dict(), ckpt)
        else:
            patience += 1
            if patience >= cfg["train"]["early_stopping_patience"]:
                logger.info("Early stopping at epoch %d", epoch)
                break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    v_logits, v_labels = predict_logits(model, val_loader, device)
    temperature = fit_temperature(v_logits, v_labels)
    logger.info("Fitted temperature on PlantVillage val: %.3f", temperature)

    ev = cfg["evaluation"]
    results: dict[str, Any] = {"experiment": cfg["experiment_name"], "seed": seed, "fewshot_k": k,
                               "temperature": temperature, "history": history}
    for domain, df in {"plantvillage_test": splits["test"], "plantdoc_test": pd_test}.items():
        logits, labels = predict_logits(model, _loader(df, cfg, False), device)
        np.savez_compressed(run_dir / f"preds_{domain}.npz", logits=logits, labels=labels,
                            paths=np.array(df["path"].tolist()))
        results[domain] = classification_report(softmax(logits), labels, class_names, ev["ece_bins"],
                                                ev["bootstrap_samples"])
        results[domain]["ece_after_temperature"] = classification_report(
            softmax(logits, temperature), labels, class_names, ev["ece_bins"], 1)["ece"]
    results["generalisation_gap_accuracy"] = (results["plantvillage_test"]["accuracy"]
                                              - results["plantdoc_test"]["accuracy"])

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    _log_mlflow(cfg, results, run_dir)
    logger.info("PV acc %.3f | PlantDoc acc %.3f | gap %.3f",
                results["plantvillage_test"]["accuracy"], results["plantdoc_test"]["accuracy"],
                results["generalisation_gap_accuracy"])
    return results


def _log_mlflow(cfg: dict[str, Any], results: dict[str, Any], run_dir: Path) -> None:
    if not cfg.get("mlflow", {}).get("enabled"):
        return
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow.enabled is true but mlflow is not installed; skipping")
        return
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    with mlflow.start_run(run_name=run_dir.parent.name + "/" + run_dir.name):
        mlflow.log_params({"model": cfg["model"]["name"], "augmentation": cfg["train"]["augmentation"],
                           "seed": results["seed"], "fewshot_k": results["fewshot_k"]})
        for dom in ("plantvillage_test", "plantdoc_test"):
            for m in ("accuracy", "macro_f1", "ece", "nll"):
                mlflow.log_metric(f"{dom}.{m}", results[dom][m])
        mlflow.log_artifacts(str(run_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--override", default=None)
    parser.add_argument("--seed", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--fewshot-k", type=int, default=None, help="Overrides train.fewshot_k")
    parser.add_argument("--epochs", type=int, default=None, help="Overrides train.epochs (for smoke tests)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.override)
    if args.fewshot_k is not None:
        cfg["train"]["fewshot_k"] = args.fewshot_k
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    for seed in args.seed:
        train_one(cfg, seed, device)


if __name__ == "__main__":
    main()
