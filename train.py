"""Train and evaluate the paper's four settings on prepared feature tables."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader

from .data import FeatureStore, PairDataset, read_pairs, validate_splits
from .model import CounterfactualCDA, objective


def _seed(value: int):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _device_batches(loader, device):
    for factors, disease, labels in loader:
        yield ({name: value.to(device) for name, value in factors.items()},
               disease.to(device), labels.to(device))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    labels, probabilities = [], []
    for factors, disease, target in _device_batches(loader, device):
        logits = model(factors, disease).final_logits
        labels.extend(target.cpu().numpy().tolist())
        probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    y, p = np.asarray(labels), np.asarray(probabilities)
    pred = p >= 0.5
    if len(np.unique(y)) < 2:
        raise ValueError("Evaluation split needs both positive and negative labels")
    return {"auc": float(roc_auc_score(y, p)),
            "aupr": float(average_precision_score(y, p)),
            "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0))}


def run(args):
    store = FeatureStore(args.features)
    train = read_pairs(args.splits / "train.csv")
    valid = read_pairs(args.splits / "valid.csv")
    test = read_pairs(args.splits / "test.csv")
    validate_splits(train, valid, test, args.setting)
    for frame in (train, valid, test):
        if len(frame.label.unique()) < 2:
            raise ValueError("Every split requires both label classes")
    device = torch.device(args.device)
    train_ds, valid_ds, test_ds = [PairDataset(f, store) for f in (train, valid, test)]
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for repeat in range(args.repeats):
        _seed(args.seed + repeat)
        model = CounterfactualCDA(store.factor_dims, store.disease_dim, args.hidden_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=args.batch_size)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size)
        best_aupr, best_state, best_epoch = -1.0, None, 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            for factors, disease, labels in _device_batches(train_loader, device):
                optimizer.zero_grad(set_to_none=True)
                output = model(factors, disease)
                loss, _ = objective(output, labels, args.lambda_inv, args.lambda_dis)
                loss.backward()
                optimizer.step()
            metrics = evaluate(model, valid_loader, device)
            if metrics["aupr"] > best_aupr:
                best_aupr, best_epoch = metrics["aupr"], epoch
                best_state = {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}
        model.load_state_dict(best_state)
        metrics = evaluate(model, test_loader, device)
        results.append(metrics)
        torch.save({"model": best_state, "factor_dims": store.factor_dims,
                    "disease_dim": store.disease_dim, "hidden_dim": args.hidden_dim,
                    "setting": args.setting, "seed": args.seed + repeat,
                    "selected_epoch": best_epoch}, args.output / f"run_{repeat + 1}.pt")
        print(f"run {repeat + 1}: selected epoch {best_epoch}, test {metrics}")
    summary = {key: {"mean": float(np.mean([r[key] for r in results])),
                     "std": float(np.std([r[key] for r in results]))}
               for key in results[0]}
    report = {"setting": args.setting, "runs": results, "summary": summary,
              "settings": {"hidden_dim": args.hidden_dim, "lambda_inv": args.lambda_inv,
                           "lambda_dis": args.lambda_dis, "batch_size": args.batch_size,
                           "learning_rate": args.lr, "epochs": args.epochs, "seed": args.seed}}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--setting", choices=["warm", "cold-circrna", "cold-disease", "cold-both"],
                        required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lambda-inv", type=float, default=0.5)
    parser.add_argument("--lambda-dis", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.repeats < 1 or args.batch_size < 1:
        parser.error("epochs, repeats, and batch-size must be positive")
    run(args)


if __name__ == "__main__":
    main()
