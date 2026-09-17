"""Score pairs with a saved run, including pairs from an independent dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader

from .data import FeatureStore, PairDataset, read_pairs
from .model import CounterfactualCDA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    store = FeatureStore(args.features)
    if store.factor_dims != checkpoint["factor_dims"] or store.disease_dim != checkpoint["disease_dim"]:
        raise ValueError("Feature dimensions do not match the checkpoint")
    raw = pd.read_csv(args.pairs, dtype={"circRNA_id": str, "disease_id": str})
    has_labels = "label" in raw.columns
    pairs = read_pairs(args.pairs) if has_labels else read_pairs_frame(raw, args.pairs)
    model = CounterfactualCDA(checkpoint["factor_dims"], checkpoint["disease_dim"],
                              checkpoint["hidden_dim"])
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device)
    model.to(device).eval()
    probabilities, factor_weights = [], []
    loader = DataLoader(PairDataset(pairs, store), batch_size=args.batch_size)
    with torch.no_grad():
        for factors, disease, _ in loader:
            output = model({name: x.to(device) for name, x in factors.items()}, disease.to(device))
            probabilities.extend(torch.sigmoid(output.final_logits).cpu().numpy().tolist())
            factor_weights.extend(output.factor_weights.cpu().numpy().tolist())
    result = raw.copy()
    result["score"] = probabilities
    for index, name in enumerate(("sequence", "structure", "regulation", "expression")):
        result[f"weight_{name}"] = np.asarray(factor_weights)[:, index]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    if has_labels:
        y = pairs.label.to_numpy()
        if len(np.unique(y)) == 2:
            p = np.asarray(probabilities)
            pred = p >= .5
            metrics = {"auc": float(roc_auc_score(y, p)),
                       "aupr": float(average_precision_score(y, p)),
                       "accuracy": float(accuracy_score(y, pred)),
                       "precision": float(precision_score(y, pred, zero_division=0)),
                       "recall": float(recall_score(y, pred, zero_division=0)),
                       "f1": float(f1_score(y, pred, zero_division=0))}
            metric_path = args.output.with_suffix(".metrics.json")
            metric_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            print(json.dumps(metrics, indent=2))
    print(f"Wrote {len(result)} scores to {args.output}")


def read_pairs_frame(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not {"circRNA_id", "disease_id"}.issubset(frame.columns):
        raise ValueError(f"{path}: requires circRNA_id,disease_id")
    copy = frame.loc[:, ["circRNA_id", "disease_id"]].copy()
    copy["label"] = 0
    if copy.isna().any().any() or copy.duplicated(["circRNA_id", "disease_id"]).any():
        raise ValueError(f"{path}: missing ID or duplicate pair")
    return copy


if __name__ == "__main__":
    main()
