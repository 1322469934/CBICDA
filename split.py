"""Prepare Warm and entity-disjoint evaluation partitions from positive pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import FeatureStore, validate_splits


def _partition(ids: list[str], rng: np.random.Generator):
    ids = np.asarray(sorted(ids), dtype=object)
    rng.shuffle(ids)
    n_train = int(len(ids) * 0.8)
    n_valid = int(len(ids) * 0.1)
    return [set(x.tolist()) for x in
            (ids[:n_train], ids[n_train:n_train + n_valid], ids[n_train + n_valid:])]


def make_splits(positive: pd.DataFrame, circ_ids: list[str], disease_ids: list[str],
                setting: str, seed: int = 42, negative_ratio: float = 1.0):
    """Use 80/10/10 partitions; discard mixed-block edges in Cold-both."""
    if not 0 < negative_ratio:
        raise ValueError("negative_ratio must be positive")
    if positive.empty or positive.duplicated(["circRNA_id", "disease_id"]).any():
        raise ValueError("Positive pairs must be nonempty and unique")
    rng = np.random.default_rng(seed)
    circ_all, disease_all = set(circ_ids), set(disease_ids)
    known = set(zip(positive.circRNA_id, positive.disease_id))
    if any(c not in circ_all or d not in disease_all for c, d in known):
        raise ValueError("Positive ID absent from feature tables")
    if setting == "warm":
        shuffled = positive.sample(frac=1, random_state=seed).reset_index(drop=True)
        a, b = int(len(shuffled) * .8), int(len(shuffled) * .9)
        positive_splits = [shuffled.iloc[:a], shuffled.iloc[a:b], shuffled.iloc[b:]]
        eligible_c = [circ_all] * 3
        eligible_d = [disease_all] * 3
    elif setting in {"cold-circrna", "cold-disease", "cold-both"}:
        eligible_c = (_partition(circ_ids, rng) if setting != "cold-disease" else [circ_all] * 3)
        eligible_d = (_partition(disease_ids, rng) if setting != "cold-circrna" else [disease_all] * 3)
        positive_splits = [positive.loc[positive.circRNA_id.isin(eligible_c[i]) &
                                         positive.disease_id.isin(eligible_d[i])]
                           for i in range(3)]
    else:
        raise ValueError(f"Unknown setting: {setting}")

    sampled = set()
    results = []
    for i, pos in enumerate(positive_splits):
        if pos.empty:
            raise ValueError(f"Split {i} has no positive edges; change seed or data")
        count = round(len(pos) * negative_ratio)
        occupied = sum(c in eligible_c[i] and d in eligible_d[i]
                       for c, d in known | sampled)
        available = len(eligible_c[i]) * len(eligible_d[i]) - occupied
        if count > available:
            raise ValueError("Not enough eligible negative pairs")
        negatives = []
        c_list, d_list = sorted(eligible_c[i]), sorted(eligible_d[i])
        attempts = 0
        while len(negatives) < count:
            candidate = (str(rng.choice(c_list)), str(rng.choice(d_list)))
            attempts += 1
            if candidate in known or candidate in sampled:
                if attempts > 100 * max(count, 1):
                    raise ValueError("Negative sampling exhausted; reduce negative_ratio")
                continue
            sampled.add(candidate)
            negatives.append((*candidate, 0))
        labeled_pos = pos.loc[:, ["circRNA_id", "disease_id"]].copy()
        labeled_pos["label"] = 1
        result = pd.concat([labeled_pos, pd.DataFrame(negatives,
                           columns=["circRNA_id", "disease_id", "label"])], ignore_index=True)
        results.append(result.sample(frac=1, random_state=seed + i).reset_index(drop=True))
    validate_splits(*results, setting)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--positive", type=Path, required=True,
                        help="CSV with circRNA_id,disease_id columns")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--setting", choices=["warm", "cold-circrna", "cold-disease", "cold-both"],
                        required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    args = parser.parse_args()
    store = FeatureStore(args.features)
    frame = pd.read_csv(args.positive, dtype={"circRNA_id": str, "disease_id": str})
    if not {"circRNA_id", "disease_id"}.issubset(frame.columns):
        raise ValueError("Positive CSV requires circRNA_id,disease_id")
    if frame[["circRNA_id", "disease_id"]].isna().any().any():
        raise ValueError("Positive CSV contains null IDs")
    splits = make_splits(frame[["circRNA_id", "disease_id"]],
                         list(store.factors["sequence"].index), list(store.disease.index),
                         args.setting, args.seed, args.negative_ratio)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in zip(("train", "valid", "test"), splits):
        frame.to_csv(args.output / f"{name}.csv", index=False)
        print(f"{name}: {len(frame)} pairs, {int(frame.label.sum())} positives")


if __name__ == "__main__":
    main()
