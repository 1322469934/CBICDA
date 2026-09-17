"""Numeric feature tables and labeled circRNA–disease pairs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .model import FACTOR_NAMES


def _features(path: Path, id_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if id_column not in frame or frame[id_column].isna().any():
        raise ValueError(f"{path}: missing or null {id_column}")
    if frame[id_column].duplicated().any():
        raise ValueError(f"{path}: duplicate {id_column}")
    values = frame.drop(columns=[id_column])
    if values.shape[1] == 0:
        raise ValueError(f"{path}: no feature columns")
    values = values.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy(dtype=np.float32)).all():
        raise ValueError(f"{path}: non-finite features")
    values.index = frame[id_column].astype(str)
    return values.astype(np.float32)


class FeatureStore:
    def __init__(self, directory: str | Path):
        directory = Path(directory)
        self.factors = {name: _features(directory / f"{name}.csv", "circRNA_id")
                        for name in FACTOR_NAMES}
        self.disease = _features(directory / "disease.csv", "disease_id")
        self.factor_dims = {name: table.shape[1] for name, table in self.factors.items()}
        self.disease_dim = self.disease.shape[1]

    def check_pairs(self, pairs: pd.DataFrame) -> None:
        for name, table in self.factors.items():
            absent = set(pairs.circRNA_id.astype(str)) - set(table.index)
            if absent:
                raise ValueError(f"{name}.csv missing {len(absent)} circRNA IDs")
        absent = set(pairs.disease_id.astype(str)) - set(self.disease.index)
        if absent:
            raise ValueError(f"disease.csv missing {len(absent)} disease IDs")


def read_pairs(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"circRNA_id": str, "disease_id": str})
    expected = {"circRNA_id", "disease_id", "label"}
    if not expected.issubset(frame.columns):
        raise ValueError(f"{path}: requires {sorted(expected)}")
    frame = frame.loc[:, ["circRNA_id", "disease_id", "label"]].copy()
    if frame.isna().any().any() or not frame.label.isin([0, 1]).all():
        raise ValueError(f"{path}: IDs must be present and labels must be 0 or 1")
    if frame.duplicated(["circRNA_id", "disease_id"]).any():
        raise ValueError(f"{path}: duplicate pair")
    return frame


def validate_splits(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame,
                    setting: str) -> None:
    splits = (train, valid, test)
    pair_sets = [set(zip(f.circRNA_id, f.disease_id)) for f in splits]
    if any(pair_sets[i] & pair_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Pair overlap across splits")
    if setting not in {"warm", "cold-circrna", "cold-disease", "cold-both"}:
        raise ValueError(f"Unknown setting: {setting}")
    for column, active in (("circRNA_id", setting in ("cold-circrna", "cold-both")),
                           ("disease_id", setting in ("cold-disease", "cold-both"))):
        if active:
            groups = [set(f[column]) for f in splits]
            if any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
                raise ValueError(f"{column} entity leakage across splits")


class PairDataset(Dataset):
    def __init__(self, pairs: pd.DataFrame, store: FeatureStore):
        store.check_pairs(pairs)
        self.x = {name: torch.tensor(table.loc[pairs.circRNA_id].to_numpy(), dtype=torch.float32)
                  for name, table in store.factors.items()}
        self.disease = torch.tensor(store.disease.loc[pairs.disease_id].to_numpy(),
                                    dtype=torch.float32)
        self.labels = torch.tensor(pairs.label.to_numpy(), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return ({name: value[index] for name, value in self.x.items()},
                self.disease[index], self.labels[index])
