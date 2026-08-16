"""Train/test splitting utilities."""
from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split a source dataset once into train and held-out test sets."""
    train, test = train_test_split(df, test_size=test_size, random_state=random_state, shuffle=True)
    return train.reset_index(drop=True), test.reset_index(drop=True)
