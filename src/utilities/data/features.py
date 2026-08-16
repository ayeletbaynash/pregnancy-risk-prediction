"""Shared feature selection, imputation, and scaling."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utilities.constants import FEATURES

def unique_preserve_order(values):
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def get_embedding_features(*dfs, feature_candidates=None):
    combined_cols = set()
    for df in dfs:
        combined_cols.update(df.columns)
    candidates = FEATURES if feature_candidates is None else unique_preserve_order(feature_candidates)
    features = [f for f in candidates if f in combined_cols]
    usable = []
    for feature in features:
        if any(feature in df.columns and df[feature].notna().any() for df in dfs):
            usable.append(feature)
    return usable


class MixedFeatureImputer:
    def __init__(self, discrete_unique_threshold=4):
        self.discrete_unique_threshold = discrete_unique_threshold
        self.fill_values_ = {}
        self.feature_types_ = {}
        self.strategies_ = {}

    def fit(self, data):
        for feature in data.columns:
            series = pd.to_numeric(data[feature], errors="coerce")
            observed = series.dropna()
            is_discrete = self._is_discrete(observed)
            strategy = "median" if is_discrete else "mean"

            if observed.empty:
                fill_value = 0.0
            elif strategy == "median":
                fill_value = float(observed.median())
            else:
                fill_value = float(observed.mean())

            self.fill_values_[feature] = fill_value
            self.feature_types_[feature] = "discrete" if is_discrete else "continuous"
            self.strategies_[feature] = strategy
        return self

    def transform(self, data):
        transformed = data.copy()
        for feature, fill_value in self.fill_values_.items():
            transformed[feature] = pd.to_numeric(transformed[feature], errors="coerce").fillna(fill_value)
        return transformed[list(self.fill_values_.keys())].to_numpy(dtype=np.float32)

    def fit_transform(self, data):
        return self.fit(data).transform(data)

    def summary(self):
        return {
            feature: {
                "feature_type": self.feature_types_[feature],
                "strategy": self.strategies_[feature],
                "fill_value": self.fill_values_[feature],
            }
            for feature in self.fill_values_
        }

    def _is_discrete(self, observed):
        if observed.empty:
            return False
        values = observed.to_numpy(dtype=float)
        integer_like = np.isclose(values, np.round(values), equal_nan=False).all()
        return bool(integer_like and observed.nunique(dropna=True) <= self.discrete_unique_threshold)


def fit_preprocessor(train_df, feature_names):
    train_data = train_df.reindex(columns=feature_names).apply(pd.to_numeric, errors="coerce")
    imputer = MixedFeatureImputer()
    scaler = StandardScaler()
    imputed = imputer.fit_transform(train_data)
    scaled = scaler.fit_transform(imputed)
    return imputer, scaler, scaled.astype(np.float32)


def transform_features(df, feature_names, imputer, scaler):
    data = df.reindex(columns=feature_names).apply(pd.to_numeric, errors="coerce")
    imputed = imputer.transform(data)
    scaled = scaler.transform(imputed)
    return scaled.astype(np.float32)
