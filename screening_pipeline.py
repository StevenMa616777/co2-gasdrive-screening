import json
import math
import os
import re

import joblib
import numpy as np
import pandas as pd
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier = None
    LGBMRegressor = None
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = None
    XGBRegressor = None


DATA_PATH = "/Users/cosmoshan/Documents/思达威实习材料/实验/注碳数据库/注碳数据库2024.xlsx"
OUTPUT_DIR = os.path.join(os.getcwd(), "outputs_reservoir_screening")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
RANDOM_STATE = 42


TARGET_COL = "换油率\n（t）"
TOTAL_OIL_COL = "合计\n增油量\n（t）"
WELL_ID_COL = "井号"
PERFORATION_COL = "注气/射孔厚度\n（m）"
LAYER_COL = "层位"


RAW_PROPERTY_CONFIG = {
    "油层中深\n（m）": {"alias": "reservoir_depth_m", "low": 0, "high": 8000},
    "有效厚度\n（m）": {"alias": "effective_thickness_m", "low": 0, "high": 300},
    "地层温度\n（℃）": {"alias": "formation_temperature_c", "low": 0, "high": 200},
    "地层压力\n（MPa）": {"alias": "formation_pressure_mpa", "low": 0, "high": 80},
    "孔隙度\n（%）": {"alias": "porosity_pct", "low": 0, "high": 50},
    "渗透率\n（mD）": {"alias": "permeability_md", "low": 0, "high": 10000},
    "含油饱和度\n（%）": {"alias": "oil_saturation_pct", "low": 0, "high": 100},
    "地面原油密度\n（g/cm3）": {"alias": "oil_density_g_cm3", "low": 0.6, "high": 1.2},
    "50℃\n原油粘度\n(mPa.s)": {"alias": "surface_viscosity_mpas", "low": 0, "high": 100000},
    "地下\n原油粘度\n（mPa.s）": {"alias": "subsurface_viscosity_mpas", "low": 0, "high": 100000},
}


CATEGORICAL_FEATURES = ["reservoir_type_1", "reservoir_type_2", "layer_primary"]
NUMERIC_FEATURES = [
    "perforated_total_thickness_m",
    "perforated_gross_span_m",
    "perforation_segment_count",
    "perforation_continuity_ratio",
    "layer_token_count",
    "reservoir_depth_m",
    "effective_thickness_m",
    "effective_to_perforated_ratio",
    "effective_to_gross_ratio",
    "formation_temperature_c",
    "formation_pressure_mpa",
    "pressure_gradient_mpa_per_100m",
    "porosity_pct",
    "porosity_range_ratio",
    "permeability_log_md",
    "permeability_range_ratio",
    "oil_saturation_pct",
    "oil_saturation_range_ratio",
    "oil_density_g_cm3",
    "viscosity_log_mpas",
    "heavy_oil_flag",
    "mobility_log_index",
    "flow_capacity_log_kh",
    "storage_capacity_index",
    "reservoir_quality_index",
    "depth_to_thickness_ratio",
    "temperature_viscosity_coupling",
]
BUCKET_LABELS = ["无效响应", "低增效", "高价值响应"]


def ensure_dirs() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)


def normalize_text(value) -> str:
    if pd.isna(value):
        return np.nan
    text = str(value).replace("\n", " ").strip()
    if not text or text.lower() == "nan" or text == "/":
        return np.nan
    return text


def extract_numbers(value) -> list[float]:
    text = normalize_text(value)
    if text is np.nan:
        return []
    normalized = (
        text.replace("～", "-")
        .replace("~", "-")
        .replace("至", "-")
        .replace("—", "-")
        .replace("–", "-")
        .replace("／", "/")
    )
    return [float(match) for match in re.findall(r"\d+(?:\.\d+)?", normalized)]


def parse_layer(value) -> tuple[str, float]:
    text = normalize_text(value)
    if text is np.nan:
        return np.nan, np.nan
    cleaned = re.sub(r"\s+", "", text)
    tokens = [token for token in re.split(r"[+/、,，]", cleaned) if token]
    if not tokens:
        return np.nan, np.nan
    return tokens[0], float(len(tokens))


def summarize_value(value, low=None, high=None) -> dict[str, float]:
    raw_numbers = extract_numbers(value)
    if not raw_numbers:
        return {
            "center": np.nan,
            "min": np.nan,
            "max": np.nan,
            "span": np.nan,
            "range_ratio": np.nan,
            "count": 0.0,
            "invalid_count": 0.0,
        }
    series = np.asarray(raw_numbers, dtype=float)
    valid = series.copy()
    if low is not None:
        valid = valid[valid >= low]
    if high is not None:
        valid = valid[valid <= high]
    invalid_count = float(series.size - valid.size)
    if valid.size == 0:
        return {
            "center": np.nan,
            "min": np.nan,
            "max": np.nan,
            "span": np.nan,
            "range_ratio": np.nan,
            "count": float(series.size),
            "invalid_count": invalid_count,
        }
    center = float(np.median(valid))
    span = float(valid.max() - valid.min())
    return {
        "center": center,
        "min": float(valid.min()),
        "max": float(valid.max()),
        "span": span,
        "range_ratio": float(span / center) if center > 0 else np.nan,
        "count": float(valid.size),
        "invalid_count": invalid_count,
    }


def summarize_perforation_interval(value) -> dict[str, float]:
    numbers = extract_numbers(value)
    if len(numbers) < 2:
        return {
            "perforated_total_thickness_m": np.nan,
            "perforated_gross_span_m": np.nan,
            "perforation_segment_count": np.nan,
            "perforation_continuity_ratio": np.nan,
            "perforation_invalid_count": 0.0 if len(numbers) == 0 else 1.0,
        }

    odd_tail = 1 if len(numbers) % 2 else 0
    usable = numbers[:-1] if odd_tail else numbers
    pairs = []
    invalid = float(odd_tail)
    for idx in range(0, len(usable), 2):
        top = min(usable[idx], usable[idx + 1])
        bottom = max(usable[idx], usable[idx + 1])
        thickness = bottom - top
        if thickness <= 0:
            invalid += 1.0
            continue
        pairs.append((top, bottom, thickness))

    if not pairs:
        return {
            "perforated_total_thickness_m": np.nan,
            "perforated_gross_span_m": np.nan,
            "perforation_segment_count": np.nan,
            "perforation_continuity_ratio": np.nan,
            "perforation_invalid_count": invalid,
        }

    total_thickness = sum(item[2] for item in pairs)
    gross_span = max(item[1] for item in pairs) - min(item[0] for item in pairs)

    if total_thickness > 500:
        invalid += 1.0
        total_thickness = np.nan
    if gross_span > 1200:
        invalid += 1.0
        gross_span = np.nan

    continuity = np.nan
    if pd.notna(total_thickness) and pd.notna(gross_span) and gross_span > 0:
        continuity = total_thickness / gross_span

    return {
        "perforated_total_thickness_m": float(total_thickness) if pd.notna(total_thickness) else np.nan,
        "perforated_gross_span_m": float(gross_span) if pd.notna(gross_span) else np.nan,
        "perforation_segment_count": float(len(pairs)),
        "perforation_continuity_ratio": float(continuity) if pd.notna(continuity) else np.nan,
        "perforation_invalid_count": invalid,
    }


def safe_ratio(numerator, denominator):
    if pd.isna(numerator) or pd.isna(denominator) or denominator <= 0:
        return np.nan
    return numerator / denominator


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    features = pd.DataFrame(index=df.index)

    features["reservoir_type_1"] = df["油藏类型1"].map(normalize_text).astype("object")
    features["reservoir_type_2"] = df["油藏类型2"].map(normalize_text).astype("object")
    layer_info = df[LAYER_COL].apply(parse_layer)
    features["layer_primary"] = layer_info.apply(lambda item: item[0]).astype("object")
    features["layer_token_count"] = layer_info.apply(lambda item: item[1])

    interval_summary = df[PERFORATION_COL].apply(summarize_perforation_interval).apply(pd.Series)
    features = pd.concat([features, interval_summary], axis=1)

    invalid_flag_columns = ["perforation_invalid_count"]
    for raw_col, config in RAW_PROPERTY_CONFIG.items():
        summary = df[raw_col].apply(lambda value: summarize_value(value, config["low"], config["high"])).apply(pd.Series)
        alias = config["alias"]
        features[alias] = summary["center"].astype(float)
        features[f"{alias}_range_ratio"] = summary["range_ratio"].astype(float)
        features[f"{alias}_invalid_count"] = summary["invalid_count"].astype(float)
        invalid_flag_columns.append(f"{alias}_invalid_count")

    best_viscosity = features["subsurface_viscosity_mpas"].combine_first(features["surface_viscosity_mpas"])
    features["viscosity_log_mpas"] = np.log1p(best_viscosity)
    features["permeability_log_md"] = np.log1p(features["permeability_md"])
    features["porosity_range_ratio"] = features["porosity_pct_range_ratio"]
    features["permeability_range_ratio"] = features["permeability_md_range_ratio"]
    features["oil_saturation_range_ratio"] = features["oil_saturation_pct_range_ratio"]
    features["heavy_oil_flag"] = np.where(best_viscosity >= 100, 1.0, np.where(best_viscosity.notna(), 0.0, np.nan))

    features["effective_to_perforated_ratio"] = features.apply(
        lambda row: safe_ratio(row["effective_thickness_m"], row["perforated_total_thickness_m"]),
        axis=1,
    )
    features["effective_to_gross_ratio"] = features.apply(
        lambda row: safe_ratio(row["effective_thickness_m"], row["perforated_gross_span_m"]),
        axis=1,
    )
    features["pressure_gradient_mpa_per_100m"] = features.apply(
        lambda row: safe_ratio(row["formation_pressure_mpa"], row["reservoir_depth_m"]) * 100.0,
        axis=1,
    )

    mobility = features["permeability_md"] / best_viscosity
    features["mobility_log_index"] = np.log1p(mobility.where(mobility > 0))
    flow_capacity = features["permeability_md"] * features["effective_thickness_m"]
    features["flow_capacity_log_kh"] = np.log1p(flow_capacity.where(flow_capacity > 0))

    porosity_frac = features["porosity_pct"] / 100.0
    saturation_frac = features["oil_saturation_pct"] / 100.0
    features["storage_capacity_index"] = features["effective_thickness_m"] * porosity_frac * saturation_frac
    features["reservoir_quality_index"] = np.where(
        (features["permeability_md"] > 0) & (features["porosity_pct"] > 0),
        0.0314 * np.sqrt(features["permeability_md"] / features["porosity_pct"]),
        np.nan,
    )
    features["depth_to_thickness_ratio"] = features.apply(
        lambda row: safe_ratio(row["reservoir_depth_m"], row["effective_thickness_m"]),
        axis=1,
    )
    features["temperature_viscosity_coupling"] = features.apply(
        lambda row: safe_ratio(row["formation_temperature_c"], row["viscosity_log_mpas"]),
        axis=1,
    )

    ratio_invalid = (
        (features["effective_to_perforated_ratio"] > 1.2)
        | (features["effective_to_perforated_ratio"] <= 0)
        | (features["effective_to_gross_ratio"] > 1.2)
        | (features["effective_to_gross_ratio"] <= 0)
    )
    features.loc[ratio_invalid, ["effective_to_perforated_ratio", "effective_to_gross_ratio"]] = np.nan
    features["ratio_invalid_count"] = ratio_invalid.fillna(False).astype(float)
    invalid_flag_columns.append("ratio_invalid_count")

    geometry_availability = features[
        ["perforated_total_thickness_m", "reservoir_depth_m", "effective_thickness_m"]
    ].notna().sum(axis=1)
    rock_availability = features[["porosity_pct", "permeability_md", "oil_saturation_pct"]].notna().sum(axis=1)
    fluid_availability = features[["oil_density_g_cm3", "viscosity_log_mpas"]].notna().sum(axis=1)
    state_availability = features[["formation_temperature_c", "formation_pressure_mpa"]].notna().sum(axis=1)

    features["core_availability_count"] = (
        geometry_availability + rock_availability + fluid_availability + state_availability
    ).astype(float)
    features["core_availability_ratio"] = features["core_availability_count"] / 10.0
    features["quality_invalid_count"] = features[invalid_flag_columns].fillna(0).sum(axis=1)
    features["training_quality_flag"] = (
        (geometry_availability >= 2)
        & (rock_availability >= 2)
        & (features["core_availability_count"] >= 5)
        & (features["quality_invalid_count"] <= 1)
    ).astype(int)
    features["screening_confidence"] = (
        0.7 * features["core_availability_ratio"].clip(0, 1)
        + 0.3 * (1 - (features["quality_invalid_count"] / 3.0).clip(0, 1))
    ).clip(0, 1)

    categorical_cols = [col for col in CATEGORICAL_FEATURES if features[col].notna().any()]
    numeric_cols = [col for col in NUMERIC_FEATURES if col in features.columns]

    cleaning_report = {
        "rows_with_no_core_info": int((features["core_availability_count"] == 0).sum()),
        "rows_failing_training_quality": int((features["training_quality_flag"] == 0).sum()),
        "rows_with_interval_parse_issue": int((features["perforation_invalid_count"] > 0).sum()),
        "rows_with_invalid_porosity": int((features["porosity_pct_invalid_count"] > 0).sum()),
        "rows_with_invalid_saturation": int((features["oil_saturation_pct_invalid_count"] > 0).sum()),
        "rows_with_invalid_density": int((features["oil_density_g_cm3_invalid_count"] > 0).sum()),
        "rows_with_invalid_viscosity": int(
            (
                features["surface_viscosity_mpas_invalid_count"].fillna(0)
                + features["subsurface_viscosity_mpas_invalid_count"].fillna(0)
            ).gt(0).sum()
        ),
        "selected_categorical_features": categorical_cols,
        "selected_numeric_features": numeric_cols,
    }
    return features, cleaning_report


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    target = pd.DataFrame(index=df.index)
    target["exchange_rate"] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    target["total_incremental_oil"] = pd.to_numeric(df[TOTAL_OIL_COL], errors="coerce")
    target["success_label"] = np.where(target["exchange_rate"].notna(), (target["exchange_rate"] > 0).astype(int), np.nan)
    return target


def make_groups(df: pd.DataFrame) -> pd.Series:
    well_ids = df[WELL_ID_COL].map(normalize_text)
    fallback = pd.Series([f"ROW_{idx}" for idx in df.index], index=df.index)
    return well_ids.fillna(fallback)


def make_preprocessor(categorical_cols: list[str], numeric_cols: list[str]) -> ColumnTransformer:
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="缺失")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
        ]
    )
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )


def build_classifier_estimators(y: pd.Series) -> dict:
    class_count = int(pd.Series(y).nunique())
    positives = max(int(y.sum()), 1)
    negatives = max(int(len(y) - positives), 1)
    scale_pos_weight = negatives / positives

    if class_count <= 2:
        estimators = {
            "logistic_regression": LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "random_forest_classifier": RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=4,
                class_weight="balanced_subsample",
                n_jobs=1,
                random_state=RANDOM_STATE,
            ),
        }
    else:
        estimators = {
            "logistic_regression": LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "random_forest_classifier": RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=4,
                class_weight="balanced_subsample",
                n_jobs=1,
                random_state=RANDOM_STATE,
            ),
        }

    if XGBClassifier is not None:
        if class_count <= 2:
            estimators["xgboost_classifier"] = XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=4,
                min_child_weight=3,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                scale_pos_weight=scale_pos_weight,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
        else:
            estimators["xgboost_classifier"] = XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=4,
                min_child_weight=3,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                objective="multi:softprob",
                num_class=class_count,
                eval_metric="mlogloss",
                random_state=RANDOM_STATE,
                n_jobs=1,
            )

    if LGBMClassifier is not None:
        if class_count <= 2:
            estimators["lightgbm_classifier"] = LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                objective="binary",
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbosity=-1,
            )
        else:
            estimators["lightgbm_classifier"] = LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                objective="multiclass",
                num_class=class_count,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbosity=-1,
            )

    return estimators


def build_regressor_estimators() -> dict:
    estimators = {
        "ridge_regression": Ridge(alpha=1.0),
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=3,
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
    }

    if XGBRegressor is not None:
        estimators["xgboost_regressor"] = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )

    if LGBMRegressor is not None:
        estimators["lightgbm_regressor"] = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            objective="regression",
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=-1,
        )

    return estimators


def evaluate_classifier_candidates(X, y, groups, categorical_cols, numeric_cols) -> tuple[str, dict]:
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    preprocessor = make_preprocessor(categorical_cols, numeric_cols)
    candidates = build_classifier_estimators(y)
    class_count = int(pd.Series(y).nunique())

    results = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", estimator),
            ]
        )

        fold_metrics = []
        for train_idx, val_idx in cv.split(X, y, groups):
            X_fold_train = X.iloc[train_idx]
            X_fold_val = X.iloc[val_idx]
            y_fold_train = y.iloc[train_idx]
            y_fold_val = y.iloc[val_idx]
            sample_weight = compute_sample_weight(class_weight="balanced", y=y_fold_train)
            pipeline.fit(X_fold_train, y_fold_train, model__sample_weight=sample_weight)

            if class_count <= 2:
                y_prob = pipeline.predict_proba(X_fold_val)[:, 1]
                fold_metrics.append(
                    {
                        "roc_auc": roc_auc_score(y_fold_val, y_prob),
                        "average_precision": average_precision_score(y_fold_val, y_prob),
                    }
                )
            else:
                y_pred = pipeline.predict(X_fold_val)
                fold_metrics.append(
                    {
                        "f1_macro": f1_score(y_fold_val, y_pred, average="macro"),
                        "f1_weighted": f1_score(y_fold_val, y_pred, average="weighted"),
                        "balanced_accuracy": balanced_accuracy_score(y_fold_val, y_pred),
                    }
                )

        if class_count <= 2:
            results[name] = {
                "cv_roc_auc_mean": float(np.mean([item["roc_auc"] for item in fold_metrics])),
                "cv_average_precision_mean": float(np.mean([item["average_precision"] for item in fold_metrics])),
            }
        else:
            results[name] = {
                "cv_f1_macro_mean": float(np.mean([item["f1_macro"] for item in fold_metrics])),
                "cv_f1_weighted_mean": float(np.mean([item["f1_weighted"] for item in fold_metrics])),
                "cv_balanced_accuracy_mean": float(np.mean([item["balanced_accuracy"] for item in fold_metrics])),
            }

    selector = "cv_average_precision_mean" if class_count <= 2 else "cv_f1_macro_mean"
    best_name = max(results, key=lambda key: results[key][selector])
    return best_name, results


def evaluate_regressor_candidates(X, y, groups, categorical_cols, numeric_cols) -> tuple[str, dict]:
    cv = GroupKFold(n_splits=5)
    preprocessor = make_preprocessor(categorical_cols, numeric_cols)
    candidates = build_regressor_estimators()

    results = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", estimator),
            ]
        )
        scores = cross_validate(
            pipeline,
            X,
            y,
            groups=groups,
            cv=cv,
            scoring={"neg_rmse": "neg_root_mean_squared_error", "r2": "r2"},
            n_jobs=1,
        )
        results[name] = {
            "cv_rmse_mean": float(-np.mean(scores["test_neg_rmse"])),
            "cv_r2_mean": float(np.mean(scores["test_r2"])),
        }

    best_name = min(results, key=lambda key: results[key]["cv_rmse_mean"])
    return best_name, results


def build_classifier(name: str, categorical_cols: list[str], numeric_cols: list[str], y: pd.Series) -> Pipeline:
    estimator_map = build_classifier_estimators(y)
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(categorical_cols, numeric_cols)),
            ("model", estimator_map[name]),
        ]
    )


def build_regressor(name: str, categorical_cols: list[str], numeric_cols: list[str]) -> Pipeline:
    estimator_map = build_regressor_estimators()
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(categorical_cols, numeric_cols)),
            ("model", estimator_map[name]),
        ]
    )


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if thresholds.size == 0:
        return 0.5
    f1_values = []
    for idx, threshold in enumerate(thresholds):
        denominator = precision[idx] + recall[idx]
        f1_values.append(0.0 if denominator == 0 else 2 * precision[idx] * recall[idx] / denominator)
    return float(thresholds[int(np.argmax(f1_values))])


def build_bucket_thresholds(exchange_rate: pd.Series) -> tuple[float, float]:
    positive = pd.to_numeric(exchange_rate, errors="coerce")
    positive = positive[positive > 0]
    split_cut = float(positive.quantile(0.5))
    reference_high_cut = float(positive.quantile(0.8))
    return split_cut, reference_high_cut


def assign_bucket_labels(exchange_rate: pd.Series, split_cut: float, reference_high_cut: float | None = None) -> pd.Series:
    target = pd.to_numeric(exchange_rate, errors="coerce")
    bucket = pd.Series(np.nan, index=target.index, dtype="float")
    bucket.loc[target.notna() & (target <= 0)] = 0
    bucket.loc[target.notna() & (target > 0) & (target <= split_cut)] = 1
    bucket.loc[target.notna() & (target > split_cut)] = 2
    return bucket


def safe_rmse(y_true, y_pred) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def compute_classifier_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "precision_at_threshold": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_threshold": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_at_threshold": float(f1_score(y_true, y_pred, zero_division=0)),
        "decision_threshold": float(threshold),
        "positive_rate": float(np.mean(y_true)),
    }


def compute_multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int], label_names: list[str] | None = None) -> dict:
    confusion = confusion_matrix(y_true, y_pred, labels=labels)
    recall_by_class = {}
    for idx, label in enumerate(labels):
        row_sum = confusion[idx].sum()
        key = label_names[idx] if label_names is not None else str(label)
        recall_by_class[key] = 0.0 if row_sum == 0 else float(confusion[idx, idx] / row_sum)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "confusion_matrix": confusion.tolist(),
        "labels": labels,
        "label_names": label_names,
        "recall_by_class": recall_by_class,
    }


def choose_balanced_threshold_binary(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    thresholds = np.unique(np.clip(y_prob, 0, 1))
    thresholds = np.concatenate(([0.0], thresholds, [1.0]))
    best_threshold = 0.5
    best_score = -1.0
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        score = balanced_accuracy_score(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def get_oof_probability_matrix(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
) -> np.ndarray:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    num_classes = int(pd.Series(y).nunique())
    oof_prob = np.zeros((len(X), num_classes), dtype=float)
    for train_idx, val_idx in splitter.split(X, y, groups):
        fold_model = clone(pipeline)
        y_train_fold = y.iloc[train_idx]
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train_fold)
        fold_model.fit(X.iloc[train_idx], y_train_fold, model__sample_weight=sample_weight)
        oof_prob[val_idx] = fold_model.predict_proba(X.iloc[val_idx])
    return oof_prob


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": safe_rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def make_importance_frame(importances, columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "feature": columns,
            "importance_mean": importances.importances_mean,
            "importance_std": importances.importances_std,
        }
    )
    return frame.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def assign_potential_grade(score: pd.Series, low_cut: float, high_cut: float) -> pd.Series:
    return pd.Series(
        np.select([score >= high_cut, score >= low_cut], ["高潜力", "中潜力"], default="低潜力"),
        index=score.index,
    )


def main() -> None:
    ensure_dirs()

    raw_df = pd.read_excel(DATA_PATH, sheet_name="Sheet1")
    features, cleaning_report = build_feature_frame(raw_df)
    targets = build_targets(raw_df)
    groups = make_groups(raw_df)

    labeled_mask = targets["exchange_rate"].notna()
    quality_mask = features["training_quality_flag"] == 1
    trainable_mask = labeled_mask & quality_mask
    split_cut, reference_high_cut = build_bucket_thresholds(targets.loc[trainable_mask, "exchange_rate"])
    bucket_target = assign_bucket_labels(targets["exchange_rate"], split_cut, reference_high_cut)

    model_frame = features[CATEGORICAL_FEATURES + NUMERIC_FEATURES].copy()
    X_model = model_frame.loc[trainable_mask]
    y_bucket = bucket_target.loc[trainable_mask].astype(int)
    y_rate = targets.loc[trainable_mask, "exchange_rate"].astype(float)
    groups_model = groups.loc[trainable_mask]

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    split_iter = splitter.split(np.zeros(len(y_bucket)), y_bucket, groups_model)
    train_idx, test_idx = next(split_iter)

    X_train = X_model.iloc[train_idx]
    X_test = X_model.iloc[test_idx]
    y_train = y_bucket.iloc[train_idx]
    y_test = y_bucket.iloc[test_idx]
    group_train = groups_model.iloc[train_idx]

    classifier_name, classifier_cv = evaluate_classifier_candidates(
        X_train,
        y_train,
        group_train,
        CATEGORICAL_FEATURES,
        NUMERIC_FEATURES,
    )
    classifier = build_classifier(classifier_name, CATEGORICAL_FEATURES, NUMERIC_FEATURES, y_train)
    train_oof_prob = get_oof_probability_matrix(classifier, X_train, y_train, group_train)
    train_sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    classifier.fit(X_train, y_train, model__sample_weight=train_sample_weight)
    train_prob = classifier.predict_proba(X_train)
    test_prob = classifier.predict_proba(X_test)
    train_pred = classifier.predict(X_train)
    test_pred = classifier.predict(X_test)
    classifier_metrics = compute_multiclass_metrics(
        y_test.to_numpy(),
        test_pred,
        labels=list(range(len(BUCKET_LABELS))),
        label_names=BUCKET_LABELS,
    )
    high_value_index = BUCKET_LABELS.index("高价值响应")
    high_value_threshold = choose_balanced_threshold_binary(
        (y_train.to_numpy() == high_value_index).astype(int),
        train_oof_prob[:, high_value_index],
    )
    response_threshold = choose_balanced_threshold_binary(
        (y_train.to_numpy() > 0).astype(int),
        1.0 - train_oof_prob[:, 0],
    )

    classifier_importance = permutation_importance(
        classifier,
        X_test,
        y_test,
        n_repeats=12,
        random_state=RANDOM_STATE,
        n_jobs=1,
        scoring="f1_macro",
    )
    classifier_importance_df = make_importance_frame(classifier_importance, X_test.columns.tolist())

    classifier_importance_df.to_csv(
        os.path.join(OUTPUT_DIR, "classifier_feature_importance.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    all_prob_matrix = classifier.predict_proba(model_frame)
    all_pred_bucket = classifier.predict(model_frame)
    bucket_score_weights = np.arange(len(BUCKET_LABELS))
    all_expected_bucket_score = all_prob_matrix @ bucket_score_weights
    response_prob = 1.0 - all_prob_matrix[:, 0]
    high_value_prob = all_prob_matrix[:, 2]
    predicted_bucket_name = pd.Series([BUCKET_LABELS[int(idx)] for idx in all_pred_bucket], index=raw_df.index)

    scored_df = raw_df.copy()
    scored_df = pd.concat([scored_df, features], axis=1)
    scored_df["真实换油率"] = targets["exchange_rate"]
    scored_df["真实是否响应"] = targets["success_label"]
    scored_df["增长潜力桶_真实"] = bucket_target.map(lambda idx: BUCKET_LABELS[int(idx)] if pd.notna(idx) else np.nan)
    scored_df["响应概率"] = response_prob
    scored_df["高价值响应概率"] = high_value_prob
    scored_df["潜力评分"] = all_expected_bucket_score
    scored_df["潜力分级"] = predicted_bucket_name
    scored_df["桶概率_无效响应"] = all_prob_matrix[:, 0]
    scored_df["桶概率_低增效"] = all_prob_matrix[:, 1]
    scored_df["桶概率_高价值响应"] = all_prob_matrix[:, 2]
    scored_df["默认规则_推荐入选"] = (
        (scored_df["高价值响应概率"] >= high_value_threshold) & (scored_df["screening_confidence"] >= 0.6)
    ).astype(int)
    scored_df["默认规则_人工复核"] = (
        (
            (scored_df["高价值响应概率"] >= high_value_threshold * 0.6)
            | (scored_df["响应概率"] >= response_threshold)
            | (scored_df["潜力分级"] == "低增效")
        )
        & (scored_df["默认规则_推荐入选"] == 0)
    ).astype(int)
    scored_df["是否纳入训练"] = trainable_mask.astype(int)

    scored_df.to_excel(os.path.join(OUTPUT_DIR, "screening_scored_dataset.xlsx"), index=False)
    scored_df.to_csv(os.path.join(OUTPUT_DIR, "cleaned_feature_dataset.csv"), index=False, encoding="utf-8-sig")

    metrics = {
        "data_summary": {
            "raw_rows": int(len(raw_df)),
            "labeled_rows": int(labeled_mask.sum()),
            "trainable_rows_after_quality_filter": int(trainable_mask.sum()),
            "positive_response_rows_after_quality_filter": int((y_rate > 0).sum()),
            "positive_response_ratio_after_quality_filter": float((y_rate > 0).mean()),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
        "cleaning_report": cleaning_report,
        "bucket_definition": {
            "labels": BUCKET_LABELS,
            "positive_split_cut": split_cut,
            "reference_high_cut_old_definition": reference_high_cut,
            "description": "0=无效响应，1=正响应低增效，2=正响应高价值响应(原中增效+高增效合并)",
        },
        "default_decision_policy": {
            "high_value_probability_threshold": high_value_threshold,
            "response_probability_review_threshold": response_threshold,
            "recommended_min_confidence": 0.6,
            "review_relative_high_value_threshold": 0.6,
            "description": "平衡型默认规则: 高价值响应概率达到平衡阈值且置信度>=0.6则推荐入选; 否则若响应概率达到平衡阈值、或高价值响应概率达到阈值的60%、或预测为低增效则进入人工复核; 其余暂缓。",
        },
        "model_scope": {
            "categorical_features": CATEGORICAL_FEATURES,
            "numeric_features": NUMERIC_FEATURES,
            "excluded_from_candidates": [
                "油田",
                "采油厂",
                "年度",
                "区块",
                "井号",
                "井别",
                "井型",
                "措施类型",
                "施工始（年月日）",
                "施工止（年月日）",
                "施工天数（天）",
                "注入方式",
                "CO2注入量（t）",
                "CO2注入速度（t/h）",
                "CO2注入速度（t/d）",
                "注入压力（MPa）",
                "本井增油量（t）",
                "对应井增油量（t）",
                "合计增油量（t）",
                "换油率（t）",
                "备注",
            ],
        },
        "classifier": {
            "selected_model": classifier_name,
            "cv_results": classifier_cv,
            "holdout_metrics": classifier_metrics,
        },
        "top_classifier_features": classifier_importance_df.head(15).to_dict(orient="records"),
    }

    with open(os.path.join(OUTPUT_DIR, "model_metrics.json"), "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    with open(os.path.join(OUTPUT_DIR, "cleaning_report.json"), "w", encoding="utf-8") as file:
        json.dump(cleaning_report, file, ensure_ascii=False, indent=2)

    joblib.dump(classifier, os.path.join(MODEL_DIR, "classifier.joblib"))
    joblib.dump(None, os.path.join(MODEL_DIR, "regressor.joblib"))
    joblib.dump(
        {
            "task_type": "multiclass_bucket",
            "categorical_features": CATEGORICAL_FEATURES,
            "numeric_features": NUMERIC_FEATURES,
            "bucket_labels": BUCKET_LABELS,
            "bucket_score_weights": bucket_score_weights.tolist(),
            "positive_split_cut": split_cut,
            "reference_high_cut_old_definition": reference_high_cut,
            "default_high_value_probability_threshold": high_value_threshold,
            "default_response_probability_review_threshold": response_threshold,
            "default_recommended_min_confidence": 0.6,
            "default_review_relative_high_value_threshold": 0.6,
        },
        os.path.join(MODEL_DIR, "metadata.joblib"),
    )

    print("Pipeline finished.")
    print(f"Trainable rows: {int(trainable_mask.sum())}")
    print(f"Classifier: {classifier_name}")
    print(json.dumps(classifier_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
