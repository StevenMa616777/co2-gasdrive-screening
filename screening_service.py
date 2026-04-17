import io
import json
import os
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from screening_pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COL,
    assign_potential_grade,
    build_feature_frame,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "outputs_reservoir_screening")
MODEL_DIR = os.path.join(ARTIFACT_DIR, "models")
REFERENCE_DATA_PATH = os.path.join(ARTIFACT_DIR, "cleaned_feature_dataset.csv")
CLASSIFIER_IMPORTANCE_PATH = os.path.join(ARTIFACT_DIR, "classifier_feature_importance.csv")
MODEL_METRICS_PATH = os.path.join(ARTIFACT_DIR, "model_metrics.json")


RAW_INPUT_COLUMNS = [
    "井号",
    "油藏类型1",
    "油藏类型2",
    "层位",
    "注气/射孔厚度\n（m）",
    "油层中深\n（m）",
    "有效厚度\n（m）",
    "地层温度\n（℃）",
    "地层压力\n（MPa）",
    "孔隙度\n（%）",
    "渗透率\n（mD）",
    "含油饱和度\n（%）",
    "地面原油密度\n（g/cm3）",
    "50℃\n原油粘度\n(mPa.s)",
    "地下\n原油粘度\n（mPa.s）",
]


FEATURE_LABELS = {
    "reservoir_type_1": "油藏类型1",
    "reservoir_type_2": "油藏类型2",
    "layer_primary": "主层位",
    "perforated_total_thickness_m": "射孔总厚度",
    "perforated_gross_span_m": "射孔总跨度",
    "perforation_segment_count": "射孔分段数",
    "perforation_continuity_ratio": "射孔连续性",
    "layer_token_count": "层位组合数",
    "reservoir_depth_m": "油层中深",
    "effective_thickness_m": "有效厚度",
    "effective_to_perforated_ratio": "净厚占射孔比",
    "effective_to_gross_ratio": "净厚占总跨度比",
    "formation_temperature_c": "地层温度",
    "formation_pressure_mpa": "地层压力",
    "pressure_gradient_mpa_per_100m": "压力梯度",
    "porosity_pct": "孔隙度",
    "porosity_range_ratio": "孔隙度离散度",
    "permeability_log_md": "渗透率水平",
    "permeability_range_ratio": "渗透率离散度",
    "oil_saturation_pct": "含油饱和度",
    "oil_saturation_range_ratio": "含油饱和度离散度",
    "oil_density_g_cm3": "地面原油密度",
    "viscosity_log_mpas": "原油粘度水平",
    "heavy_oil_flag": "重油特征",
    "mobility_log_index": "流度指标",
    "flow_capacity_log_kh": "流动能力(kh)",
    "storage_capacity_index": "储集能力",
    "reservoir_quality_index": "储层质量指数",
    "depth_to_thickness_ratio": "埋深厚度比",
    "temperature_viscosity_coupling": "温度-粘度耦合",
}


NUMERIC_REASON_TEMPLATES = {
    "formation_pressure_mpa": ("偏高", "偏低"),
    "pressure_gradient_mpa_per_100m": ("偏高", "偏低"),
    "formation_temperature_c": ("偏高", "偏低"),
    "depth_to_thickness_ratio": ("更合理", "偏大"),
    "flow_capacity_log_kh": ("较强", "偏弱"),
    "porosity_pct": ("较好", "偏低"),
    "perforated_gross_span_m": ("较大", "偏小"),
    "perforated_total_thickness_m": ("较厚", "偏薄"),
    "effective_to_gross_ratio": ("较高", "偏低"),
    "effective_to_perforated_ratio": ("较高", "偏低"),
    "storage_capacity_index": ("较强", "偏弱"),
    "reservoir_quality_index": ("较高", "偏低"),
    "oil_saturation_pct": ("较高", "偏低"),
    "permeability_log_md": ("较高", "偏低"),
    "mobility_log_index": ("较好", "偏弱"),
    "viscosity_log_mpas": ("较低", "偏高"),
}


def ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for column in RAW_INPUT_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = np.nan
    return normalized


def build_template_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=RAW_INPUT_COLUMNS)


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    buffer.seek(0)
    return buffer.getvalue()


@lru_cache(maxsize=1)
def load_bundle() -> dict:
    required_paths = [
        os.path.join(MODEL_DIR, "classifier.joblib"),
        os.path.join(MODEL_DIR, "regressor.joblib"),
        os.path.join(MODEL_DIR, "metadata.joblib"),
        REFERENCE_DATA_PATH,
        CLASSIFIER_IMPORTANCE_PATH,
        MODEL_METRICS_PATH,
    ]
    missing_paths = [path for path in required_paths if not os.path.exists(path)]
    if missing_paths:
        joined = "\n".join(missing_paths)
        raise FileNotFoundError(
            "缺少部署所需模型产物，请确认仓库中包含以下文件:\n"
            f"{joined}"
        )

    classifier = joblib.load(os.path.join(MODEL_DIR, "classifier.joblib"))
    regressor = joblib.load(os.path.join(MODEL_DIR, "regressor.joblib"))
    metadata = joblib.load(os.path.join(MODEL_DIR, "metadata.joblib"))
    reference_df = pd.read_csv(REFERENCE_DATA_PATH)
    importance_df = pd.read_csv(CLASSIFIER_IMPORTANCE_PATH)
    with open(MODEL_METRICS_PATH, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    train_df = reference_df.loc[reference_df["是否纳入训练"] == 1].copy()
    overall_positive_rate = train_df["真实是否响应"].mean()

    numeric_profiles = {}
    for feature in NUMERIC_FEATURES:
        series = pd.to_numeric(train_df[feature], errors="coerce")
        positive = pd.to_numeric(train_df.loc[train_df["真实是否响应"] == 1, feature], errors="coerce")
        if series.notna().sum() < 20 or positive.notna().sum() < 10:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = max(float(q3 - q1), 1e-6)
        overall_median = float(series.median())
        positive_median = float(positive.median())
        direction = 1 if positive_median > overall_median else -1 if positive_median < overall_median else 0
        numeric_profiles[feature] = {
            "overall_median": overall_median,
            "positive_median": positive_median,
            "iqr": iqr,
            "direction": direction,
        }

    categorical_profiles = {}
    for feature in CATEGORICAL_FEATURES:
        stats = (
            train_df.loc[train_df[feature].notna()]
            .groupby(feature)
            .agg(
                count=("真实是否响应", "size"),
                success_rate=("真实是否响应", "mean"),
            )
            .reset_index()
        )
        stats = stats.loc[stats["count"] >= 8].copy()
        categorical_profiles[feature] = stats.set_index(feature).to_dict(orient="index")

    importance_map = importance_df.set_index("feature")["importance_mean"].to_dict()

    return {
        "classifier": classifier,
        "regressor": regressor,
        "metadata": metadata,
        "metrics": metrics,
        "reference_df": reference_df,
        "train_df": train_df,
        "overall_positive_rate": overall_positive_rate,
        "numeric_profiles": numeric_profiles,
        "categorical_profiles": categorical_profiles,
        "importance_map": importance_map,
    }


def make_quality_note(row: pd.Series) -> str:
    notes = []
    if row.get("training_quality_flag", 0) == 0:
        notes.append("核心油藏参数不足，建议人工复核")
    if row.get("quality_invalid_count", 0) > 0:
        notes.append("存在区间解析或异常值，系统已自动降权")
    if row.get("screening_confidence", 0) < 0.5:
        notes.append("置信度偏低，建议结合地质认识判断")
    if not notes:
        return "数据质量可用于自动初筛"
    return "；".join(notes)


def build_numeric_reason(feature: str, value: float, profile: dict, grade: str) -> tuple[float, str] | None:
    if pd.isna(value):
        return None
    direction = profile["direction"]
    if direction == 0:
        return None
    aligned_score = direction * ((value - profile["overall_median"]) / profile["iqr"])
    if grade == "高价值响应" and aligned_score <= 0.35:
        return None
    if grade in {"低增效", "无效响应"} and aligned_score >= -0.35:
        return None

    positive_text, negative_text = NUMERIC_REASON_TEMPLATES.get(feature, ("偏高", "偏低"))
    label = FEATURE_LABELS.get(feature, feature)
    if grade in {"低增效", "无效响应"}:
        text = f"{label}{negative_text}，与历史有效井特征偏离"
        score = abs(aligned_score)
    else:
        text = f"{label}{positive_text}，更接近历史有效井特征"
        score = aligned_score
    return score, text


def build_categorical_reason(feature: str, value, profiles: dict, overall_positive_rate: float, grade: str) -> tuple[float, str] | None:
    if pd.isna(value):
        return None
    feature_stats = profiles.get(feature, {})
    if value not in feature_stats:
        return None
    item = feature_stats[value]
    uplift = item["success_rate"] - overall_positive_rate
    if grade == "高价值响应" and uplift <= 0.02:
        return None
    if grade in {"低增效", "无效响应"} and uplift >= -0.02:
        return None
    label = FEATURE_LABELS.get(feature, feature)
    if grade in {"低增效", "无效响应"}:
        text = f"{label}={value} 在历史样本中的响应率偏低"
        score = abs(uplift)
    else:
        text = f"{label}={value} 在历史样本中的响应率偏高"
        score = uplift
    return score * min(item["count"] / 30.0, 1.5), text


def explain_row(row: pd.Series, bundle: dict) -> list[str]:
    grade = row["潜力分级"]
    reasons = []
    importance_features = sorted(
        bundle["importance_map"].items(),
        key=lambda item: item[1],
        reverse=True,
    )[:12]

    for feature, importance in importance_features:
        if feature in bundle["numeric_profiles"]:
            candidate = build_numeric_reason(feature, row.get(feature), bundle["numeric_profiles"][feature], grade)
            if candidate is not None:
                reasons.append((importance * candidate[0], candidate[1]))
        elif feature in bundle["categorical_profiles"]:
            candidate = build_categorical_reason(
                feature,
                row.get(feature),
                bundle["categorical_profiles"],
                bundle["overall_positive_rate"],
                grade,
            )
            if candidate is not None:
                reasons.append((importance * candidate[0], candidate[1]))

    if row.get("screening_confidence", 0) >= 0.8 and grade not in {"低增效", "无效响应"}:
        reasons.append((0.1, "核心油藏参数完整度较高，筛选结论更稳定"))
    if row.get("screening_confidence", 0) < 0.5:
        reasons.append((0.1, "核心参数缺失较多，本次结果主要用于粗筛"))

    deduped = []
    seen = set()
    for _, text in sorted(reasons, key=lambda item: item[0], reverse=True):
        if text not in seen:
            deduped.append(text)
            seen.add(text)
        if len(deduped) == 3:
            break

    while len(deduped) < 3:
        fallback = [
            "系统综合了储层、流体和地层状态参数进行排序",
            "建议结合现场工艺可达性做二次复核",
            "结果适合做初筛，不建议直接替代工程论证",
        ][len(deduped)]
        deduped.append(fallback)
    return deduped


def make_recommendation(row: pd.Series, metadata: dict) -> str:
    grade = row["潜力分级"]
    confidence = float(row.get("screening_confidence", 0))
    high_value_prob = float(row.get("高价值响应概率", 0))
    response_prob = float(row.get("响应概率", 0))
    high_value_threshold = metadata.get("default_high_value_probability_threshold", 0.5)
    response_threshold = metadata.get("default_response_probability_review_threshold", 0.5)
    min_confidence = metadata.get("default_recommended_min_confidence", 0.6)
    relative_review = metadata.get("default_review_relative_high_value_threshold", 0.6)

    if high_value_prob >= high_value_threshold and confidence >= min_confidence:
        return "推荐入选"
    if (
        response_prob >= response_threshold
        or high_value_prob >= high_value_threshold * relative_review
        or grade == "低增效"
    ):
        return "人工复核"
    return "暂缓优先"


def score_dataframe(input_df: pd.DataFrame) -> pd.DataFrame:
    bundle = load_bundle()
    source_df = ensure_required_columns(input_df)
    features, _ = build_feature_frame(source_df)

    metadata = bundle["metadata"]
    model_frame = features[metadata["categorical_features"] + metadata["numeric_features"]].copy()
    if metadata.get("task_type") == "multiclass_bucket":
        probability_matrix = bundle["classifier"].predict_proba(model_frame)
        bucket_labels = metadata["bucket_labels"]
        bucket_scores = np.asarray(metadata["bucket_score_weights"], dtype=float)
        predicted_bucket = bundle["classifier"].predict(model_frame)
        response_prob = 1.0 - probability_matrix[:, 0]
        score = probability_matrix @ bucket_scores
        grade = pd.Series([bucket_labels[int(idx)] for idx in predicted_bucket], index=source_df.index)
    else:
        response_prob = bundle["classifier"].predict_proba(model_frame)[:, 1]
        conditional_rate = np.expm1(bundle["regressor"].predict(model_frame))
        conditional_rate = np.clip(conditional_rate, a_min=0, a_max=None)
        score = response_prob * conditional_rate
        grade = assign_potential_grade(
            pd.Series(score, index=source_df.index),
            metadata["medium_threshold"],
            metadata["high_threshold"],
        )
        probability_matrix = None

    result = source_df.copy()
    result = pd.concat([result, features], axis=1)
    result["响应概率"] = response_prob
    result["潜力评分"] = score
    result["潜力分级"] = grade
    if probability_matrix is not None:
        for idx, label in enumerate(metadata["bucket_labels"]):
            result[f"桶概率_{label}"] = probability_matrix[:, idx]
        if "桶概率_高价值响应" in result.columns:
            result["高价值响应概率"] = result["桶概率_高价值响应"]
    result["筛选建议"] = result.apply(lambda row: make_recommendation(row, metadata), axis=1)
    result["数据质量提示"] = result.apply(make_quality_note, axis=1)

    reasons = result.apply(lambda row: explain_row(row, bundle), axis=1)
    result["关键原因1"] = reasons.apply(lambda item: item[0])
    result["关键原因2"] = reasons.apply(lambda item: item[1])
    result["关键原因3"] = reasons.apply(lambda item: item[2])
    return result


def score_file(input_path: str, output_path: str, sheet_name=0) -> str:
    if input_path.lower().endswith(".csv"):
        df = pd.read_csv(input_path)
    else:
        df = pd.read_excel(input_path, sheet_name=sheet_name)
    scored_df = score_dataframe(df)
    if output_path.lower().endswith(".csv"):
        scored_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        scored_df.to_excel(output_path, index=False)
    return output_path
