"""
Weather Trend Forecasting Assessment

This script performs end-to-end cleaning, EDA, anomaly detection, forecasting,
feature importance, air-quality correlation, and spatial analysis for Kaggle's
Global / World Weather Repository dataset.

Expected input:
    data/Global Weather Repository.csv

Run:
    python src/weather_trend_forecasting.py
"""

from __future__ import annotations

import json
import math
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(OUTPUT_DIR / ".cache"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="Set2")

try:
    import country_converter as coco
except ImportError:
    coco = None


DATA_CANDIDATES = [
    PROJECT_ROOT / "data" / "Global Weather Repository.csv",
    PROJECT_ROOT / "data" / "GlobalWeatherRepository.csv",
]
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_PATH = PROJECT_ROOT / "report" / "Weather_Trend_Forecasting_Report.md"


@dataclass
class Columns:
    date: str
    city: str | None
    country: str | None
    temperature: str
    precipitation: str | None
    latitude: str | None
    longitude: str | None
    humidity: str | None
    wind: str | None
    pressure: str | None
    air_quality: list[str]


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def find_column(columns: Iterable[str], candidates: Iterable[str], required: bool = False) -> str | None:
    normalized = {normalize_name(col): col for col in columns}
    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key]
    if required:
        raise ValueError(f"Required column not found. Tried: {', '.join(candidates)}")
    return None


def discover_columns(df: pd.DataFrame) -> Columns:
    cols = df.columns
    date_col = find_column(cols, ["last_updated", "lastupdated", "date", "datetime"], required=True)
    temperature_col = find_column(
        cols,
        ["temperature_celsius", "temperature_c", "temp_c", "temperature", "temperature_fahrenheit"],
        required=True,
    )

    air_quality_cols = [
        col
        for col in cols
        if any(token in normalize_name(col) for token in ["air_quality", "pm2_5", "pm10", "carbon_monoxide", "ozone"])
    ]

    return Columns(
        date=date_col,
        city=find_column(cols, ["location_name", "city", "name"]),
        country=find_column(cols, ["country"]),
        temperature=temperature_col,
        precipitation=find_column(cols, ["precip_mm", "precipitation_mm", "precipitation", "rain_mm"]),
        latitude=find_column(cols, ["latitude", "lat"]),
        longitude=find_column(cols, ["longitude", "lon", "lng"]),
        humidity=find_column(cols, ["humidity", "humidity_percent"]),
        wind=find_column(cols, ["wind_kph", "wind_mph", "wind_speed", "wind_speed_kph"]),
        pressure=find_column(cols, ["pressure_mb", "pressure", "pressure_millibars"]),
        air_quality=air_quality_cols,
    )


def resolve_data_path() -> Path:
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Dataset not found. Download it from Kaggle and save it as one of:\n"
        + "\n".join(f"- {path}" for path in DATA_CANDIDATES)
    )


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download it from Kaggle and save it as data/Global Weather Repository.csv."
        )
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame, columns: Columns) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    df[columns.date] = pd.to_datetime(df[columns.date], errors="coerce")
    df = df.dropna(subset=[columns.date, columns.temperature])

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    missing_before = df.isna().sum().sort_values(ascending=False).head(15).to_dict()

    # Median imputation is robust for weather data because extreme values are common.
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("Unknown")

    outlier_summary = {}
    clipped = df.copy()
    for col in numeric_cols:
        q1 = clipped[col].quantile(0.25)
        q3 = clipped[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or math.isnan(iqr):
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (clipped[col] < lower) | (clipped[col] > upper)
        outlier_summary[col] = int(mask.sum())
        clipped[col] = clipped[col].clip(lower, upper)

    scaler = StandardScaler()
    scaled_cols = [f"{col}_scaled" for col in numeric_cols]
    clipped[scaled_cols] = scaler.fit_transform(clipped[numeric_cols])

    if columns.country and "continent" not in clipped.columns and coco is not None:
        clipped["continent"] = coco.convert(names=clipped[columns.country].tolist(), to="continent", not_found="Unknown")

    clipped = clipped.sort_values(columns.date).reset_index(drop=True)
    cleaning_summary = {
        "rows_after_cleaning": int(len(clipped)),
        "columns": int(clipped.shape[1]),
        "missing_before_top15": missing_before,
        "outliers_iqr_clipped": outlier_summary,
        "numeric_columns_scaled": scaled_cols,
    }
    return clipped, cleaning_summary


def save_basic_eda(df: pd.DataFrame, columns: Columns) -> dict:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    eda_summary = {}

    daily = df.set_index(columns.date).resample("D").agg({columns.temperature: "mean"})
    if columns.precipitation:
        daily[columns.precipitation] = df.set_index(columns.date)[columns.precipitation].resample("D").mean()

    plt.figure(figsize=(12, 5))
    daily[columns.temperature].plot()
    plt.title("Global Average Daily Temperature Trend")
    plt.xlabel("Date")
    plt.ylabel("Temperature")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "temperature_trend.png", dpi=160)
    plt.close()

    if columns.precipitation:
        plt.figure(figsize=(12, 5))
        daily[columns.precipitation].plot(color="#3b82f6")
        plt.title("Global Average Daily Precipitation Trend")
        plt.xlabel("Date")
        plt.ylabel("Precipitation")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "precipitation_trend.png", dpi=160)
        plt.close()

    corr_cols = [col for col in [columns.temperature, columns.precipitation, columns.humidity, columns.wind, columns.pressure] if col]
    corr_cols.extend(columns.air_quality[:6])
    corr_cols = [col for col in dict.fromkeys(corr_cols) if col in df.columns]
    if len(corr_cols) >= 2:
        plt.figure(figsize=(11, 8))
        sns.heatmap(df[corr_cols].corr(), cmap="coolwarm", center=0, annot=False)
        plt.title("Weather and Air Quality Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "correlation_heatmap.png", dpi=160)
        plt.close()

    if columns.country:
        top_country_temp = (
            df.groupby(columns.country)[columns.temperature]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .round(2)
            .to_dict()
        )
        eda_summary["top_10_hottest_countries"] = top_country_temp

    eda_summary["temperature_summary"] = df[columns.temperature].describe().round(2).to_dict()
    if columns.precipitation:
        eda_summary["precipitation_summary"] = df[columns.precipitation].describe().round(2).to_dict()
    return eda_summary


def add_time_features(df: pd.DataFrame, columns: Columns) -> pd.DataFrame:
    model_df = df.copy()
    model_df["year"] = model_df[columns.date].dt.year
    model_df["month"] = model_df[columns.date].dt.month
    model_df["dayofyear"] = model_df[columns.date].dt.dayofyear
    model_df["weekofyear"] = model_df[columns.date].dt.isocalendar().week.astype(int)
    model_df["sin_day"] = np.sin(2 * np.pi * model_df["dayofyear"] / 365.25)
    model_df["cos_day"] = np.cos(2 * np.pi * model_df["dayofyear"] / 365.25)

    group_col = columns.city or columns.country
    if group_col:
        model_df = model_df.sort_values([group_col, columns.date])
        model_df["temp_lag_1"] = model_df.groupby(group_col)[columns.temperature].shift(1)
        model_df["temp_lag_7"] = model_df.groupby(group_col)[columns.temperature].shift(7)
        model_df["temp_rolling_7"] = model_df.groupby(group_col)[columns.temperature].shift(1).rolling(7).mean()
    else:
        model_df = model_df.sort_values(columns.date)
        model_df["temp_lag_1"] = model_df[columns.temperature].shift(1)
        model_df["temp_lag_7"] = model_df[columns.temperature].shift(7)
        model_df["temp_rolling_7"] = model_df[columns.temperature].shift(1).rolling(7).mean()

    return model_df.dropna(subset=["temp_lag_1", "temp_lag_7", "temp_rolling_7"])


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    denominator = np.where(y_true == 0, np.nan, y_true)
    mape = np.nanmean(np.abs((y_true - y_pred) / denominator)) * 100
    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "MAPE_percent": round(float(mape), 4),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
    }


def build_forecasting_models(df: pd.DataFrame, columns: Columns) -> tuple[pd.DataFrame, dict]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_df = add_time_features(df, columns)
    feature_cols = [
        "year",
        "month",
        "dayofyear",
        "weekofyear",
        "sin_day",
        "cos_day",
        "temp_lag_1",
        "temp_lag_7",
        "temp_rolling_7",
    ]
    for optional_col in [columns.precipitation, columns.humidity, columns.wind, columns.pressure]:
        if optional_col and optional_col in model_df.columns:
            feature_cols.append(optional_col)

    model_df = model_df.sort_values(columns.date)
    split_idx = int(len(model_df) * 0.8)
    train = model_df.iloc[:split_idx]
    test = model_df.iloc[split_idx:]

    X_train = train[feature_cols]
    y_train = train[columns.temperature]
    X_test = test[feature_cols]
    y_test = test[columns.temperature]

    models = {
        "Persistence_Baseline": None,
        "Linear_Regression": LinearRegression(),
        "Random_Forest": RandomForestRegressor(n_estimators=100, min_samples_leaf=3, random_state=42, n_jobs=1),
        "Gradient_Boosting": GradientBoostingRegressor(random_state=42),
    }

    predictions = {"actual": y_test.values, "date": test[columns.date].astype(str).values}
    metrics = {}

    baseline_pred = test["temp_lag_1"].values
    predictions["Persistence_Baseline"] = baseline_pred
    metrics["Persistence_Baseline"] = regression_metrics(y_test, baseline_pred)

    trained = {}
    for name, model in models.items():
        if model is None:
            continue
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions[name] = pred
        metrics[name] = regression_metrics(y_test, pred)
        trained[name] = model
        joblib.dump(model, MODEL_DIR / f"{name.lower()}.joblib")

    ensemble_pred = np.mean(
        [predictions["Linear_Regression"], predictions["Random_Forest"], predictions["Gradient_Boosting"]],
        axis=0,
    )
    predictions["Ensemble_Average"] = ensemble_pred
    metrics["Ensemble_Average"] = regression_metrics(y_test, ensemble_pred)

    results = pd.DataFrame(predictions)
    results.to_csv(OUTPUT_DIR / "forecast_predictions.csv", index=False)
    pd.DataFrame(metrics).T.sort_values("RMSE").to_csv(OUTPUT_DIR / "model_metrics.csv")

    plt.figure(figsize=(13, 6))
    preview = results.tail(min(250, len(results)))
    plt.plot(preview["date"], preview["actual"], label="Actual", linewidth=2)
    plt.plot(preview["date"], preview["Ensemble_Average"], label="Ensemble Forecast", linewidth=2)
    plt.xticks(rotation=45, ha="right")
    plt.title("Temperature Forecast: Actual vs Ensemble")
    plt.xlabel("Date")
    plt.ylabel("Temperature")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "forecast_actual_vs_ensemble.png", dpi=160)
    plt.close()

    feature_importance = compute_feature_importance(trained["Random_Forest"], X_test, y_test, feature_cols)
    return pd.DataFrame(metrics).T.sort_values("RMSE"), feature_importance


def compute_feature_importance(model, X_test: pd.DataFrame, y_test: pd.Series, feature_cols: list[str]) -> dict:
    rf_importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    sample_size = min(5000, len(X_test))
    X_sample = X_test.sample(sample_size, random_state=42)
    y_sample = y_test.loc[X_sample.index]
    perm = permutation_importance(model, X_sample, y_sample, n_repeats=3, random_state=42, n_jobs=1)
    perm_importance = pd.Series(perm.importances_mean, index=feature_cols).sort_values(ascending=False)

    importance_df = pd.DataFrame(
        {
            "random_forest_importance": rf_importance,
            "permutation_importance": perm_importance,
        }
    ).sort_values("permutation_importance", ascending=False)
    importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv")

    plt.figure(figsize=(10, 6))
    importance_df["permutation_importance"].head(12).sort_values().plot(kind="barh", color="#0f766e")
    plt.title("Top Forecasting Feature Importance")
    plt.xlabel("Permutation Importance")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "feature_importance.png", dpi=160)
    plt.close()

    return {
        "top_random_forest_features": rf_importance.head(10).round(4).to_dict(),
        "top_permutation_features": perm_importance.head(10).round(4).to_dict(),
    }


def detect_anomalies(df: pd.DataFrame, columns: Columns) -> dict:
    anomaly_cols = [
        col
        for col in [columns.temperature, columns.precipitation, columns.humidity, columns.wind, columns.pressure]
        if col and col in df.columns
    ]
    anomaly_cols.extend([col for col in columns.air_quality[:5] if col in df.columns])
    anomaly_cols = list(dict.fromkeys(anomaly_cols))
    if len(anomaly_cols) < 2:
        return {"note": "Not enough numeric weather columns for Isolation Forest anomaly detection."}

    model = IsolationForest(contamination=0.03, random_state=42)
    anomaly_frame = df[[columns.date] + anomaly_cols].copy()
    anomaly_frame["anomaly_flag"] = model.fit_predict(df[anomaly_cols])
    anomaly_frame["is_anomaly"] = anomaly_frame["anomaly_flag"] == -1
    anomaly_frame.to_csv(OUTPUT_DIR / "anomaly_detection_results.csv", index=False)

    plt.figure(figsize=(12, 5))
    normal = anomaly_frame[~anomaly_frame["is_anomaly"]]
    abnormal = anomaly_frame[anomaly_frame["is_anomaly"]]
    plt.scatter(normal[columns.date], normal[columns.temperature], s=8, alpha=0.35, label="Normal")
    plt.scatter(abnormal[columns.date], abnormal[columns.temperature], s=18, color="#dc2626", label="Anomaly")
    plt.title("Detected Temperature Anomalies")
    plt.xlabel("Date")
    plt.ylabel("Temperature")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "temperature_anomalies.png", dpi=160)
    plt.close()

    return {
        "method": "IsolationForest",
        "features_used": anomaly_cols,
        "anomaly_count": int(anomaly_frame["is_anomaly"].sum()),
        "anomaly_rate": round(float(anomaly_frame["is_anomaly"].mean()), 4),
    }


def climate_and_environment_analysis(df: pd.DataFrame, columns: Columns) -> dict:
    summary = {}
    df = df.copy()
    df["year_month"] = df[columns.date].dt.to_period("M").astype(str)

    if columns.country:
        country_trends = (
            df.groupby([columns.country, "year_month"])[columns.temperature]
            .mean()
            .reset_index()
        )
        country_trends.to_csv(OUTPUT_DIR / "country_monthly_temperature_trends.csv", index=False)
        summary["country_temperature_examples"] = (
            country_trends.groupby(columns.country)[columns.temperature].mean().sort_values(ascending=False).head(8).round(2).to_dict()
        )

    if "continent" in df.columns:
        plt.figure(figsize=(12, 6))
        continent_monthly = df.groupby(["continent", "year_month"])[columns.temperature].mean().reset_index()
        sns.lineplot(data=continent_monthly, x="year_month", y=columns.temperature, hue="continent")
        plt.xticks(rotation=45, ha="right")
        plt.title("Monthly Temperature Patterns by Continent")
        plt.xlabel("Month")
        plt.ylabel("Average Temperature")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "continent_temperature_patterns.png", dpi=160)
        plt.close()

    if columns.air_quality:
        aq_cols = [col for col in columns.air_quality if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        weather_cols = [col for col in [columns.temperature, columns.precipitation, columns.humidity, columns.wind, columns.pressure] if col]
        corr = df[aq_cols + weather_cols].corr().loc[aq_cols, weather_cols]
        corr.to_csv(OUTPUT_DIR / "air_quality_weather_correlations.csv")
        summary["strongest_air_quality_weather_correlations"] = (
            corr.abs().stack().sort_values(ascending=False).head(10).round(3).to_dict()
        )

    return summary


def spatial_analysis(df: pd.DataFrame, columns: Columns) -> dict:
    if not (columns.latitude and columns.longitude):
        return {"note": "Latitude and longitude columns were not available for spatial analysis."}

    sample = df.sample(min(15000, len(df)), random_state=42)
    fig = px.scatter_geo(
        sample,
        lat=columns.latitude,
        lon=columns.longitude,
        color=columns.temperature,
        hover_name=columns.city if columns.city else columns.country,
        projection="natural earth",
        title="Global Temperature Distribution",
        color_continuous_scale="RdYlBu_r",
    )
    fig.write_html(FIGURE_DIR / "spatial_temperature_map.html")

    if columns.country:
        country_geo = (
            df.groupby(columns.country)
            .agg(
                avg_temperature=(columns.temperature, "mean"),
                avg_latitude=(columns.latitude, "mean"),
                avg_longitude=(columns.longitude, "mean"),
            )
            .reset_index()
            .sort_values("avg_temperature", ascending=False)
        )
        country_geo.to_csv(OUTPUT_DIR / "country_spatial_temperature_summary.csv", index=False)
        return {
            "hottest_country_averages": country_geo.head(10).round(2).to_dict(orient="records"),
            "map_file": "outputs/figures/spatial_temperature_map.html",
        }
    return {"map_file": "outputs/figures/spatial_temperature_map.html"}


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def write_report(
    cleaning: dict,
    eda: dict,
    anomaly: dict,
    metrics: pd.DataFrame,
    feature_importance: dict,
    climate_env: dict,
    spatial: dict,
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_model = metrics.index[0]
    best_rmse = metrics.iloc[0]["RMSE"]
    report = f"""# Weather Trend Forecasting Report

## PM Accelerator Mission

Product Manager Accelerator describes its mission as helping professionals become confident product leaders and, through PMA Kids, breaking down financial barriers by offering free product-management education to underserved teenagers. This project aligns with that mission by turning a real-world weather dataset into decision-ready insights using reproducible data science.

Source reviewed: https://www.pmaccelerator.io/

## Project Objective

This project analyzes the Kaggle Global Weather Repository dataset and forecasts future weather trends using `last_updated` as the time-series field. The work covers the basic assessment requirements and extends them with anomaly detection, multiple forecasting models, an ensemble model, feature importance, air-quality analysis, climate patterns, and spatial/geographical analysis.

## Data Cleaning and Preprocessing

- Parsed `last_updated` into datetime format.
- Removed records without usable timestamps or target temperature values.
- Imputed numeric missing values with medians and categorical missing values with `Unknown`.
- Detected and clipped numeric outliers with the IQR rule.
- Created standardized numeric columns for normalized downstream analysis.

Cleaning summary:

```json
{json.dumps(json_safe(cleaning), indent=2, default=str)}
```

## Exploratory Data Analysis

EDA focused on temperature trends, precipitation trends, correlations, and geographic variation.

Key EDA summary:

```json
{json.dumps(json_safe(eda), indent=2, default=str)}
```

Generated figures:

- `outputs/figures/temperature_trend.png`
- `outputs/figures/precipitation_trend.png`
- `outputs/figures/correlation_heatmap.png`

## Anomaly Detection

Anomalies were identified with Isolation Forest using weather and air-quality numeric features. This helps flag unusual combinations such as extreme temperature, wind, pressure, precipitation, or pollution readings.

```json
{json.dumps(json_safe(anomaly), indent=2, default=str)}
```

Figure:

- `outputs/figures/temperature_anomalies.png`

## Forecasting Models

The forecasting target is temperature. Models use date-derived features, seasonal sine/cosine features, lag features, rolling temperature features, and available weather covariates.

Models compared:

- Persistence baseline
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- Average ensemble of Linear Regression, Random Forest, and Gradient Boosting

Model metrics:

{metrics.to_markdown()}

Best model by RMSE: **{best_model}** with RMSE **{best_rmse:.4f}**.

Figure:

- `outputs/figures/forecast_actual_vs_ensemble.png`

## Feature Importance

Feature importance was assessed using Random Forest impurity-based importance and permutation importance.

```json
{json.dumps(json_safe(feature_importance), indent=2, default=str)}
```

Figure:

- `outputs/figures/feature_importance.png`

## Climate, Environmental, and Geographic Insights

Climate and environmental analysis includes monthly temperature variation by geography, country-level temperature trends, and correlations between air-quality variables and weather parameters.

```json
{json.dumps(json_safe(climate_env), indent=2, default=str)}
```

Spatial analysis:

```json
{json.dumps(json_safe(spatial), indent=2, default=str)}
```

Generated files:

- `outputs/country_monthly_temperature_trends.csv`
- `outputs/air_quality_weather_correlations.csv`
- `outputs/figures/spatial_temperature_map.html`

## Conclusion

The project demonstrates an end-to-end data science workflow: cleaning, exploratory analysis, anomaly detection, supervised forecasting, ensemble modeling, model evaluation, feature importance, and climate/geospatial interpretation. The modular Python script can be rerun as the Kaggle repository updates with new daily weather records.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_data(resolve_data_path())
    columns = discover_columns(raw)
    cleaned, cleaning_summary = clean_data(raw, columns)
    cleaned.to_csv(OUTPUT_DIR / "cleaned_weather_data.csv", index=False)

    eda_summary = save_basic_eda(cleaned, columns)
    anomaly_summary = detect_anomalies(cleaned, columns)
    metrics, feature_importance = build_forecasting_models(cleaned, columns)
    climate_env_summary = climate_and_environment_analysis(cleaned, columns)
    spatial_summary = spatial_analysis(cleaned, columns)

    write_report(
        cleaning_summary,
        eda_summary,
        anomaly_summary,
        metrics,
        feature_importance,
        climate_env_summary,
        spatial_summary,
    )

    print("Analysis complete.")
    print(f"Report: {REPORT_PATH}")
    print(f"Figures: {FIGURE_DIR}")
    print(f"Metrics: {OUTPUT_DIR / 'model_metrics.csv'}")


if __name__ == "__main__":
    main()
