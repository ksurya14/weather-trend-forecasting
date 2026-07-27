# Weather Trend Forecasting Report

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
{
  "rows_after_cleaning": 154361,
  "columns": 71,
  "missing_before_top15": {
    "country": 0,
    "feels_like_fahrenheit": 0,
    "visibility_miles": 0,
    "uv_index": 0,
    "gust_mph": 0,
    "gust_kph": 0,
    "air_quality_Carbon_Monoxide": 0,
    "air_quality_Ozone": 0,
    "air_quality_Nitrogen_dioxide": 0,
    "air_quality_Sulphur_dioxide": 0,
    "air_quality_PM2.5": 0,
    "air_quality_PM10": 0,
    "air_quality_us-epa-index": 0,
    "air_quality_gb-defra-index": 0,
    "sunrise": 0
  },
  "outliers_iqr_clipped": {
    "latitude": 0,
    "longitude": 11872,
    "last_updated_epoch": 0,
    "temperature_celsius": 2983,
    "temperature_fahrenheit": 2979,
    "wind_mph": 2375,
    "wind_kph": 2577,
    "wind_degree": 0,
    "pressure_mb": 4260,
    "pressure_in": 5502,
    "precip_mm": 30972,
    "humidity": 0,
    "cloud": 0,
    "feels_like_celsius": 3026,
    "feels_like_fahrenheit": 3022,
    "uv_index": 155,
    "gust_mph": 4169,
    "gust_kph": 4119,
    "air_quality_Carbon_Monoxide": 13622,
    "air_quality_Ozone": 4500,
    "air_quality_Nitrogen_dioxide": 16119,
    "air_quality_Sulphur_dioxide": 21318,
    "air_quality_PM2.5": 12835,
    "air_quality_PM10": 16424,
    "air_quality_us-epa-index": 9995,
    "air_quality_gb-defra-index": 13793,
    "moon_illumination": 0
  },
  "numeric_columns_scaled": [
    "latitude_scaled",
    "longitude_scaled",
    "last_updated_epoch_scaled",
    "temperature_celsius_scaled",
    "temperature_fahrenheit_scaled",
    "wind_mph_scaled",
    "wind_kph_scaled",
    "wind_degree_scaled",
    "pressure_mb_scaled",
    "pressure_in_scaled",
    "precip_mm_scaled",
    "precip_in_scaled",
    "humidity_scaled",
    "cloud_scaled",
    "feels_like_celsius_scaled",
    "feels_like_fahrenheit_scaled",
    "visibility_km_scaled",
    "visibility_miles_scaled",
    "uv_index_scaled",
    "gust_mph_scaled",
    "gust_kph_scaled",
    "air_quality_Carbon_Monoxide_scaled",
    "air_quality_Ozone_scaled",
    "air_quality_Nitrogen_dioxide_scaled",
    "air_quality_Sulphur_dioxide_scaled",
    "air_quality_PM2.5_scaled",
    "air_quality_PM10_scaled",
    "air_quality_us-epa-index_scaled",
    "air_quality_gb-defra-index_scaled",
    "moon_illumination_scaled"
  ]
}
```

## Exploratory Data Analysis

EDA focused on temperature trends, precipitation trends, correlations, and geographic variation.

Key EDA summary:

```json
{
  "top_10_hottest_countries": {
    "Saudi Arabien": 45.0,
    "Marrocos": 40.3,
    "Turkm\u00e9nistan": 37.8,
    "\u0422\u0443\u0440\u0446\u0438\u044f": 34.0,
    "Qatar": 32.58,
    "United Arab Emirates": 32.37,
    "Cambodia": 32.06,
    "Oman": 31.89,
    "Djibouti": 31.42,
    "Thailand": 31.25
  },
  "temperature_summary": {
    "count": 154361.0,
    "mean": 21.47,
    "std": 9.17,
    "min": -1.6,
    "25%": 16.1,
    "50%": 23.7,
    "75%": 27.9,
    "max": 45.6
  },
  "precipitation_summary": {
    "count": 154361.0,
    "mean": 0.01,
    "std": 0.02,
    "min": 0.0,
    "25%": 0.0,
    "50%": 0.0,
    "75%": 0.02,
    "max": 0.05
  }
}
```

Generated figures:

- `outputs/figures/temperature_trend.png`
- `outputs/figures/precipitation_trend.png`
- `outputs/figures/correlation_heatmap.png`

## Anomaly Detection

Anomalies were identified with Isolation Forest using weather and air-quality numeric features. This helps flag unusual combinations such as extreme temperature, wind, pressure, precipitation, or pollution readings.

```json
{
  "method": "IsolationForest",
  "features_used": [
    "temperature_celsius",
    "precip_mm",
    "humidity",
    "wind_kph",
    "pressure_mb",
    "air_quality_Carbon_Monoxide",
    "air_quality_Ozone",
    "air_quality_Nitrogen_dioxide",
    "air_quality_Sulphur_dioxide",
    "air_quality_PM2.5"
  ],
  "anomaly_count": 4631,
  "anomaly_rate": 0.03
}
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

|                      |    MAE |   RMSE |   MAPE_percent |     R2 |
|:---------------------|-------:|-------:|---------------:|-------:|
| Ensemble_Average     | 1.3667 | 1.9392 |        20.9215 | 0.9519 |
| Random_Forest        | 1.3687 | 1.9461 |        21.7453 | 0.9515 |
| Gradient_Boosting    | 1.4095 | 1.9821 |        20.7188 | 0.9497 |
| Linear_Regression    | 1.4591 | 2.0502 |        21.4792 | 0.9462 |
| Persistence_Baseline | 1.5747 | 2.3142 |        21.3728 | 0.9315 |

Best model by RMSE: **Ensemble_Average** with RMSE **1.9392**.

Figure:

- `outputs/figures/forecast_actual_vs_ensemble.png`

## Feature Importance

Feature importance was assessed using Random Forest impurity-based importance and permutation importance.

```json
{
  "top_random_forest_features": {
    "temp_lag_1": 0.8433,
    "temp_rolling_7": 0.1128,
    "humidity": 0.0155,
    "pressure_mb": 0.0063,
    "temp_lag_7": 0.005,
    "wind_kph": 0.0046,
    "cos_day": 0.0041,
    "sin_day": 0.0035,
    "dayofyear": 0.0019,
    "precip_mm": 0.0011
  },
  "top_permutation_features": {
    "temp_lag_1": 0.8909,
    "temp_rolling_7": 0.1718,
    "humidity": 0.0523,
    "pressure_mb": 0.0148,
    "temp_lag_7": 0.0081,
    "wind_kph": 0.0011,
    "precip_mm": 0.0006,
    "cos_day": 0.0005,
    "year": 0.0,
    "month": -0.0
  }
}
```

Figure:

- `outputs/figures/feature_importance.png`

## Climate, Environmental, and Geographic Insights

Climate and environmental analysis includes monthly temperature variation by geography, country-level temperature trends, and correlations between air-quality variables and weather parameters.

```json
{
  "country_temperature_examples": {
    "Saudi Arabien": 45.0,
    "Marrocos": 40.3,
    "Turkm\u00e9nistan": 37.8,
    "\u0422\u0443\u0440\u0446\u0438\u044f": 34.0,
    "Qatar": 32.68,
    "United Arab Emirates": 32.43,
    "Cambodia": 32.04,
    "Oman": 31.99
  },
  "strongest_air_quality_weather_correlations": {
    "('air_quality_Ozone', 'humidity')": 0.417,
    "('air_quality_PM10', 'humidity')": 0.354,
    "('air_quality_PM2.5', 'humidity')": 0.286,
    "('air_quality_us-epa-index', 'humidity')": 0.281,
    "('air_quality_gb-defra-index', 'humidity')": 0.28,
    "('air_quality_Carbon_Monoxide', 'wind_kph')": 0.261,
    "('air_quality_Ozone', 'temperature_celsius')": 0.252,
    "('air_quality_Sulphur_dioxide', 'humidity')": 0.232,
    "('air_quality_PM10', 'precip_mm')": 0.221,
    "('air_quality_Nitrogen_dioxide', 'temperature_celsius')": 0.21
  }
}
```

Spatial analysis:

```json
{
  "hottest_country_averages": [
    {
      "country": "Saudi Arabien",
      "avg_temperature": 45.0,
      "avg_latitude": 24.64,
      "avg_longitude": 46.77
    },
    {
      "country": "Marrocos",
      "avg_temperature": 40.3,
      "avg_latitude": 31.63,
      "avg_longitude": -8.0
    },
    {
      "country": "Turkm\u00e9nistan",
      "avg_temperature": 37.8,
      "avg_latitude": 37.7,
      "avg_longitude": 65.37
    },
    {
      "country": "\u0422\u0443\u0440\u0446\u0438\u044f",
      "avg_temperature": 34.0,
      "avg_latitude": 39.55,
      "avg_longitude": 27.62
    },
    {
      "country": "Qatar",
      "avg_temperature": 32.58,
      "avg_latitude": 25.29,
      "avg_longitude": 51.53
    },
    {
      "country": "United Arab Emirates",
      "avg_temperature": 32.37,
      "avg_latitude": 24.47,
      "avg_longitude": 54.37
    },
    {
      "country": "Cambodia",
      "avg_temperature": 32.06,
      "avg_latitude": 11.55,
      "avg_longitude": 104.92
    },
    {
      "country": "Oman",
      "avg_temperature": 31.89,
      "avg_latitude": 23.61,
      "avg_longitude": 58.59
    },
    {
      "country": "Djibouti",
      "avg_temperature": 31.42,
      "avg_latitude": 11.6,
      "avg_longitude": 43.15
    },
    {
      "country": "Thailand",
      "avg_temperature": 31.25,
      "avg_latitude": 16.27,
      "avg_longitude": 100.65
    }
  ],
  "map_file": "outputs/figures/spatial_temperature_map.html"
}
```

Generated files:

- `outputs/country_monthly_temperature_trends.csv`
- `outputs/air_quality_weather_correlations.csv`
- `outputs/figures/spatial_temperature_map.html`

## Conclusion

The project demonstrates an end-to-end data science workflow: cleaning, exploratory analysis, anomaly detection, supervised forecasting, ensemble modeling, model evaluation, feature importance, and climate/geospatial interpretation. The modular Python script can be rerun as the Kaggle repository updates with new daily weather records.
