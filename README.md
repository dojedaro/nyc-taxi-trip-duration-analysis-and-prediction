# NYC Taxi Trip Duration — Analysis and Prediction

An end-to-end data science and machine learning project using the NYC taxi trip duration dataset.  
This repository combines **exploratory data analysis, geospatial visualization, and supervised machine learning** to understand and predict taxi trip duration in New York City.

The project is intentionally structured as **two focused notebooks** that represent distinct stages of a real-world ML workflow.

---

## Project overview

This repository answers two key questions:

1. **How do NYC taxi trips behave spatially and temporally?**
2. **Can we accurately predict trip duration using location and time-based features?**

Both notebooks use the same dataset but serve **different purposes**, reflecting a practical end-to-end pipeline:  
**understand the data first, then build a predictive model.**

---

## Notebook 1 — Exploratory Data Analysis & Geospatial Insights

**File:** `01_eda_geospatial_analysis.ipynb`

This notebook focuses on **understanding and validating the data** before modeling.

### What it covers

- Dataset inspection and data-type validation
- Datetime parsing and feature extraction
- Distance and speed computation
- Outlier filtering and data cleaning
- Distribution analysis:
  - Trip duration
  - Distance
  - Speed
- Temporal patterns:
  - Average trip duration by hour
  - Trip volume by hour
  - Average duration by day of week
- Passenger count analysis
- Spatial visualization:
  - Pickup and dropoff heatmaps (Folium)
  - Pickup hotspot clustering using DBSCAN
- Interactive exploration using Plotly

### Goal

Identify mobility patterns, validate data quality, and uncover spatial and temporal dynamics that inform downstream feature engineering.

---

## Notebook 2 — Trip Duration Prediction (Machine Learning)

**File:** `02_trip_duration_prediction.ipynb`

This notebook builds on insights from the EDA and focuses on **predictive modeling and data robustness**.

### What it covers

- Explicit data diagnostics to detect real-world issues
- Recalculation and validation of:
  - Trip duration (from timestamps)
  - Distance (Haversine formula)
- Automatic replacement of faulty columns when discrepancies are detected
- Aggressive but controlled outlier removal based on:
  - Duration
  - Distance
  - Speed
  - NYC geographic boundaries
- Feature engineering grounded in urban mobility behavior:
  - Haversine distance
  - Manhattan distance
  - Pickup hour
  - Day of week
  - Weekend indicator
  - Rush-hour indicator
  - Pickup and dropoff coordinates
- Supervised regression using LightGBM
- Model evaluation:
  - R² score
  - RMSE
- Feature importance analysis


### Model performance

After fixing data issues and engineering features:

- **R² score:** ~0.77  
- **RMSE:** ~311 seconds  

This represents a strong baseline for a noisy, real-world regression problem.
