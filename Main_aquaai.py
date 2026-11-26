# ================================================================
# AquaAI — ONE-FILE MULTI-PAGE STREAMLIT APPLICATION
# Working standalone version
#
# Install:
#   python3 -m pip install -r requirements.txt
#
# Run:
#   streamlit run aquaai.py
#
# ADMINISTRATOR SETUP (no admin password is stored in this file)
#   The first administrator is created automatically, once, when the app
#   starts and no admin exists yet. Provide the credentials with EITHER
#   environment variables:
#       AQUAAI_ADMIN_USERNAME   (a username or an email, 3+ characters)
#       AQUAAI_ADMIN_PASSWORD   (10+ characters, with a letter and a number)
#   OR Streamlit Secrets (.streamlit/secrets.toml, or the host's secrets UI):
#       [admin]
#       username = "..."
#       password = "..."
#   After the account exists, change the password from Admin > Admin Profile;
#   the variables/secrets are then ignored. Optional: AQUAAI_DB_PATH moves the
#   SQLite database to a persistent location.
# ================================================================

from __future__ import annotations

import calendar
import hashlib
import hmac
import html
import json
import math
import os
import re
import secrets
import sqlite3
import tempfile
import warnings
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False


# ----------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------
st.set_page_config(
    page_title="AquaAI | Smart Water Management",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "aquaai_data"

# Some hosting environments (e.g. read-only containers) don't allow writing
# next to the script. Persisting cached CSV/model files is a nice-to-have,
# not a requirement for the app to run, so failures here must never crash
# the app on import.
try:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _DATA_DIR_WRITABLE = True
except OSError:
    _DATA_DIR_WRITABLE = False

DATA_FILE = DATA_DIR / "household_water_data.csv"
PREDICTIONS_FILE = DATA_DIR / "predictions.csv"
MODEL_FILE = DATA_DIR / "water_demand_model.pkl"

# User accounts live in SQLite. Unlike the cached CSV/model artefacts above,
# the database is what makes accounts work, so if the app folder is read-only
# we fall back to the system temp directory. Set AQUAAI_DB_PATH to put the
# database somewhere persistent (e.g. a mounted volume) in production.
_db_override = os.environ.get("AQUAAI_DB_PATH")
if _db_override:
    DB_FILE = Path(_db_override)
else:
    DB_FILE = (
        DATA_DIR if _DATA_DIR_WRITABLE else Path(tempfile.gettempdir()) / "aquaai_data"
    ) / "aquaai_users.db"

RANDOM_SEED = 42
TARGET = "actual_consumption_lpd"

# Baseline flush volume (litres) used as the "before efficiency" comparison
# point when estimating potential water savings from low-flow toilets.
BASELINE_FLUSH_LITRES = 8
EFFICIENT_FLUSH_LITRES = 4

FEATURES = [
    "household_size",
    "temperature_c",
    "rainfall_mm",
    "water_supply_lpd",
    "previous_consumption_lpd",
    "toilet_flushes",
    "flush_litres",
    "greywater_available_lpd",
    "recycled_water_lpd",
    "leakage_lpd",
    "awareness_score",
    "water_stress_index",
    "month",
    "day_of_week",
]


# ----------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if "aqua_navigation" not in st.session_state:
    st.session_state.aqua_navigation = "🏠 Overview"

# None = browsing as a guest; otherwise the signed-in user's database id.
if "user_id" not in st.session_state:
    st.session_state.user_id = None

# Administrator session: a random token whose hash is stored server-side in
# SQLite. It is separate from the user session and is re-validated on every
# admin page view and every admin action.
if "admin_token" not in st.session_state:
    st.session_state.admin_token = None

if "admin_navigation" not in st.session_state:
    st.session_state.admin_navigation = "📊 Admin Overview"


# ----------------------------------------------------------------
# CSS
# ----------------------------------------------------------------
def inject_css(dark: bool = False) -> None:
    """Inject the app's custom CSS theme (light or dark variant)."""
    if dark:
        bg = "linear-gradient(135deg,#06151e,#0a2533 55%,#031017)"
        text = "#e9fbff"
        section_color = "#d8f7ff"
        glass_bg = "rgba(255,255,255,.075)"
        glass_border = "rgba(255,255,255,.12)"
        dashboard_bg = "rgba(255,255,255,.075)"
        dashboard_text = "#bdebf5"
    else:
        bg = (
            "radial-gradient(circle at 10% 10%,rgba(95,224,239,.28),transparent 28%),"
            "radial-gradient(circle at 90% 15%,rgba(0,119,182,.18),transparent 27%),"
            "linear-gradient(135deg,#eafcff,#f8fdff 45%,#eef9ff)"
        )
        text = "#102a43"
        section_color = "#073b59"
        glass_bg = "rgba(255,255,255,.58)"
        glass_border = "rgba(255,255,255,.72)"
        dashboard_bg = "rgba(255,255,255,.58)"
        dashboard_text = "#416b82"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: Inter, sans-serif;
        }}

        .stApp {{
            background: {bg};
            color: {text};
            min-height: 100vh;
        }}

        .block-container {{
            max-width: 1500px;
            padding: 1.4rem 2rem 4rem;
        }}

        /* HERO */
        .hero {{
            position: relative;
            overflow: hidden;
            padding: 38px 42px;
            margin-bottom: 25px;
            border-radius: 30px;
            color: white;
            background: linear-gradient(
                115deg,
                rgba(3,48,71,.98),
                rgba(0,119,182,.88),
                rgba(18,181,203,.78)
            );
            box-shadow: 0 28px 70px rgba(0,92,135,.25);
        }}

        .hero:after {{
            content: "";
            position: absolute;
            width: 320px;
            height: 320px;
            right: -100px;
            top: -130px;
            border-radius: 50%;
            background: rgba(255,255,255,.09);
        }}

        .hero::before {{
            content: "💧";
            position: absolute;
            right: 7%;
            bottom: 8%;
            font-size: 92px;
            opacity: .13;
            transform: rotate(7deg);
        }}

        .hero h1 {{
            position: relative;
            z-index: 2;
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: clamp(40px,5vw,62px);
            margin: 10px 0 0;
            font-weight: 800;
            letter-spacing: -.05em;
        }}

        .hero p {{
            position: relative;
            z-index: 2;
            max-width: 900px;
            font-size: 17px;
            line-height: 1.6;
            color: rgba(255,255,255,.92);
        }}

        .badge {{
            display: inline-block;
            position: relative;
            z-index: 3;
            padding: 7px 12px;
            margin: 3px 4px 5px 0;
            border-radius: 999px;
            background: rgba(255,255,255,.12);
            border: 1px solid rgba(255,255,255,.22);
            font-size: 12px;
            font-weight: 700;
        }}

        .section-title {{
            margin: 28px 0 13px;
            color: {section_color};
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 25px;
            font-weight: 800;
        }}

        /* METRIC CARDS */
        .metric {{
            position: relative;
            overflow: hidden;
            isolation: isolate;
            padding: 24px;
            border-radius: 25px;
            color: white;
            min-height: 150px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            border: 1px solid rgba(255,255,255,.28);
            background: linear-gradient(135deg,#177da3,#42bfd1);
            box-shadow:
                0 18px 35px rgba(4,73,103,.16),
                0 5px 12px rgba(4,73,103,.08),
                inset 0 1px 0 rgba(255,255,255,.22);
            animation: aquaMetricFloat 5.8s ease-in-out infinite;
            transition: transform .28s ease, box-shadow .28s ease;
        }}

        .metric::before {{
            content: "";
            position: absolute;
            width: 105px;
            height: 105px;
            right: -18px;
            bottom: -30px;
            border-radius: 50%;
            background: rgba(255,255,255,.15);
            z-index: -1;
        }}

        .metric::after {{
            content: "";
            position: absolute;
            width: 210px;
            height: 70px;
            left: -45px;
            top: -32px;
            border-radius: 50%;
            background: rgba(255,255,255,.08);
            transform: rotate(-10deg);
            z-index: -1;
        }}

        .metric:hover {{
            transform: translateY(-8px) scale(1.015);
            box-shadow: 0 28px 50px rgba(4,73,103,.23);
        }}

        .metric-0 {{ background: linear-gradient(135deg,#207b9f,#42b6cb); }}
        .metric-1 {{ background: linear-gradient(135deg,#17865e,#4cc08e); animation-delay:-.8s; }}
        .metric-2 {{ background: linear-gradient(135deg,#6847a5,#8e6bd1); animation-delay:-1.6s; }}
        .metric-3 {{ background: linear-gradient(135deg,#bd6c17,#efa947); animation-delay:-2.4s; }}
        .metric-4 {{ background: linear-gradient(135deg,#ad3945,#df626c); animation-delay:-3.2s; }}

        .metric h4 {{
            position: relative;
            z-index: 2;
            margin: 0;
            font-size: 15px;
            opacity: .92;
            font-weight: 700;
        }}

        .metric h2 {{
            position: relative;
            z-index: 2;
            margin: 22px 0 0;
            font-size: clamp(27px, 2.2vw, 38px);
            line-height: 1.05;
            letter-spacing: -.035em;
            font-weight: 800;
            text-shadow: 0 2px 12px rgba(0,0,0,.12);
        }}

        @keyframes aquaMetricFloat {{
            0%,100% {{ transform:translateY(0); }}
            50% {{ transform:translateY(-7px); }}
        }}

        /* BUTTONS */
        .stButton > button,
        .stDownloadButton > button {{
            border: 0;
            border-radius: 15px;
            min-height: 46px;
            font-weight: 700;
            color: white;
            background: linear-gradient(135deg,#075985,#12b5cb);
            box-shadow: 0 9px 22px rgba(7,89,133,.20);
            transition: all .25s ease;
        }}

        .stButton > button:hover,
        .stDownloadButton > button:hover {{
            transform: translateY(-2px);
            filter: brightness(1.06);
            box-shadow: 0 14px 28px rgba(7,89,133,.27);
        }}

        /* SIDEBAR */
        section[data-testid="stSidebar"] {{
            min-width: 330px;
            background:
                radial-gradient(circle at 15% 15%,rgba(0,212,255,.12),transparent 28%),
                linear-gradient(180deg,#021923 0%,#043b55 50%,#021923 100%);
        }}

        section[data-testid="stSidebar"] > div {{
            padding: 1.1rem .9rem 1.5rem;
        }}

        section[data-testid="stSidebar"] * {{
            color: #ecfeff;
        }}

        .aqua-brand {{
            text-align: center;
            padding: 10px 0 25px;
        }}

        .aqua-drop {{
            font-size: 45px;
            filter: drop-shadow(0 0 15px rgba(0,212,255,.5));
            animation: aqua-float 3s ease-in-out infinite;
        }}

        .aqua-name {{
            color: white;
            font-size: 29px;
            font-weight: 800;
        }}

        .aqua-tagline {{
            color: #7ddcf0;
            font-size: 9px;
            letter-spacing: 2px;
            font-weight: 700;
            margin-top: 4px;
        }}

        .nav-title {{
            color: #ffffff;
            font-size: 14px;
            font-weight: 800;
            letter-spacing: 2px;
            margin: 20px 8px 12px;
        }}

        section[data-testid="stSidebar"] .stButton {{
            margin: 0 0 10px 0;
        }}

        section[data-testid="stSidebar"] .stButton > button {{
            position: relative;
            width: 100%;
            min-height: 60px;
            padding: 0 18px;
            display: flex;
            align-items: center;
            justify-content: flex-start;
            border-radius: 17px;
            border: 1px solid rgba(180,240,255,.18);
            color: #e9fbff !important;
            background: linear-gradient(
                135deg,
                rgba(255,255,255,.13),
                rgba(255,255,255,.045)
            );
            box-shadow:
                0 10px 25px rgba(0,0,0,.18),
                inset 0 1px 0 rgba(255,255,255,.18);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            font-size: 16px;
            font-weight: 700;
            text-align: left;
            overflow: hidden;
            transition: transform .25s ease, background .25s ease,
                        border-color .25s ease, box-shadow .25s ease;
        }}

        section[data-testid="stSidebar"] .stButton > button::before {{
            content: "";
            position: absolute;
            left: 0;
            top: 0;
            width: 100%;
            height: 48%;
            background: linear-gradient(180deg,rgba(255,255,255,.13),transparent);
            pointer-events: none;
        }}

        section[data-testid="stSidebar"] .stButton > button:hover {{
            transform: translateY(-4px) scale(1.012) !important;
            background: linear-gradient(
                135deg,
                rgba(50,205,235,.27),
                rgba(255,255,255,.08)
            );
            border-color: rgba(150,235,255,.48);
            box-shadow:
                0 16px 32px rgba(0,0,0,.28),
                0 0 20px rgba(50,210,240,.13);
        }}

        .nav-active .stButton > button {{
            background: linear-gradient(
                135deg,
                rgba(35,194,224,.38),
                rgba(70,110,220,.22)
            ) !important;
            border-color: rgba(165,240,255,.68) !important;
            box-shadow:
                0 15px 34px rgba(0,65,95,.34),
                0 0 24px rgba(40,210,240,.18),
                inset 0 1px 0 rgba(255,255,255,.36) !important;
            transform: translateX(4px);
        }}

        .sidebar-status {{
            margin-top: 18px;
            padding: 15px;
            border-radius: 15px;
            background: rgba(255,255,255,.065);
            border: 1px solid rgba(255,255,255,.09);
            color: #bdebf5;
            font-size: 11px;
            line-height: 1.65;
            box-shadow: inset 0 1px 0 rgba(255,255,255,.08);
        }}

        .sidebar-footer {{
            text-align: center;
            color: #6caaba;
            font-size: 9px;
            margin-top: 25px;
            line-height: 1.6;
        }}

        .dashboard-status {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 20px;
            margin: 8px 0 20px;
            padding: 15px 20px;
            border-radius: 17px;
            color: {dashboard_text};
            background: {dashboard_bg};
            border: 1px solid rgba(255,255,255,.8);
            box-shadow: 0 10px 28px rgba(5,72,105,.07);
            backdrop-filter: blur(15px);
            font-size: 14px;
        }}

        .dashboard-status strong {{ color: {section_color}; }}

        @keyframes aqua-float {{
            0%,100% {{ transform:translateY(0); }}
            50% {{ transform:translateY(-5px); }}
        }}

        @media(max-width:700px) {{
            .block-container {{ padding: 1rem; }}
            .hero {{ padding: 27px 23px; border-radius: 23px; }}
            .dashboard-status {{
                flex-direction: column;
                align-items: flex-start;
            }}
            section[data-testid="stSidebar"] {{ min-width: 290px; }}
        }}

        @media(prefers-reduced-motion: reduce) {{
            .metric, .aqua-drop {{ animation:none !important; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------
# UI HELPERS
# ----------------------------------------------------------------
def hero(title: str, subtitle: str) -> None:
    """Render the gradient hero banner used at the top of every page."""
    st.markdown(
        f"""
        <div class="hero">
            <span class="badge">💧 AQUA INTELLIGENCE PLATFORM</span>
            <h1>{title}</h1>
            <p>{subtitle}</p>
            <span class="badge">• Random Forest AI</span>
            <span class="badge">• Live Analytics</span>
            <span class="badge">• Water Sustainability</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str) -> None:
    """Render a section heading."""
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True,
    )


def metric_cards(items: list[tuple[str, str]]) -> None:
    """Render a row of KPI cards from a list of (label, value) pairs."""
    if not items:
        return
    cols = st.columns(len(items), gap="medium")
    for index, (col, (label, value)) in enumerate(zip(cols, items)):
        with col:
            st.markdown(
                f"""
                <div class="metric metric-{index % 5}">
                    <h4>{label}</h4>
                    <h2>{value}</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )


def dashboard_status(record_count: int, r2: float, high_risk_count: int) -> None:
    """Render the small status strip showing record count, model R² and risk."""
    st.markdown(
        f"""
        <div class="dashboard-status">
            <div>
                ◉ <strong>Dashboard ready</strong>
                <span> · {record_count:,} records in view</span>
            </div>
            <div>
                AI model · <strong>Random Forest</strong>
                · R² {r2:.3f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------
# DATA
# ----------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_data(n: int = 5000) -> pd.DataFrame:
    """Generate a synthetic household water-usage dataset.

    Uses a fixed random seed so results are reproducible across reruns
    (important since Streamlit re-executes the script on every interaction).
    """
    rng = np.random.default_rng(RANDOM_SEED)

    dates = pd.date_range(
        start="2024-01-01",
        periods=730,
        freq="D",
    )

    df = pd.DataFrame(
        {
            "date": rng.choice(dates, n),
            "household_id": rng.integers(1, 501, n),
            "city": rng.choice(
                [
                    "Delhi",
                    "Mumbai",
                    "Bengaluru",
                    "Jaipur",
                    "Chennai",
                    "Hyderabad",
                    "Ahmedabad",
                ],
                n,
            ),
            "household_size": rng.integers(2, 8, n),
            "temperature_c": rng.normal(29, 6, n).round(1),
            "rainfall_mm": rng.gamma(2, 4, n).round(1),
            "water_supply_lpd": np.clip(rng.normal(650, 100, n), 350, 900).round(1),
            "previous_consumption_lpd": np.clip(
                rng.normal(500, 100, n), 250, 800
            ).round(1),
            "toilet_flushes": rng.integers(8, 30, n),
            "flush_litres": rng.choice([3, 4, 5, 6, 8, 10], n),
            "greywater_available_lpd": np.clip(
                rng.normal(100, 30, n), 20, 200
            ).round(1),
            "recycled_water_lpd": np.clip(
                rng.normal(70, 25, n), 0, 150
            ).round(1),
            "leakage_lpd": np.clip(rng.exponential(15, n), 0, 100).round(1),
            "awareness_score": rng.integers(30, 101, n),
            "water_stress_index": rng.uniform(0.2, 0.95, n).round(3),
        }
    )

    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month

    season_factor = np.where(
        df["month"].isin([4, 5, 6]),
        1.12,
        np.where(df["month"].isin([7, 8, 9]), 0.92, 1.0),
    )

    base = (
        df["household_size"] * 90
        + df["temperature_c"] * 3
        + df["previous_consumption_lpd"] * 0.35
        + df["toilet_flushes"] * df["flush_litres"] * 0.35
        + df["leakage_lpd"] * 0.5
        - df["rainfall_mm"] * 0.7
        - df["awareness_score"] * 0.35
        - df["recycled_water_lpd"] * 0.25
    )

    noise = rng.normal(0, 35, n)
    df[TARGET] = (base * season_factor + noise).clip(150, 1200).round(2)

    return df.drop(columns=["month"]).sort_values("date").reset_index(drop=True)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive model-ready features (calendar parts, totals, balances) from raw data."""
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().all():
        raise ValueError("No valid dates were found in the dataset.")

    df["date"] = df["date"].fillna(df["date"].median())

    numeric = df.select_dtypes(include=np.number).columns
    if len(numeric):
        df[numeric] = df[numeric].fillna(df[numeric].median())

    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek

    df["toilet_water_lpd"] = df["toilet_flushes"] * df["flush_litres"]
    df["total_available_lpd"] = (
        df["water_supply_lpd"] + df["recycled_water_lpd"]
    )

    if TARGET in df.columns:
        df["water_balance_lpd"] = (
            df["total_available_lpd"] - df[TARGET]
        )

    return df


# ----------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame):
    """Train the Random Forest demand model and return it with evaluation metrics."""
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_SEED,
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    mae = mean_absolute_error(y_test, prediction)
    rmse = np.sqrt(mean_squared_error(y_test, prediction))
    r2 = r2_score(y_test, prediction)

    importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    evaluation = pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "predicted": prediction,
        }
    )

    return model, mae, rmse, r2, importance, evaluation


def add_predictions(df: pd.DataFrame, model: RandomForestRegressor) -> pd.DataFrame:
    """Attach AI predictions, water balance, risk tier and recommendations to df."""
    df = df.copy()

    df["predicted_demand_lpd"] = model.predict(df[FEATURES])

    df["total_available_lpd"] = (
        df["water_supply_lpd"] + df["recycled_water_lpd"]
    )

    df["water_balance_lpd"] = (
        df["total_available_lpd"] - df["predicted_demand_lpd"]
    )

    df["shortage_risk"] = np.select(
        [
            df["water_balance_lpd"] < -100,
            df["water_balance_lpd"] < 0,
        ],
        ["HIGH", "MEDIUM"],
        default="LOW",
    )

    def recommendation(row):
        if row["shortage_risk"] == "HIGH":
            return (
                "Reduce freshwater use, increase recycled-water use, "
                "check leakage and adopt efficient flushing."
            )
        if row["shortage_risk"] == "MEDIUM":
            return (
                "Monitor consumption, reuse greywater and reduce "
                "toilet flush volume."
            )
        return "Maintain efficient consumption and monitor water balance."

    df["recommendation"] = df.apply(recommendation, axis=1)

    df["current_flush_water"] = df["toilet_flushes"] * BASELINE_FLUSH_LITRES
    df["efficient_flush_water"] = df["toilet_flushes"] * EFFICIENT_FLUSH_LITRES
    df["potential_flush_saving_lpd"] = (
        df["current_flush_water"] - df["efficient_flush_water"]
    ).clip(lower=0)
    df["annual_flush_saving_litres"] = (
        df["potential_flush_saving_lpd"] * 365
    )

    return df


@st.cache_data(show_spinner=False)
def build_dataset():
    return prepare_features(generate_data())


@st.cache_resource(show_spinner=False)
def build_ai():
    prepared = build_dataset()

    model, mae, rmse, r2, importance, evaluation = train_model(prepared)
    predictions = add_predictions(prepared, model)

    # Persist artifacts to disk when possible, but the app must never depend
    # on this succeeding (e.g. read-only filesystems, disk quota issues).
    if _DATA_DIR_WRITABLE:
        try:
            predictions.to_csv(PREDICTIONS_FILE, index=False)
            generate_data().to_csv(DATA_FILE, index=False)
            joblib.dump(model, MODEL_FILE)
        except OSError:
            pass

    return predictions, model, mae, rmse, r2, importance, evaluation


# ----------------------------------------------------------------
# ACCOUNTS — SQLITE STORAGE
# ----------------------------------------------------------------
PBKDF2_ITERATIONS = 310_000
MIN_PASSWORD_LENGTH = 8
MAX_FAILED_LOGINS = 5
LOCKOUT_MINUTES = 5
MAX_SCENARIOS = 100
OTHER_CITY = "Other"
FLUSH_OPTIONS = [3, 4, 5, 6, 8, 10]
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]{2,}$")
GENERIC_LOGIN_ERROR = "Incorrect email or password."

# A household's saved water profile. Every key is also a column in `profiles`.
PROFILE_DEFAULTS = {
    "city": "Delhi",
    "household_size": 4,
    "water_supply_lpd": 650,
    "previous_consumption_lpd": 500,
    "toilet_flushes": 18,
    "flush_litres": 6,
    "greywater_available_lpd": 100,
    "recycled_water_lpd": 70,
    "leakage_lpd": 10.0,
    "awareness_score": 70,
    "water_stress_index": 0.5,
    "daily_goal_lpd": 450,
}
PROFILE_LIMITS = {
    "household_size": (2, 12),
    "water_supply_lpd": (200, 1200),
    "previous_consumption_lpd": (150, 1200),
    "toilet_flushes": (1, 40),
    "flush_litres": (3, 10),
    "greywater_available_lpd": (0, 250),
    "recycled_water_lpd": (0, 200),
    "leakage_lpd": (0, 100),
    "awareness_score": (0, 100),
    "water_stress_index": (0, 1),
    "daily_goal_lpd": (100, 2000),
}
PROFILE_INT_FIELDS = {
    "household_size",
    "water_supply_lpd",
    "previous_consumption_lpd",
    "toilet_flushes",
    "flush_litres",
    "greywater_available_lpd",
    "recycled_water_lpd",
    "awareness_score",
    "daily_goal_lpd",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name       TEXT NOT NULL,
    email           TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash   TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    last_login      TEXT,
    failed_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until    TEXT,
    is_active       INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS profiles (
    user_id                  INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    city                     TEXT    NOT NULL,
    household_size           INTEGER NOT NULL,
    water_supply_lpd         REAL    NOT NULL,
    previous_consumption_lpd REAL    NOT NULL,
    toilet_flushes           INTEGER NOT NULL,
    flush_litres             INTEGER NOT NULL,
    greywater_available_lpd  REAL    NOT NULL,
    recycled_water_lpd       REAL    NOT NULL,
    leakage_lpd              REAL    NOT NULL,
    awareness_score          INTEGER NOT NULL,
    water_stress_index       REAL    NOT NULL,
    daily_goal_lpd           REAL    NOT NULL,
    updated_at               TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS scenarios (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name          TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    inputs_json   TEXT NOT NULL,
    predicted_lpd REAL NOT NULL,
    available_lpd REAL NOT NULL,
    balance_lpd   REAL NOT NULL,
    risk          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scenarios_user ON scenarios(user_id, id);

CREATE TABLE IF NOT EXISTS usage_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    log_date   TEXT NOT NULL,
    litres     REAL NOT NULL,
    note       TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (user_id, log_date)
);

-- Administrators are a completely separate identity from users.
CREATE TABLE IF NOT EXISTS admins (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    last_login    TEXT
);

-- Server-side admin sessions; only a hash of the session token is stored.
CREATE TABLE IF NOT EXISTS admin_sessions (
    token_hash TEXT PRIMARY KEY,
    admin_id   INTEGER NOT NULL REFERENCES admins(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    last_seen  TEXT NOT NULL
);

-- Every sign-in attempt (user or admin): outcome is success / failure /
-- blocked / deactivated. user_id is kept as NULL if the user is later deleted.
CREATE TABLE IF NOT EXISTS login_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL,
    kind       TEXT NOT NULL,
    identifier TEXT NOT NULL,
    outcome    TEXT NOT NULL,
    user_id    INTEGER REFERENCES users(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_login_events_lookup ON login_events(kind, identifier, ts);
CREATE INDEX IF NOT EXISTS idx_login_events_ts ON login_events(ts);

-- Which AquaAI pages members open (software-usage analytics).
CREATE TABLE IF NOT EXISTS page_views (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    page    TEXT NOT NULL,
    ts      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_page_views_ts ON page_views(ts);

-- What administrators did (activate / deactivate / delete / unlock).
CREATE TABLE IF NOT EXISTS admin_audit (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             TEXT NOT NULL,
    admin_username TEXT NOT NULL,
    action         TEXT NOT NULL,
    target         TEXT
);
"""


@contextmanager
def db():
    """Open a short-lived connection; commit on success, roll back on error."""
    conn = sqlite3.connect(DB_FILE, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def migrate_db(conn: sqlite3.Connection) -> None:
    """Upgrade an older database in place. Idempotent; never deletes user data."""
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(users)")}
    if "is_active" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1")
    # Keep the security log from growing forever.
    conn.execute(
        "DELETE FROM login_events WHERE ts < ?", (_stamp(_now() - timedelta(days=180)),)
    )


def init_db() -> bool:
    """Create the database and tables if needed. Returns False if unavailable."""
    try:
        DB_FILE.parent.mkdir(parents=True, exist_ok=True)
        with db() as conn:
            conn.executescript(SCHEMA)
            migrate_db(conn)
        return True
    except (sqlite3.Error, OSError):
        return False


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(moment: datetime | None = None) -> str:
    return (moment or _now()).isoformat(timespec="seconds")


# ---- passwords -------------------------------------------------------
def hash_password(password: str) -> str:
    """PBKDF2-HMAC-SHA256 with a per-user random salt."""
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iterations, salt_hex, digest_hex = stored.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            int(iterations),
        )
        return hmac.compare_digest(digest.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


def validate_password(password: str) -> str | None:
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
    if len(password) > 128:
        return "Password must be 128 characters or fewer."
    if not (re.search(r"[A-Za-z]", password) and re.search(r"\d", password)):
        return "Password needs at least one letter and one number."
    return None


def validate_new_account(name: str, email: str, password: str, confirm: str) -> list[str]:
    errors = []
    if not 2 <= len(name.strip()) <= 60:
        errors.append("Enter your name (2–60 characters).")
    if len(email.strip()) > 254 or not EMAIL_RE.match(email.strip()):
        errors.append("Enter a valid email address.")
    password_error = validate_password(password)
    if password_error:
        errors.append(password_error)
    elif password != confirm:
        errors.append("The two passwords do not match.")
    return errors


# ---- profiles --------------------------------------------------------
def sanitize_profile(raw: dict) -> dict:
    """Clamp a profile into the ranges the model was trained on."""
    clean = {}
    for key, default in PROFILE_DEFAULTS.items():
        value = raw.get(key, default)
        if key == "city":
            clean[key] = str(value).strip()[:60] or default
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = float(default)
        if not math.isfinite(number):
            number = float(default)
        low, high = PROFILE_LIMITS[key]
        number = min(max(number, low), high)
        clean[key] = int(round(number)) if key in PROFILE_INT_FIELDS else round(number, 2)
    return clean


# ---- users -----------------------------------------------------------
def create_user(name: str, email: str, password: str, profile: dict) -> tuple[int | None, str | None]:
    """Insert a user and their profile. Returns (user_id, error_message)."""
    clean = sanitize_profile(profile)
    now = _stamp()
    try:
        with db() as conn:
            if conn.execute(
                "SELECT 1 FROM admins WHERE username = ?", (email.strip().lower(),)
            ).fetchone():
                return None, "An account with this email already exists. Try signing in instead."
            cursor = conn.execute(
                "INSERT INTO users (full_name, email, password_hash, created_at, last_login) "
                "VALUES (?, ?, ?, ?, ?)",
                (name.strip(), email.strip().lower(), hash_password(password), now, now),
            )
            user_id = cursor.lastrowid
            columns = ", ".join(PROFILE_DEFAULTS)
            marks = ", ".join("?" for _ in PROFILE_DEFAULTS)
            conn.execute(
                f"INSERT INTO profiles (user_id, {columns}, updated_at) VALUES (?, {marks}, ?)",
                (user_id, *clean.values(), now),
            )
        return user_id, None
    except sqlite3.IntegrityError:
        return None, "An account with this email already exists. Try signing in instead."
    except sqlite3.Error:
        return None, "We couldn't create your account right now. Please try again."


DEACTIVATED_MESSAGE = "This account has been deactivated. Please contact an administrator."


def record_event(
    conn: sqlite3.Connection, kind: str, identifier: str, outcome: str, user_id: int | None = None
) -> None:
    """Append a sign-in attempt to the security log (never stores passwords)."""
    conn.execute(
        "INSERT INTO login_events (ts, kind, identifier, outcome, user_id) VALUES (?, ?, ?, ?, ?)",
        (_stamp(), kind, identifier.strip().lower()[:254], outcome, user_id),
    )


def authenticate(email: str, password: str) -> tuple[int | None, str | None]:
    """Check credentials with a lockout after repeated failures."""
    email = email.strip().lower()
    if len(email) > 254 or len(password) > 1000:
        return None, GENERIC_LOGIN_ERROR
    try:
        with db() as conn:
            row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
            if row is None:
                hash_password(password)  # keep timing similar to a real check
                record_event(conn, "user", email, "failure")
                return None, GENERIC_LOGIN_ERROR

            if row["locked_until"]:
                locked_until = datetime.fromisoformat(row["locked_until"])
                if locked_until > _now():
                    minutes = max(1, math.ceil((locked_until - _now()).total_seconds() / 60))
                    record_event(conn, "user", email, "blocked", row["id"])
                    return None, (
                        f"Too many failed attempts. Try again in {minutes} minute"
                        f"{'s' if minutes != 1 else ''}."
                    )

            if verify_password(password, row["password_hash"]):
                if not row["is_active"]:
                    # Only revealed after the correct password, so this can't
                    # be used to discover which emails are registered.
                    record_event(conn, "user", email, "deactivated", row["id"])
                    return None, DEACTIVATED_MESSAGE
                conn.execute(
                    "UPDATE users SET failed_attempts = 0, locked_until = NULL, last_login = ? "
                    "WHERE id = ?",
                    (_stamp(), row["id"]),
                )
                record_event(conn, "user", email, "success", row["id"])
                return int(row["id"]), None

            attempts = row["failed_attempts"] + 1
            locked_until_value = None
            if attempts >= MAX_FAILED_LOGINS:
                locked_until_value = _stamp(_now() + timedelta(minutes=LOCKOUT_MINUTES))
                attempts = 0
            conn.execute(
                "UPDATE users SET failed_attempts = ?, locked_until = ? WHERE id = ?",
                (attempts, locked_until_value, row["id"]),
            )
            record_event(conn, "user", email, "failure", row["id"])
            return None, GENERIC_LOGIN_ERROR
    except sqlite3.Error:
        return None, "Sign-in is temporarily unavailable. Please try again."


def get_user(user_id: int) -> dict | None:
    columns = ", ".join(f"p.{name}" for name in PROFILE_DEFAULTS)
    with db() as conn:
        row = conn.execute(
            f"SELECT u.id, u.full_name, u.email, u.created_at, u.last_login, u.is_active, {columns} "
            "FROM users u JOIN profiles p ON p.user_id = u.id WHERE u.id = ?",
            (user_id,),
        ).fetchone()
    return dict(row) if row else None


def update_account(user_id: int, name: str, profile: dict) -> None:
    clean = sanitize_profile(profile)
    assignments = ", ".join(f"{name_} = ?" for name_ in PROFILE_DEFAULTS)
    with db() as conn:
        conn.execute("UPDATE users SET full_name = ? WHERE id = ?", (name.strip(), user_id))
        conn.execute(
            f"UPDATE profiles SET {assignments}, updated_at = ? WHERE user_id = ?",
            (*clean.values(), _stamp(), user_id),
        )


def change_password(user_id: int, current: str, new: str) -> str | None:
    """Returns an error message, or None on success."""
    problem = validate_password(new)
    if problem:
        return problem
    with db() as conn:
        row = conn.execute("SELECT password_hash FROM users WHERE id = ?", (user_id,)).fetchone()
        if row is None or not verify_password(current, row["password_hash"]):
            return "Your current password is incorrect."
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?", (hash_password(new), user_id)
        )
    return None


def delete_user(user_id: int, password: str) -> bool:
    with db() as conn:
        row = conn.execute("SELECT password_hash FROM users WHERE id = ?", (user_id,)).fetchone()
        if row is None or not verify_password(password, row["password_hash"]):
            return False
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))  # cascades
    return True


# ---- usage log -------------------------------------------------------
def upsert_usage(user_id: int, log_date: date, litres: float, note: str) -> None:
    with db() as conn:
        conn.execute(
            "INSERT INTO usage_log (user_id, log_date, litres, note, created_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(user_id, log_date) DO UPDATE SET litres = excluded.litres, "
            "note = excluded.note",
            (user_id, log_date.isoformat(), float(litres), note.strip()[:120], _stamp()),
        )


def list_usage(user_id: int) -> pd.DataFrame:
    with db() as conn:
        frame = pd.read_sql_query(
            "SELECT log_date, litres, note FROM usage_log WHERE user_id = ? ORDER BY log_date",
            conn,
            params=(user_id,),
        )
    frame["log_date"] = pd.to_datetime(frame["log_date"])
    return frame


def delete_usage(user_id: int, log_date: str) -> None:
    with db() as conn:
        conn.execute(
            "DELETE FROM usage_log WHERE user_id = ? AND log_date = ?", (user_id, log_date)
        )


# ---- saved scenarios -------------------------------------------------
def save_scenario(
    user_id: int,
    name: str,
    inputs: dict,
    predicted: float,
    available: float,
    balance: float,
    risk: str,
) -> str | None:
    """Returns an error message, or None on success."""
    label = name.strip()[:80] or f"Scenario {datetime.now():%d %b %H:%M}"
    with db() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM scenarios WHERE user_id = ?", (user_id,)
        ).fetchone()[0]
        if count >= MAX_SCENARIOS:
            return f"You can keep up to {MAX_SCENARIOS} scenarios. Delete a few to save more."
        conn.execute(
            "INSERT INTO scenarios (user_id, name, created_at, inputs_json, predicted_lpd, "
            "available_lpd, balance_lpd, risk) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                user_id,
                label,
                _stamp(),
                json.dumps(inputs, default=str),
                float(predicted),
                float(available),
                float(balance),
                risk,
            ),
        )
    return None


def list_scenarios(user_id: int) -> pd.DataFrame:
    with db() as conn:
        return pd.read_sql_query(
            "SELECT id, name, created_at, predicted_lpd, available_lpd, balance_lpd, risk, "
            "inputs_json FROM scenarios WHERE user_id = ? ORDER BY id DESC",
            conn,
            params=(user_id,),
        )


def delete_scenario(user_id: int, scenario_id: int) -> None:
    with db() as conn:
        conn.execute(
            "DELETE FROM scenarios WHERE user_id = ? AND id = ?", (user_id, scenario_id)
        )


# ----------------------------------------------------------------
# ADMINISTRATORS — STORAGE, SESSIONS AND ACCESS CONTROL
# ----------------------------------------------------------------
MIN_ADMIN_PASSWORD_LENGTH = 10
ADMIN_LOCKOUT_MINUTES = 15
ADMIN_SESSION_IDLE_MINUTES = 30
ADMIN_SESSION_MAX_HOURS = 12
ACTIVE_WINDOW_DAYS = 7
GENERIC_ADMIN_ERROR = "Incorrect administrator username or password."


def validate_admin_password(password: str) -> str | None:
    if len(password) < MIN_ADMIN_PASSWORD_LENGTH:
        return f"Admin password must be at least {MIN_ADMIN_PASSWORD_LENGTH} characters."
    if len(password) > 128:
        return "Password must be 128 characters or fewer."
    if not (re.search(r"[A-Za-z]", password) and re.search(r"\d", password)):
        return "Admin password needs at least one letter and one number."
    return None


def _admin_bootstrap_credentials() -> tuple[str, str]:
    """Initial admin credentials from the environment, else Streamlit Secrets."""
    username = os.environ.get("AQUAAI_ADMIN_USERNAME", "").strip()
    password = os.environ.get("AQUAAI_ADMIN_PASSWORD", "")
    if username and password:
        return username, password
    try:
        section = st.secrets["admin"]
        return str(section["username"]).strip(), str(section["password"])
    except Exception:  # no secrets file / no [admin] section
        return "", ""


def bootstrap_admin() -> str:
    """Create the first administrator from env/secrets if none exists yet.

    Returns "ready", "not_configured", "invalid_credentials" or "unavailable".
    Nothing is hardcoded: with no credentials configured, admin login is off.
    """
    try:
        with db() as conn:
            if conn.execute("SELECT COUNT(*) FROM admins").fetchone()[0]:
                return "ready"
            username, password = _admin_bootstrap_credentials()
            if not username or not password:
                return "not_configured"
            if not 3 <= len(username) <= 254 or validate_admin_password(password):
                return "invalid_credentials"
            conn.execute(
                "INSERT INTO admins (username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, hash_password(password), _stamp()),
            )
            return "ready"
    except sqlite3.IntegrityError:  # another session created it first
        return "ready"
    except sqlite3.Error:
        return "unavailable"


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _admin_row_from_token(conn: sqlite3.Connection, token) -> sqlite3.Row | None:
    """Look up a live admin session (and refresh its idle timer)."""
    if not token or not isinstance(token, str):
        return None
    digest = _hash_token(token)
    row = conn.execute(
        "SELECT s.admin_id, s.created_at AS session_created, s.last_seen, "
        "a.username, a.created_at, a.last_login "
        "FROM admin_sessions s JOIN admins a ON a.id = s.admin_id WHERE s.token_hash = ?",
        (digest,),
    ).fetchone()
    if row is None:
        return None
    now = _now()
    idle = now - datetime.fromisoformat(row["last_seen"])
    age = now - datetime.fromisoformat(row["session_created"])
    if idle > timedelta(minutes=ADMIN_SESSION_IDLE_MINUTES) or age > timedelta(
        hours=ADMIN_SESSION_MAX_HOURS
    ):
        conn.execute("DELETE FROM admin_sessions WHERE token_hash = ?", (digest,))
        return None
    conn.execute("UPDATE admin_sessions SET last_seen = ? WHERE token_hash = ?", (_stamp(), digest))
    return row


def _require_admin(conn: sqlite3.Connection, token) -> int:
    """Server-side role check used by EVERY admin action. Returns the admin id."""
    row = _admin_row_from_token(conn, token)
    if row is None:
        raise PermissionError("An administrator session is required.")
    return int(row["admin_id"])


def get_admin_session(token) -> dict | None:
    with db() as conn:
        row = _admin_row_from_token(conn, token)
    if row is None:
        return None
    return {
        "id": int(row["admin_id"]),
        "username": row["username"],
        "created_at": row["created_at"],
        "last_login": row["last_login"],
    }


def _admin_lock_remaining(conn: sqlite3.Connection, identifier: str) -> int:
    """Minutes left on an admin lockout (0 = not locked).

    Counted from the security log, so it also throttles guesses against
    usernames that don't exist. A success resets the count.
    """
    window_start = _stamp(_now() - timedelta(minutes=ADMIN_LOCKOUT_MINUTES))
    # Only failures AFTER the most recent success count (ordered by event id,
    # which is exact, unlike second-resolution timestamps).
    last_ok_id = conn.execute(
        "SELECT COALESCE(MAX(id), 0) FROM login_events WHERE kind = 'admin' "
        "AND identifier = ? AND outcome = 'success'",
        (identifier,),
    ).fetchone()[0]
    times = [
        r[0]
        for r in conn.execute(
            "SELECT ts FROM login_events WHERE kind = 'admin' AND identifier = ? "
            "AND outcome = 'failure' AND id > ? AND ts >= ? ORDER BY id",
            (identifier, last_ok_id, window_start),
        )
    ]
    if len(times) < MAX_FAILED_LOGINS:
        return 0
    frees_at = datetime.fromisoformat(times[len(times) - MAX_FAILED_LOGINS]) + timedelta(
        minutes=ADMIN_LOCKOUT_MINUTES
    )
    return max(1, math.ceil((frees_at - _now()).total_seconds() / 60))


def admin_authenticate(identifier: str, password: str) -> tuple[str | None, str | None]:
    """Admin sign-in. Returns (session_token, error). Never touches `users`."""
    identifier = identifier.strip().lower()
    if not identifier or len(identifier) > 254 or len(password) > 1000:
        return None, GENERIC_ADMIN_ERROR
    try:
        with db() as conn:
            minutes = _admin_lock_remaining(conn, identifier)
            if minutes:
                record_event(conn, "admin", identifier, "blocked")
                return None, (
                    f"Too many failed attempts. Try again in {minutes} minute"
                    f"{'s' if minutes != 1 else ''}."
                )
            row = conn.execute(
                "SELECT id, password_hash FROM admins WHERE username = ?", (identifier,)
            ).fetchone()
            if row is None:
                hash_password(password)  # similar timing for unknown usernames
                valid = False
            else:
                valid = verify_password(password, row["password_hash"])
            if not valid:
                record_event(conn, "admin", identifier, "failure")
                return None, GENERIC_ADMIN_ERROR

            token = secrets.token_urlsafe(32)
            now = _stamp()
            conn.execute(
                "DELETE FROM admin_sessions WHERE last_seen < ?",
                (_stamp(_now() - timedelta(hours=ADMIN_SESSION_MAX_HOURS)),),
            )
            conn.execute(
                "INSERT INTO admin_sessions (token_hash, admin_id, created_at, last_seen) "
                "VALUES (?, ?, ?, ?)",
                (_hash_token(token), row["id"], now, now),
            )
            conn.execute("UPDATE admins SET last_login = ? WHERE id = ?", (now, row["id"]))
            record_event(conn, "admin", identifier, "success")
            return token, None
    except sqlite3.Error:
        return None, "Sign-in is temporarily unavailable. Please try again."


def admin_logout(token) -> None:
    if not token or not isinstance(token, str):
        return
    try:
        with db() as conn:
            conn.execute("DELETE FROM admin_sessions WHERE token_hash = ?", (_hash_token(token),))
    except sqlite3.Error:
        pass


def _audit(conn: sqlite3.Connection, admin_id: int, action: str, target: str) -> None:
    username = conn.execute("SELECT username FROM admins WHERE id = ?", (admin_id,)).fetchone()[0]
    conn.execute(
        "INSERT INTO admin_audit (ts, admin_username, action, target) VALUES (?, ?, ?, ?)",
        (_stamp(), username, action, target),
    )


def admin_change_password(token, current: str, new: str) -> str | None:
    """Change the signed-in admin's password. Returns an error message or None."""
    with db() as conn:
        admin_id = _require_admin(conn, token)
        row = conn.execute("SELECT password_hash FROM admins WHERE id = ?", (admin_id,)).fetchone()
        if not verify_password(current, row["password_hash"]):
            return "Your current password is incorrect."
        problem = validate_admin_password(new)
        if problem:
            return problem
        if verify_password(new, row["password_hash"]):
            return "Choose a password different from the current one."
        conn.execute("UPDATE admins SET password_hash = ? WHERE id = ?", (hash_password(new), admin_id))
        # Sign out every other session of this administrator.
        conn.execute(
            "DELETE FROM admin_sessions WHERE admin_id = ? AND token_hash != ?",
            (admin_id, _hash_token(token)),
        )
        _audit(conn, admin_id, "changed own password", "")
    return None


def _daily_counts(rows, days: int) -> pd.Series:
    """Per-day counts for the last `days` days (zeros filled in)."""
    today = _now().date()
    labels = [(today - timedelta(days=i)).isoformat() for i in range(days - 1, -1, -1)]
    lookup = {day: count for day, count in rows}
    return pd.Series([lookup.get(day, 0) for day in labels], index=pd.to_datetime(labels))


def admin_overview(token) -> dict:
    with db() as conn:
        _require_admin(conn, token)
        active_since = _stamp(_now() - timedelta(days=ACTIVE_WINDOW_DAYS))
        day_ago = _stamp(_now() - timedelta(hours=24))
        one = lambda sql, *args: conn.execute(sql, args).fetchone()[0]  # noqa: E731
        return {
            "total": one("SELECT COUNT(*) FROM users"),
            "active": one(
                "SELECT COUNT(*) FROM users WHERE is_active = 1 AND last_login >= ?", active_since
            ),
            "deactivated": one("SELECT COUNT(*) FROM users WHERE is_active = 0"),
            "new_week": one("SELECT COUNT(*) FROM users WHERE created_at >= ?", active_since),
            "failed_24h": one(
                "SELECT COUNT(*) FROM login_events WHERE kind = 'user' AND outcome != 'success' "
                "AND ts >= ?",
                day_ago,
            ),
            "signups": _daily_counts(
                conn.execute(
                    "SELECT substr(created_at, 1, 10), COUNT(*) FROM users WHERE created_at >= ? "
                    "GROUP BY 1",
                    (_stamp(_now() - timedelta(days=30)),),
                ),
                30,
            ).rename("Registrations"),
            "logins": _daily_counts(
                conn.execute(
                    "SELECT substr(ts, 1, 10), COUNT(*) FROM login_events WHERE kind = 'user' "
                    "AND outcome = 'success' AND ts >= ? GROUP BY 1",
                    (_stamp(_now() - timedelta(days=30)),),
                ),
                30,
            ).rename("Logins"),
        }


def admin_list_users(token, query: str = "", status: str = "All") -> pd.DataFrame:
    sql = (
        "SELECT u.id, u.full_name, u.email, p.city, p.household_size, u.created_at, "
        "u.last_login, u.is_active, u.locked_until, "
        "(SELECT COUNT(*) FROM usage_log l WHERE l.user_id = u.id) AS usage_entries, "
        "(SELECT COUNT(*) FROM scenarios s WHERE s.user_id = u.id) AS scenarios "
        "FROM users u LEFT JOIN profiles p ON p.user_id = u.id WHERE 1 = 1"
    )
    params: list = []
    text = query.strip()[:100]
    if text:
        escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        sql += " AND (u.full_name LIKE ? ESCAPE '\\' OR u.email LIKE ? ESCAPE '\\')"
        params += [f"%{escaped}%", f"%{escaped}%"]
    if status == "Active":
        sql += " AND u.is_active = 1"
    elif status == "Deactivated":
        sql += " AND u.is_active = 0"
    sql += " ORDER BY u.id DESC LIMIT 500"
    with db() as conn:
        _require_admin(conn, token)
        return pd.read_sql_query(sql, conn, params=params)


def admin_set_user_active(token, user_id: int, active: bool) -> bool:
    with db() as conn:
        admin_id = _require_admin(conn, token)
        row = conn.execute("SELECT email FROM users WHERE id = ?", (user_id,)).fetchone()
        if row is None:
            return False
        conn.execute("UPDATE users SET is_active = ? WHERE id = ?", (1 if active else 0, user_id))
        _audit(conn, admin_id, "activated user" if active else "deactivated user", row["email"])
    return True


def admin_unlock_user(token, user_id: int) -> bool:
    with db() as conn:
        admin_id = _require_admin(conn, token)
        row = conn.execute("SELECT email FROM users WHERE id = ?", (user_id,)).fetchone()
        if row is None:
            return False
        conn.execute(
            "UPDATE users SET failed_attempts = 0, locked_until = NULL WHERE id = ?", (user_id,)
        )
        _audit(conn, admin_id, "unlocked user", row["email"])
    return True


def admin_delete_user(token, user_id: int) -> bool:
    with db() as conn:
        admin_id = _require_admin(conn, token)
        row = conn.execute("SELECT email FROM users WHERE id = ?", (user_id,)).fetchone()
        if row is None:
            return False
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))  # cascades
        _audit(conn, admin_id, "deleted user", row["email"])
    return True


def admin_usage(token) -> dict:
    with db() as conn:
        _require_admin(conn, token)
        since = _stamp(_now() - timedelta(days=30))
        one = lambda sql, *args: conn.execute(sql, args).fetchone()[0]  # noqa: E731
        by_page = pd.read_sql_query(
            "SELECT page AS Page, COUNT(*) AS Views FROM page_views WHERE ts >= ? "
            "GROUP BY page ORDER BY Views DESC",
            conn,
            params=(since,),
        )
        top_users = pd.read_sql_query(
            "SELECT u.full_name AS Name, u.email AS Email, "
            "(SELECT COUNT(*) FROM page_views v WHERE v.user_id = u.id AND v.ts >= ?) AS PageViews, "
            "(SELECT COUNT(*) FROM usage_log l WHERE l.user_id = u.id) AS UsageEntries, "
            "(SELECT COUNT(*) FROM scenarios s WHERE s.user_id = u.id) AS Scenarios "
            "FROM users u ORDER BY PageViews DESC, UsageEntries DESC LIMIT 10",
            conn,
            params=(since,),
        )
        top_users.columns = ["Name", "Email", "Page views (30d)", "Usage entries", "Scenarios"]
        cities = pd.read_sql_query(
            "SELECT city AS City, COUNT(*) AS Users FROM profiles GROUP BY city ORDER BY Users DESC",
            conn,
        )
        return {
            "views_30d": one("SELECT COUNT(*) FROM page_views WHERE ts >= ?", since),
            "active_30d": one("SELECT COUNT(DISTINCT user_id) FROM page_views WHERE ts >= ?", since),
            "usage_entries": one("SELECT COUNT(*) FROM usage_log"),
            "scenarios": one("SELECT COUNT(*) FROM scenarios"),
            "users_total": one("SELECT COUNT(*) FROM users"),
            "users_logging": one("SELECT COUNT(DISTINCT user_id) FROM usage_log"),
            "users_scenarios": one("SELECT COUNT(DISTINCT user_id) FROM scenarios"),
            "by_page": by_page,
            "daily_views": _daily_counts(
                conn.execute(
                    "SELECT substr(ts, 1, 10), COUNT(*) FROM page_views WHERE ts >= ? GROUP BY 1",
                    (since,),
                ),
                30,
            ).rename("Page views"),
            "daily_users": _daily_counts(
                conn.execute(
                    "SELECT substr(ts, 1, 10), COUNT(DISTINCT user_id) FROM page_views "
                    "WHERE ts >= ? GROUP BY 1",
                    (since,),
                ),
                30,
            ).rename("Active users"),
            "top_users": top_users,
            "cities": cities,
        }


def admin_security(token) -> dict:
    with db() as conn:
        _require_admin(conn, token)
        day_ago = _stamp(_now() - timedelta(hours=24))
        week_ago = _stamp(_now() - timedelta(days=7))
        one = lambda sql, *args: conn.execute(sql, args).fetchone()[0]  # noqa: E731

        def per_day(condition: str) -> pd.Series:
            return _daily_counts(
                conn.execute(
                    f"SELECT substr(ts, 1, 10), COUNT(*) FROM login_events WHERE ts >= ? "
                    f"AND {condition} GROUP BY 1",
                    (week_ago,),
                ),
                7,
            )

        return {
            "success_24h": one(
                "SELECT COUNT(*) FROM login_events WHERE kind = 'user' AND outcome = 'success' "
                "AND ts >= ?",
                day_ago,
            ),
            "failed_24h": one(
                "SELECT COUNT(*) FROM login_events WHERE kind = 'user' AND outcome != 'success' "
                "AND ts >= ?",
                day_ago,
            ),
            "admin_failed_24h": one(
                "SELECT COUNT(*) FROM login_events WHERE kind = 'admin' AND outcome != 'success' "
                "AND ts >= ?",
                day_ago,
            ),
            "locked": one(
                "SELECT COUNT(*) FROM users WHERE locked_until IS NOT NULL AND locked_until > ?",
                _stamp(),
            ),
            "daily": pd.DataFrame(
                {
                    "Successful": per_day("outcome = 'success'"),
                    "Failed": per_day("outcome != 'success'"),
                }
            ),
            "events": pd.read_sql_query(
                "SELECT ts, kind, identifier, outcome FROM login_events ORDER BY id DESC LIMIT 200",
                conn,
            ),
            "locked_users": pd.read_sql_query(
                "SELECT id, full_name, email, locked_until FROM users "
                "WHERE locked_until IS NOT NULL AND locked_until > ?",
                conn,
                params=(_stamp(),),
            ),
            "top_failing": pd.read_sql_query(
                "SELECT identifier AS Identifier, kind AS Type, COUNT(*) AS Failures "
                "FROM login_events WHERE ts >= ? AND outcome != 'success' "
                "GROUP BY kind, identifier ORDER BY Failures DESC LIMIT 10",
                conn,
                params=(day_ago,),
            ),
            "audit": pd.read_sql_query(
                "SELECT ts, admin_username, action, target FROM admin_audit "
                "ORDER BY id DESC LIMIT 50",
                conn,
            ),
        }


# ---- member activity (feeds the admin usage analytics) ---------------
def log_page_view(user_id: int, page: str) -> None:
    try:
        with db() as conn:
            conn.execute(
                "INSERT INTO page_views (user_id, page, ts) VALUES (?, ?, ?)",
                (user_id, page, _stamp()),
            )
    except sqlite3.Error:
        pass


DB_READY = init_db()
ADMIN_STATUS = bootstrap_admin() if DB_READY else "unavailable"


# ----------------------------------------------------------------
# LOAD
# ----------------------------------------------------------------
with st.spinner("Initializing AquaAI AI engine..."):
    df, model, mae, rmse, r2, importance, evaluation = build_ai()

inject_css(st.session_state.dark_mode)


# ----------------------------------------------------------------
# ACCOUNTS — STYLES
# ----------------------------------------------------------------
ACCOUNT_CSS = """
<style>
/* forms become glass cards */
div[data-testid="stForm"] {
    border-radius: 24px;
    border: 1px solid @EDGE@;
    background: @CARD@;
    padding: 26px;
    box-shadow: @SHADOW@;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
}
.stFormSubmitButton > button {
    border: 0;
    border-radius: 15px;
    min-height: 48px;
    font-weight: 700;
    color: white;
    background: linear-gradient(135deg,#075985,#12b5cb);
    box-shadow: 0 9px 22px rgba(7,89,133,.20);
    transition: all .25s ease;
}
.stFormSubmitButton > button:hover {
    transform: translateY(-2px);
    filter: brightness(1.06);
    box-shadow: 0 14px 28px rgba(7,89,133,.27);
}
.stFormSubmitButton > button p { color: white; }
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div { border-radius: 14px !important; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    font-weight: 700;
    padding: 10px 16px;
    border-radius: 12px 12px 0 0;
}
@DARK_LABELS@

/* sign-in / register promo panel */
.auth-promo {
    position: relative;
    overflow: hidden;
    min-height: 480px;
    padding: 34px 32px;
    border-radius: 28px;
    color: white;
    background: linear-gradient(150deg,#033047,#0077b6 60%,#12b5cb);
    box-shadow: 0 26px 60px rgba(0,92,135,.25);
}
.auth-promo::after {
    content: "";
    position: absolute;
    width: 280px; height: 280px;
    right: -90px; bottom: -110px;
    border-radius: 50%;
    background: rgba(255,255,255,.10);
}
.auth-promo h2 {
    position: relative; z-index: 2;
    margin: 12px 0 8px;
    color: white;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 30px; font-weight: 800;
    line-height: 1.15; letter-spacing: -.03em;
}
.auth-promo p {
    position: relative; z-index: 2;
    margin: 0 0 18px;
    font-size: 15px; line-height: 1.6;
    color: rgba(255,255,255,.88);
}
.auth-feature {
    position: relative; z-index: 2;
    display: flex; align-items: flex-start; gap: 14px;
    margin: 14px 0;
}
.auth-feature .ico {
    flex: 0 0 40px; height: 40px;
    display: grid; place-items: center;
    border-radius: 12px; font-size: 20px;
    background: rgba(255,255,255,.16);
    border: 1px solid rgba(255,255,255,.25);
}
.auth-feature b { display: block; font-size: 15px; color: white; }
.auth-feature span { font-size: 13px; line-height: 1.45; color: rgba(255,255,255,.82); }
.auth-note {
    position: relative; z-index: 2;
    margin-top: 22px; padding: 12px 14px;
    border-radius: 14px; font-size: 13px;
    color: rgba(255,255,255,.92);
    background: rgba(255,255,255,.12);
    border: 1px solid rgba(255,255,255,.20);
}

/* personalised dashboard */
.chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin: -6px 0 18px; }
.chip {
    padding: 7px 13px;
    border-radius: 999px;
    font-size: 13px; font-weight: 700;
    color: @INK@;
    background: @CARD@;
    border: 1px solid @EDGE@;
    box-shadow: 0 6px 16px rgba(5,72,105,.06);
}
.goal-card {
    margin: 22px 0 4px; padding: 20px 24px;
    border-radius: 22px; color: @INK@;
    background: @CARD@;
    border: 1px solid @EDGE@;
    box-shadow: @SHADOW@;
}
.goal-head {
    display: flex; flex-wrap: wrap; justify-content: space-between; gap: 12px;
    font-size: 15px; font-weight: 700;
}
.goal-track {
    height: 14px; margin: 14px 0 10px;
    border-radius: 999px; overflow: hidden;
    background: rgba(16,42,67,.12);
}
.goal-fill {
    height: 100%; border-radius: 999px;
    background: linear-gradient(90deg,#17865e,#4cc08e);
}
.goal-warn .goal-fill { background: linear-gradient(90deg,#bd6c17,#efa947); }
.goal-bad .goal-fill { background: linear-gradient(90deg,#ad3945,#df626c); }
.goal-msg { font-size: 13px; color: @MUTED@; }
.insight {
    margin: 0 0 10px; padding: 14px 18px;
    border-radius: 16px;
    font-size: 14.5px; line-height: 1.5;
    color: @INK@;
    background: @CARD@;
    border: 1px solid @EDGE@;
    border-left: 4px solid #12b5cb;
}
.empty-state {
    padding: 34px 20px; text-align: center;
    border-radius: 22px; color: @MUTED@;
    background: @CARD@;
    border: 1.5px dashed rgba(18,181,203,.55);
}
.empty-state b { display: block; margin-bottom: 4px; font-size: 17px; color: @INK@; }

/* sidebar account card + active navigation item */
.acct-card {
    display: flex; align-items: center; gap: 12px;
    margin: 0 0 12px; padding: 13px 14px;
    border-radius: 16px;
    background: rgba(255,255,255,.08);
    border: 1px solid rgba(255,255,255,.14);
}
.acct-avatar {
    flex: 0 0 42px; width: 42px; height: 42px;
    display: grid; place-items: center;
    border-radius: 50%;
    font-size: 18px; font-weight: 800; color: white;
    background: linear-gradient(135deg,#12b5cb,#6847a5);
}
.acct-name { font-size: 15px; font-weight: 800; line-height: 1.2; color: white; }
.acct-sub { margin-top: 2px; font-size: 12px; color: #9fd8e6; }
section[data-testid="stSidebar"] .st-key-acct_signout button,
section[data-testid="stSidebar"] .st-key-admin_signout button {
    min-height: 44px !important;
    font-size: 14px !important;
    justify-content: center !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"],
section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg,rgba(35,194,224,.38),rgba(70,110,220,.22)) !important;
    border-color: rgba(165,240,255,.68) !important;
    box-shadow:
        0 15px 34px rgba(0,65,95,.34),
        0 0 24px rgba(40,210,240,.18),
        inset 0 1px 0 rgba(255,255,255,.36) !important;
}
</style>
"""


def inject_account_css(dark: bool = False) -> None:
    """Styles for the account screens (sign-in, dashboard, sidebar card)."""
    palette = {
        "@INK@": "#e9fbff" if dark else "#102a43",
        "@MUTED@": "#9cc7d6" if dark else "#58798c",
        "@CARD@": "rgba(255,255,255,.07)" if dark else "rgba(255,255,255,.72)",
        "@EDGE@": "rgba(255,255,255,.14)" if dark else "rgba(255,255,255,.90)",
        "@SHADOW@": "0 18px 44px rgba(0,0,0,.28)" if dark else "0 18px 44px rgba(5,72,105,.10)",
        "@DARK_LABELS@": (
            "[data-testid='stWidgetLabel'] p, .stTabs button p { color: #e9fbff !important; }"
            if dark
            else ""
        ),
    }
    css = ACCOUNT_CSS
    for token, value in palette.items():
        css = css.replace(token, value)
    st.markdown(css, unsafe_allow_html=True)


inject_account_css(st.session_state.dark_mode)


# ----------------------------------------------------------------
# ACCOUNTS — AI HELPERS (personalised predictions)
# ----------------------------------------------------------------
def esc(text) -> str:
    """HTML-escape anything user-supplied before it goes into unsafe HTML."""
    return html.escape(str(text), quote=True)


def classify_risk(balance: float) -> str:
    return "HIGH" if balance < -100 else "MEDIUM" if balance < 0 else "LOW"


def city_temperature(city: str) -> float:
    subset = df.loc[df["city"] == city, "temperature_c"]
    return float(subset.mean() if len(subset) else df["temperature_c"].mean())


def city_average_demand(city: str) -> float:
    subset = df.loc[df["city"] == city, "predicted_demand_lpd"]
    return float(subset.mean() if len(subset) else df["predicted_demand_lpd"].mean())


def predict_many(profiles: list[dict]) -> np.ndarray:
    """Predicted daily demand for each profile in each of the 12 months.

    Returns an array shaped (len(profiles), 12). Predicting all rows in a
    single call keeps the random forest fast.
    """
    rainfall = float(df["rainfall_mm"].mean())
    weekday = date.today().weekday()
    temperatures: dict[str, float] = {}
    rows = []
    for p in profiles:
        if p["city"] not in temperatures:
            temperatures[p["city"]] = city_temperature(p["city"])
        for month in range(1, 13):
            rows.append(
                {
                    "household_size": p["household_size"],
                    "temperature_c": temperatures[p["city"]],
                    "rainfall_mm": rainfall,
                    "water_supply_lpd": p["water_supply_lpd"],
                    "previous_consumption_lpd": p["previous_consumption_lpd"],
                    "toilet_flushes": p["toilet_flushes"],
                    "flush_litres": p["flush_litres"],
                    "greywater_available_lpd": p["greywater_available_lpd"],
                    "recycled_water_lpd": p["recycled_water_lpd"],
                    "leakage_lpd": p["leakage_lpd"],
                    "awareness_score": p["awareness_score"],
                    "water_stress_index": p["water_stress_index"],
                    "month": month,
                    "day_of_week": weekday,
                }
            )
    frame = pd.DataFrame(rows)[FEATURES]
    return model.predict(frame).reshape(len(profiles), 12)


def build_savings_plan(profile: dict) -> tuple[pd.DataFrame, dict, list[str]]:
    """Ask the model what each water-saving action would change for this home.

    Returns (per-action table, combined effect, actions already in place).
    Effects are year-average figures so seasonal swings don't mask them.
    """
    levers = [
        ("🔧 Fix leaks (down to 5 L/day)", {"leakage_lpd": min(profile["leakage_lpd"], 5.0)}),
        ("🚽 Switch to a 4-litre flush", {"flush_litres": min(profile["flush_litres"], 4)}),
        (
            "♻️ Add 40 L/day of recycled water",
            {"recycled_water_lpd": min(profile["recycled_water_lpd"] + 40, 200)},
        ),
        (
            "🧠 Raise awareness score by 20",
            {"awareness_score": min(profile["awareness_score"] + 20, 100)},
        ),
    ]
    active = [(l, c) for l, c in levers if any(profile[k] != v for k, v in c.items())]
    already = [l for l, c in levers if not any(profile[k] != v for k, v in c.items())]

    combined = dict(profile)
    for _, change in active:
        combined.update(change)

    scenarios = [profile] + [{**profile, **c} for _, c in active] + [combined]
    demand = predict_many(scenarios).mean(axis=1)
    base_balance = profile["water_supply_lpd"] + profile["recycled_water_lpd"] - demand[0]

    def effect(scenario: dict, value: float) -> tuple[float, float]:
        balance = scenario["water_supply_lpd"] + scenario["recycled_water_lpd"] - value
        return value - demand[0], balance - base_balance

    rows = []
    for (label, change), value in zip(active, demand[1:-1]):
        d_demand, d_balance = effect({**profile, **change}, value)
        rows.append(
            {
                "Action": label,
                "Demand change (L/day)": d_demand,
                "Balance gain (L/day)": d_balance,
                "Annual gain (L/year)": d_balance * 365,
            }
        )
    c_demand, c_balance = effect(combined, demand[-1])
    summary = {"demand_change": c_demand, "balance_gain": c_balance, "annual": c_balance * 365}
    return pd.DataFrame(rows), summary, already


# ----------------------------------------------------------------
# ACCOUNTS — SESSION HELPERS
# ----------------------------------------------------------------
def load_active_user() -> dict | None:
    """The signed-in user (fresh from the database), or None for guests."""
    user_id = st.session_state.get("user_id")
    if not user_id or not DB_READY:
        return None
    try:
        user = get_user(user_id)
    except sqlite3.Error:
        user = None
    if user is not None and not user.get("is_active", 1):
        # An administrator deactivated this account: end the session at once.
        st.session_state.auth_notice = DEACTIVATED_MESSAGE
        user = None
    if user is None:
        st.session_state.user_id = None
    return user


def load_active_admin() -> dict | None:
    """The signed-in administrator, validated against the database on every run."""
    token = st.session_state.get("admin_token")
    if not token or not DB_READY:
        return None
    try:
        admin = get_admin_session(token)
    except sqlite3.Error:
        admin = None
    if admin is None:
        st.session_state.admin_token = None
        st.session_state.auth_notice = "Your administrator session has ended. Please sign in again."
    return admin


def sign_in_user(user_id: int, notice: str) -> None:
    admin_logout(st.session_state.get("admin_token"))
    st.session_state.admin_token = None
    st.session_state.user_id = user_id
    st.session_state.aqua_navigation = "🏠 Overview"
    st.session_state.last_logged_page = None
    st.session_state.auth_notice = notice
    st.rerun()


def sign_out_user() -> None:
    st.session_state.user_id = None
    st.session_state.aqua_navigation = "🏠 Overview"
    st.session_state.last_logged_page = None
    st.session_state.auth_notice = "You've been signed out."
    st.rerun()


def sign_in_admin(token: str) -> None:
    st.session_state.user_id = None  # one identity at a time
    st.session_state.admin_token = token
    st.session_state.admin_navigation = "📊 Admin Overview"
    st.session_state.auth_notice = "Signed in as administrator."
    st.rerun()


def sign_out_admin() -> None:
    admin_logout(st.session_state.get("admin_token"))
    st.session_state.admin_token = None
    st.session_state.admin_navigation = "📊 Admin Overview"
    st.session_state.auth_notice = "Administrator signed out."
    st.rerun()


active_user = load_active_user()
is_signed_in = active_user is not None


# ----------------------------------------------------------------
# ACCOUNTS — UI
# ----------------------------------------------------------------
def profile_inputs(prefix: str, values: dict, compact: bool = False) -> dict:
    """Render the household-profile fields; returns the entered values.

    Used by both registration (compact: details tucked in an expander) and the
    account page. Widget keys are namespaced by `prefix`.
    """
    cities = sorted(df["city"].unique().tolist()) + [OTHER_CITY]
    city = values["city"] if values["city"] in cities else OTHER_CITY
    out: dict = {}

    c1, c2, c3 = st.columns(3)
    out["city"] = c1.selectbox("City", cities, index=cities.index(city), key=f"{prefix}_city")
    out["household_size"] = c2.number_input(
        "Household size", min_value=2, max_value=12, value=int(values["household_size"]),
        step=1, key=f"{prefix}_size",
    )
    out["daily_goal_lpd"] = c3.number_input(
        "Daily goal (L/day)", min_value=100, max_value=2000, value=int(values["daily_goal_lpd"]),
        step=10, key=f"{prefix}_goal",
    )

    flush = int(values["flush_litres"])
    with st.expander("🚿 Home water profile", expanded=not compact):
        left, right = st.columns(2)
        with left:
            out["water_supply_lpd"] = st.slider(
                "Water supply (L/day)", 200, 1200, int(values["water_supply_lpd"]),
                key=f"{prefix}_supply",
            )
            out["previous_consumption_lpd"] = st.slider(
                "Typical past consumption (L/day)", 150, 1200,
                int(values["previous_consumption_lpd"]), key=f"{prefix}_previous",
            )
            out["toilet_flushes"] = st.slider(
                "Toilet flushes per day", 1, 40, int(values["toilet_flushes"]),
                key=f"{prefix}_flushes",
            )
            out["flush_litres"] = st.select_slider(
                "Flush volume (litres)", options=FLUSH_OPTIONS,
                value=flush if flush in FLUSH_OPTIONS else 6, key=f"{prefix}_flush",
            )
            out["leakage_lpd"] = st.slider(
                "Estimated leakage (L/day)", 0.0, 100.0, float(values["leakage_lpd"]),
                step=0.5, key=f"{prefix}_leak",
            )
        with right:
            out["recycled_water_lpd"] = st.slider(
                "Recycled water (L/day)", 0, 200, int(values["recycled_water_lpd"]),
                key=f"{prefix}_recycled",
            )
            out["greywater_available_lpd"] = st.slider(
                "Greywater available (L/day)", 0, 250, int(values["greywater_available_lpd"]),
                key=f"{prefix}_grey",
            )
            out["awareness_score"] = st.slider(
                "Water-use awareness (0–100)", 0, 100, int(values["awareness_score"]),
                key=f"{prefix}_awareness",
            )
            out["water_stress_index"] = st.slider(
                "Local water stress (0–1)", 0.0, 1.0, float(values["water_stress_index"]),
                step=0.05, key=f"{prefix}_stress",
            )
    return out


def render_auth_panel() -> None:
    """Sign-in / register screen shown to guests on account pages."""
    left, right = st.columns([5, 6], gap="large")

    with left:
        st.markdown(
            """
            <div class="auth-promo">
                <span class="badge">💧 FREE ACCOUNT</span>
                <h2>Water insights built around your home</h2>
                <p>Save your household profile once and AquaAI keeps your numbers,
                goals and savings ideas ready every time you come back.</p>
                <div class="auth-feature"><div class="ico">📊</div>
                    <div><b>Personal dashboard</b>
                    <span>Your predicted demand, shortage risk and how you compare with your city.</span></div></div>
                <div class="auth-feature"><div class="ico">📝</div>
                    <div><b>Daily usage log</b>
                    <span>Track real consumption against a daily goal you set.</span></div></div>
                <div class="auth-feature"><div class="ico">💡</div>
                    <div><b>AI savings plan</b>
                    <span>See which fixes would save you the most water, in litres per year.</span></div></div>
                <div class="auth-feature"><div class="ico">💾</div>
                    <div><b>Saved scenarios</b>
                    <span>Keep your Household Simulator experiments and revisit them.</span></div></div>
                <div class="auth-note">🔒 Your account unlocks the full AquaAI dashboard: predictions, forecasts, risk monitoring, reports and more.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        if not DB_READY:
            st.error(
                "Account storage isn't available on this server right now. "
                "You can keep exploring AquaAI as a guest."
            )
            return

        tab_in, tab_up, tab_admin = st.tabs(["🔐 Sign in", "✨ Create account", "🛡️ Admin"])

        with tab_in:
            with st.form("aqua_login_form"):
                email = st.text_input("Email", placeholder="you@example.com", key="login_email")
                password = st.text_input("Password", type="password", key="login_password")
                submitted = st.form_submit_button("Sign in", use_container_width=True)
            if submitted:
                if not email.strip() or not password:
                    st.error("Enter your email and password.")
                else:
                    user_id, error = authenticate(email, password)
                    if error:
                        st.error(error)
                    else:
                        sign_in_user(user_id, "Welcome back!")

        with tab_up:
            with st.form("aqua_register_form"):
                name = st.text_input("Full name", max_chars=60, key="reg_name")
                email = st.text_input("Email", placeholder="you@example.com", key="reg_email")
                p1, p2 = st.columns(2)
                password = p1.text_input(
                    "Password", type="password", key="reg_password",
                    help=f"At least {MIN_PASSWORD_LENGTH} characters, with a letter and a number.",
                )
                confirm = p2.text_input("Confirm password", type="password", key="reg_confirm")
                st.markdown("**Your household**")
                profile = profile_inputs("reg", PROFILE_DEFAULTS, compact=True)
                submitted = st.form_submit_button("Create my account", use_container_width=True)
            if submitted:
                problems = validate_new_account(name, email, password, confirm)
                if problems:
                    for problem in problems:
                        st.error(problem)
                else:
                    user_id, error = create_user(name, email, password, profile)
                    if error:
                        st.error(error)
                    else:
                        first = name.strip().split()[0]
                        sign_in_user(user_id, f"Welcome to AquaAI, {first}!")

        with tab_admin:
            if ADMIN_STATUS != "ready":
                st.info(
                    "Administrator sign-in isn't available on this server. The server owner "
                    "needs to configure admin credentials (see the setup notes at the top "
                    "of aquaai.py)."
                )
            else:
                st.caption(
                    "Restricted area. Administrator accounts are provisioned by the server "
                    "owner and can't be created from the registration form."
                )
                with st.form("aqua_admin_form"):
                    admin_login = st.text_input("Admin username or email", key="admin_login_id")
                    admin_password = st.text_input(
                        "Admin password", type="password", key="admin_login_password"
                    )
                    submitted = st.form_submit_button(
                        "Sign in as administrator", use_container_width=True
                    )
                if submitted:
                    if not admin_login.strip() or not admin_password:
                        st.error("Enter the administrator username and password.")
                    else:
                        token, error = admin_authenticate(admin_login, admin_password)
                        if error:
                            st.error(error)
                        else:
                            sign_in_admin(token)


def goal_card(demand: float, goal: float) -> None:
    ratio = demand / goal if goal else 0.0
    if ratio <= 1:
        tone, message = "ok", f"{(1 - ratio) * 100:.0f}% under your goal"
    elif ratio <= 1.15:
        tone, message = "warn", f"{(ratio - 1) * 100:.0f}% over your goal"
    else:
        tone, message = "bad", f"{(ratio - 1) * 100:.0f}% over your goal"
    st.markdown(
        f"""
        <div class="goal-card goal-{tone}">
            <div class="goal-head">
                <span>🎯 Daily goal</span>
                <strong>{demand:,.0f} of {goal:,.0f} L/day</strong>
            </div>
            <div class="goal-track"><div class="goal-fill" style="width:{min(ratio * 100, 100):.0f}%"></div></div>
            <div class="goal-msg">{message} · AI-predicted demand for this month</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_list(items: list[str]) -> None:
    for item in items:
        st.markdown(f'<div class="insight">{item}</div>', unsafe_allow_html=True)


def empty_state(title: str, body: str) -> None:
    st.markdown(
        f'<div class="empty-state"><b>{title}</b>{body}</div>', unsafe_allow_html=True
    )


def render_dashboard_page() -> None:
    user = active_user
    if user is None:
        hero(
            "My Dashboard",
            "Create a free account to save your household profile, track daily usage "
            "against a goal and get a personalised water-saving plan.",
        )
        render_auth_panel()
        return

    first_name = esc(user["full_name"].split()[0])
    profile = {key: user[key] for key in PROFILE_DEFAULTS}
    goal = float(profile["daily_goal_lpd"])
    member_since = datetime.fromisoformat(user["created_at"]).strftime("%b %Y")

    hero(
        f"Hello, {first_name}",
        "Your personalised water snapshot: AI-predicted demand, shortage risk and "
        "the quickest wins for your home.",
    )
    st.markdown(
        f"""
        <div class="chip-row">
            <span class="chip">📍 {esc(profile['city'])}</span>
            <span class="chip">👥 {profile['household_size']} people</span>
            <span class="chip">🎯 Goal {goal:,.0f} L/day</span>
            <span class="chip">🗓️ Member since {member_since}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # AI predictions for this household, month by month.
    months = [f"{i:02d} {name}" for i, name in enumerate(calendar.month_abbr) if i]
    year = predict_many([profile])[0]
    demand = float(year[date.today().month - 1])
    available = float(profile["water_supply_lpd"] + profile["recycled_water_lpd"])
    balance = available - demand
    risk = classify_risk(balance)
    city_avg = city_average_demand(profile["city"])
    overall_avg = float(df["predicted_demand_lpd"].mean())
    versus_city = (demand / city_avg - 1) * 100 if city_avg else 0.0

    metric_cards(
        [
            ("My Predicted Demand", f"{demand:.0f} L/day"),
            ("My Available Water", f"{available:.0f} L/day"),
            ("Water Balance", f"{balance:+.0f} L/day"),
            ("Shortage Risk", risk),
            ("vs City Average", f"{versus_city:+.0f}%"),
        ]
    )
    goal_card(demand, goal)

    tab_snapshot, tab_usage, tab_plan, tab_saved = st.tabs(
        ["📊 Snapshot", "📝 Usage log", "💡 Savings plan", "💾 Saved scenarios"]
    )

    # ---- Snapshot ----------------------------------------------------
    with tab_snapshot:
        col_a, col_b = st.columns([3, 2], gap="large")
        with col_a:
            section("📈 My Year Ahead")
            st.line_chart(
                pd.DataFrame(
                    {
                        "My predicted demand": year,
                        "My available water": available,
                        "Daily goal": goal,
                    },
                    index=months,
                )
            )
        with col_b:
            section("🏙️ How I Compare")
            st.bar_chart(
                pd.Series(
                    {
                        "You": demand,
                        f"{profile['city']} average": city_avg,
                        "All-city average": overall_avg,
                    }
                )
            )

        section("🧭 Insights for My Home")
        tips = []
        if risk == "HIGH":
            tips.append("🚨 Predicted demand is well above your available water. Start with the Savings plan tab.")
        elif risk == "MEDIUM":
            tips.append("⚠️ Demand slightly exceeds supply this month. Small fixes should bring you back to balance.")
        else:
            tips.append("🟢 Your supply covers predicted demand this month. Keep monitoring your balance.")
        if demand > goal:
            tips.append(f"🎯 You're predicted to use about {demand - goal:.0f} L/day more than your goal.")
        if profile["leakage_lpd"] > 25:
            tips.append("💧 Leakage is high. Check taps, pipes and cisterns first, since it's usually the cheapest saving.")
        if profile["flush_litres"] > 6:
            tips.append("🚽 A lower-volume flush would cut toilet water noticeably.")
        if profile["recycled_water_lpd"] < 50:
            tips.append("♻️ Reusing more greywater or recycled water reduces reliance on fresh supply.")
        if profile["awareness_score"] < 60:
            tips.append("🧠 Habit changes (shorter showers, full loads) add up for homes with a lower awareness score.")
        insight_list(tips)

        st.caption(
            "Figures come from AquaAI's Random Forest model trained on a synthetic dataset, "
            "using your saved profile and your city's typical temperature."
        )

    # ---- Usage log ---------------------------------------------------
    with tab_usage:
        left, right = st.columns([1, 2], gap="large")
        with left:
            section("➕ Log Usage")
            with st.form("usage_form"):
                log_date = st.date_input("Date", value=date.today(), max_value=date.today())
                litres = st.number_input(
                    "Water used (litres)", min_value=0.0, max_value=5000.0,
                    value=float(round(demand / 10) * 10), step=10.0,
                )
                note = st.text_input("Note (optional)", max_chars=120)
                saved = st.form_submit_button("Save entry", use_container_width=True)
            if saved:
                upsert_usage(user["id"], log_date, litres, note)
                st.session_state.auth_notice = f"Saved usage for {log_date:%d %b %Y}."
                st.rerun()

        usage = list_usage(user["id"])
        with right:
            section("📉 Actual vs Goal")
            if usage.empty:
                empty_state(
                    "No entries yet",
                    "Log your household's daily water use to compare it with your goal and AquaAI's prediction.",
                )
            else:
                recent = usage.tail(60).set_index("log_date")[["litres"]]
                recent.columns = ["Logged usage"]
                recent["Daily goal"] = goal
                recent["AI prediction"] = demand
                st.line_chart(recent)

        if not usage.empty:
            average = float(usage.tail(7)["litres"].mean())
            within = float((usage.tail(30)["litres"] <= goal).mean() * 100)
            metric_cards(
                [
                    ("Recent average", f"{average:.0f} L/day"),
                    ("Days within goal", f"{within:.0f}%"),
                    ("vs AI prediction", f"{(average / demand - 1) * 100:+.0f}%"),
                    ("Entries logged", f"{len(usage):,}"),
                ]
            )
            section("🗂️ My Entries")
            table = usage.sort_values("log_date", ascending=False).copy()
            table["log_date"] = table["log_date"].dt.strftime("%Y-%m-%d")
            table.columns = ["Date", "Litres", "Note"]
            st.dataframe(table, use_container_width=True, hide_index=True)

            with st.expander("🗑️ Delete an entry"):
                chosen = st.selectbox("Entry date", table["Date"].tolist(), key="usage_delete_date")
                if st.button("Delete entry", key="usage_delete_button"):
                    delete_usage(user["id"], chosen)
                    st.session_state.auth_notice = f"Deleted the entry for {chosen}."
                    st.rerun()

    # ---- Savings plan --------------------------------------------------
    with tab_plan:
        plan, combined, already = build_savings_plan(profile)
        if plan.empty:
            st.success("🎉 Your profile already includes every action AquaAI checks. Great work!")
        else:
            section("🎯 If You Did Everything")
            metric_cards(
                [
                    ("Combined demand change", f"{combined['demand_change']:+.0f} L/day"),
                    ("Combined balance gain", f"{combined['balance_gain']:+.0f} L/day"),
                    ("Annual gain", f"{combined['annual']:,.0f} L"),
                ]
            )
            section("🔍 Action by Action")
            st.bar_chart(plan.set_index("Action")["Balance gain (L/day)"])
            st.dataframe(plan.round(1), use_container_width=True, hide_index=True)
            st.caption(
                "Balance gain = extra water headroom (available minus predicted demand), averaged over the "
                "year. Estimates come from the AI model on synthetic data, so treat them as a guide, "
                "not a guarantee."
            )
        if already:
            st.markdown("**Already in place:** " + ", ".join(already))

    # ---- Saved scenarios -----------------------------------------------
    with tab_saved:
        saved_df = list_scenarios(user["id"])
        if saved_df.empty:
            empty_state(
                "No saved scenarios yet",
                "Open the Household Simulator, tune the sliders and press “Save scenario”.",
            )
        else:
            chart_df = saved_df.head(15).copy()
            chart_df["label"] = "#" + chart_df["id"].astype(str) + " " + chart_df["name"]
            st.bar_chart(
                chart_df.set_index("label")[["predicted_lpd", "available_lpd"]].rename(
                    columns={"predicted_lpd": "Predicted demand", "available_lpd": "Available water"}
                )
            )
            shown = saved_df.drop(columns=["inputs_json"]).copy()
            shown["created_at"] = pd.to_datetime(shown["created_at"]).dt.strftime("%d %b %Y %H:%M")
            shown.columns = ["ID", "Scenario", "Saved at (UTC)", "Demand (L/day)", "Available (L/day)", "Balance (L/day)", "Risk"]
            st.dataframe(shown.round(1), use_container_width=True, hide_index=True)

            with st.expander("🗑️ Delete a scenario"):
                options = dict(zip(saved_df["id"], "#" + saved_df["id"].astype(str) + " " + saved_df["name"]))
                chosen_id = st.selectbox(
                    "Scenario", list(options), format_func=lambda i: options[i], key="scenario_delete_id"
                )
                if st.button("Delete scenario", key="scenario_delete_button"):
                    delete_scenario(user["id"], int(chosen_id))
                    st.session_state.auth_notice = "Scenario deleted."
                    st.rerun()


def render_account_page() -> None:
    user = active_user
    if user is None:
        hero("Account", "Sign in or create a free AquaAI account to personalise your experience.")
        render_auth_panel()
        return

    hero("Account", "Manage your household profile, password and personal data.")
    tab_profile, tab_security, tab_data = st.tabs(["👤 Profile & household", "🔒 Security", "🗄️ Data & privacy"])

    with tab_profile:
        with st.form("profile_form"):
            name = st.text_input("Full name", value=user["full_name"], max_chars=60)
            st.text_input("Email", value=user["email"], disabled=True)
            profile = profile_inputs("acct", user)
            saved = st.form_submit_button("Save changes", use_container_width=True)
        if saved:
            if not 2 <= len(name.strip()) <= 60:
                st.error("Enter your name (2–60 characters).")
            else:
                update_account(user["id"], name, profile)
                st.session_state.auth_notice = "Profile updated."
                st.rerun()

    with tab_security:
        with st.form("password_form"):
            current = st.text_input("Current password", type="password")
            new = st.text_input(
                "New password", type="password",
                help=f"At least {MIN_PASSWORD_LENGTH} characters, with a letter and a number.",
            )
            confirm = st.text_input("Confirm new password", type="password")
            changed = st.form_submit_button("Change password", use_container_width=True)
        if changed:
            if new != confirm:
                st.error("The new passwords do not match.")
            else:
                error = change_password(user["id"], current, new)
                if error:
                    st.error(error)
                else:
                    st.success("Password updated.")

    with tab_data:
        section("📥 Export My Data")
        usage = list_usage(user["id"])
        scenarios = list_scenarios(user["id"])
        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                "📝 Download usage log (CSV)",
                data=usage.to_csv(index=False).encode("utf-8"),
                file_name="aquaai_my_usage.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_b:
            st.download_button(
                "💾 Download saved scenarios (CSV)",
                data=scenarios.to_csv(index=False).encode("utf-8"),
                file_name="aquaai_my_scenarios.csv",
                mime="text/csv",
                use_container_width=True,
            )

        section("⚠️ Delete Account")
        with st.expander("Permanently delete my account"):
            st.warning("This removes your profile, usage log and saved scenarios. It can't be undone.")
            with st.form("delete_form"):
                password = st.text_input("Confirm with your password", type="password")
                sure = st.checkbox("I understand this is permanent")
                delete = st.form_submit_button("Delete my account", use_container_width=True)
            if delete:
                if not sure:
                    st.error("Tick the box to confirm.")
                elif delete_user(user["id"], password):
                    st.session_state.user_id = None
                    st.session_state.aqua_navigation = "🏠 Overview"
                    st.session_state.auth_notice = "Your account has been deleted."
                    st.rerun()
                else:
                    st.error("That password is incorrect.")


# ----------------------------------------------------------------
# ADMIN CONSOLE (separate from the member app; every call is role-checked)
# ----------------------------------------------------------------
ADMIN_PAGES = [
    "📊 Admin Overview",
    "👥 User Management",
    "📈 Usage Analytics",
    "🛡️ Security Monitoring",
    "🔑 Admin Profile",
]


def fmt_ts(value) -> str:
    """Readable UTC timestamp for tables."""
    if value is None or (isinstance(value, float) and math.isnan(value)) or value == "":
        return "—"
    try:
        return datetime.fromisoformat(str(value)).strftime("%d %b %Y %H:%M")
    except ValueError:
        return str(value)


def admin_hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <span class="badge">🛡️ ADMINISTRATOR CONSOLE</span>
            <h1>{title}</h1>
            <p>{subtitle}</p>
            <span class="badge">• Role-checked on the server</span>
            <span class="badge">• Actions audit-logged</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_admin_overview(token: str) -> None:
    stats = admin_overview(token)
    admin_hero(
        "Admin Overview",
        "Registrations, recent activity and sign-in health across all AquaAI accounts.",
    )
    metric_cards(
        [
            ("Total Users", f"{stats['total']:,}"),
            (f"Active Users ({ACTIVE_WINDOW_DAYS}d)", f"{stats['active']:,}"),
            ("Deactivated", f"{stats['deactivated']:,}"),
            ("New This Week", f"{stats['new_week']:,}"),
            ("Failed Logins (24h)", f"{stats['failed_24h']:,}"),
        ]
    )
    st.caption(
        f"“Active users” are enabled accounts that signed in during the last {ACTIVE_WINDOW_DAYS} days."
    )
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        section("🆕 New Registrations (30 days)")
        st.bar_chart(stats["signups"])
    with col_b:
        section("🔑 Successful Logins (30 days)")
        st.line_chart(stats["logins"])

    section("🕒 Latest Registrations")
    latest = admin_list_users(token).head(8)
    if latest.empty:
        st.info("No users have registered yet.")
    else:
        table = pd.DataFrame(
            {
                "Name": latest["full_name"],
                "Email": latest["email"],
                "City": latest["city"],
                "Registered (UTC)": latest["created_at"].map(fmt_ts),
                "Last login (UTC)": latest["last_login"].map(fmt_ts),
            }
        )
        st.dataframe(table, use_container_width=True, hide_index=True)


def render_user_management(token: str) -> None:
    admin_hero(
        "User Management",
        "Search registered users, review account details and activate, deactivate, unlock or delete accounts.",
    )
    col_search, col_status = st.columns([3, 1])
    query = col_search.text_input(
        "Search by name or email", placeholder="e.g. priya or @example.com", key="admin_user_search"
    )
    status = col_status.selectbox("Status", ["All", "Active", "Deactivated"], key="admin_user_status")
    users = admin_list_users(token, query, status)

    st.caption(f"{len(users):,} user{'s' if len(users) != 1 else ''} found")
    if users.empty:
        st.info("No users match this search.")
        return

    table = pd.DataFrame(
        {
            "ID": users["id"],
            "Name": users["full_name"],
            "Email": users["email"],
            "City": users["city"],
            "Household": users["household_size"],
            "Status": users["is_active"].map({1: "✅ Active", 0: "⛔ Deactivated"}),
            "Registered (UTC)": users["created_at"].map(fmt_ts),
            "Last login (UTC)": users["last_login"].map(fmt_ts),
            "Usage entries": users["usage_entries"],
            "Scenarios": users["scenarios"],
        }
    )
    st.dataframe(table, use_container_width=True, hide_index=True)

    section("🛠️ Manage an Account")
    labels = dict(zip(users["id"].tolist(), (users["full_name"] + " — " + users["email"]).tolist()))
    picked = st.selectbox(
        "Select a user", list(labels), format_func=lambda i: labels[i], key="admin_user_pick"
    )
    chosen = int(picked if picked in labels else next(iter(labels)))
    row = users[users["id"] == chosen].iloc[0]
    is_active = bool(row["is_active"])
    locked = bool(row["locked_until"]) and datetime.fromisoformat(row["locked_until"]) > _now()

    insight_list(
        [
            f"👤 <b>{esc(row['full_name'])}</b> · {esc(row['email'])}",
            f"📍 {esc(row['city'])} · {int(row['household_size'])} people · "
            f"registered {fmt_ts(row['created_at'])} · last login {fmt_ts(row['last_login'])} (UTC)",
            f"📝 {int(row['usage_entries'])} usage entries · 💾 {int(row['scenarios'])} saved scenarios · "
            f"{'✅ Active' if is_active else '⛔ Deactivated'}{' · 🔒 Locked by failed logins' if locked else ''}",
        ]
    )

    act_a, act_b = st.columns(2)
    with act_a:
        if is_active:
            if st.button("⛔ Deactivate account", key="admin_deactivate", use_container_width=True):
                admin_set_user_active(token, chosen, False)
                st.session_state.auth_notice = f"Deactivated {row['email']}."
                st.rerun()
        else:
            if st.button("✅ Activate account", key="admin_activate", use_container_width=True):
                admin_set_user_active(token, chosen, True)
                st.session_state.auth_notice = f"Activated {row['email']}."
                st.rerun()
    with act_b:
        if st.button(
            "🔓 Unlock account", key="admin_unlock", use_container_width=True, disabled=not locked
        ):
            admin_unlock_user(token, chosen)
            st.session_state.auth_notice = f"Unlocked {row['email']}."
            st.rerun()

    with st.expander("🗑️ Delete this account permanently"):
        st.warning("This removes the user's profile, usage log and saved scenarios. It can't be undone.")
        confirmed = st.checkbox(
            "I understand this permanently deletes the account and all its data",
            key=f"admin_delete_confirm_{chosen}",
        )
        if st.button("Delete account", key="admin_delete_button"):
            if not confirmed:
                st.error("Tick the confirmation box first.")
            else:
                admin_delete_user(token, chosen)
                st.session_state.auth_notice = f"Deleted {row['email']}."
                st.rerun()


def render_usage_analytics(token: str) -> None:
    data = admin_usage(token)
    admin_hero(
        "Usage Analytics",
        "How members use AquaAI: page views, active members and feature adoption over the last 30 days.",
    )
    metric_cards(
        [
            ("Page Views (30d)", f"{data['views_30d']:,}"),
            ("Active Members (30d)", f"{data['active_30d']:,}"),
            ("Usage Log Entries", f"{data['usage_entries']:,}"),
            ("Saved Scenarios", f"{data['scenarios']:,}"),
        ]
    )
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        section("📄 Most Used Pages")
        if data["by_page"].empty:
            st.info("No page views recorded yet.")
        else:
            st.bar_chart(data["by_page"].set_index("Page")["Views"])
    with col_b:
        section("📆 Daily Activity")
        st.line_chart(pd.concat([data["daily_views"], data["daily_users"]], axis=1))

    section("🌱 Feature Adoption")
    total = max(1, data["users_total"])
    metric_cards(
        [
            ("Members logging usage", f"{data['users_logging']:,} of {data['users_total']:,}"),
            ("Members saving scenarios", f"{data['users_scenarios']:,} of {data['users_total']:,}"),
            ("Usage-log adoption", f"{data['users_logging'] / total * 100:.0f}%"),
        ]
    )
    section("🏆 Most Active Members")
    if data["top_users"].empty:
        st.info("No members yet.")
    else:
        st.dataframe(data["top_users"], use_container_width=True, hide_index=True)
    section("🏙️ Members by City")
    if not data["cities"].empty:
        st.bar_chart(data["cities"].set_index("City")["Users"])


def render_security_monitoring(token: str) -> None:
    data = admin_security(token)
    admin_hero(
        "Security Monitoring",
        "Sign-in attempts, failed and blocked logins, locked accounts and administrator actions.",
    )
    metric_cards(
        [
            ("Member Logins (24h)", f"{data['success_24h']:,}"),
            ("Failed Member Logins (24h)", f"{data['failed_24h']:,}"),
            ("Failed Admin Logins (24h)", f"{data['admin_failed_24h']:,}"),
            ("Locked Accounts", f"{data['locked']:,}"),
        ]
    )
    section("📊 Login Attempts (7 days)")
    st.line_chart(data["daily"])

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        section("🚨 Top Failing Identifiers (24h)")
        if data["top_failing"].empty:
            st.success("No failed attempts in the last 24 hours.")
        else:
            st.dataframe(data["top_failing"], use_container_width=True, hide_index=True)
    with col_b:
        section("🔒 Currently Locked Members")
        if data["locked_users"].empty:
            st.success("No members are locked out.")
        else:
            locked = data["locked_users"].copy()
            locked["locked_until"] = locked["locked_until"].map(fmt_ts)
            locked.columns = ["ID", "Name", "Email", "Locked until (UTC)"]
            st.dataframe(locked, use_container_width=True, hide_index=True)
            st.caption("Unlock accounts from User Management.")

    section("🧾 Recent Sign-in Attempts")
    events = data["events"].copy()
    if events.empty:
        st.info("No sign-in attempts recorded yet.")
    else:
        events["ts"] = events["ts"].map(fmt_ts)
        events["outcome"] = events["outcome"].map(
            {"success": "✅ success", "failure": "❌ failure", "blocked": "⛔ blocked", "deactivated": "🚫 deactivated"}
        ).fillna(events["outcome"])
        events.columns = ["Time (UTC)", "Type", "Identifier", "Outcome"]
        st.dataframe(events, use_container_width=True, hide_index=True)

    section("📝 Administrator Activity")
    audit = data["audit"].copy()
    if audit.empty:
        st.info("No administrator actions recorded yet.")
    else:
        audit["ts"] = audit["ts"].map(fmt_ts)
        audit.columns = ["Time (UTC)", "Administrator", "Action", "Target"]
        st.dataframe(audit, use_container_width=True, hide_index=True)


def render_admin_profile(token: str, admin: dict) -> None:
    admin_hero("Admin Profile", "Your administrator account and password.")
    insight_list(
        [
            f"🛡️ Signed in as <b>{esc(admin['username'])}</b>",
            f"🗓️ Account created {fmt_ts(admin['created_at'])} · last sign-in {fmt_ts(admin['last_login'])} (UTC)",
            f"⏱️ Sessions end after {ADMIN_SESSION_IDLE_MINUTES} minutes of inactivity.",
        ]
    )
    section("🔑 Change Password")
    with st.form("admin_password_form"):
        current = st.text_input("Current password", type="password", key="admin_pw_current")
        new = st.text_input(
            "New password", type="password", key="admin_pw_new",
            help=f"At least {MIN_ADMIN_PASSWORD_LENGTH} characters, with a letter and a number.",
        )
        confirm = st.text_input("Confirm new password", type="password", key="admin_pw_confirm")
        changed = st.form_submit_button("Change password", use_container_width=True)
    if changed:
        if new != confirm:
            st.error("The new passwords do not match.")
        else:
            error = admin_change_password(token, current, new)
            if error:
                st.error(error)
            else:
                st.success("Password updated. Any other administrator sessions were signed out.")


def render_admin_app(admin: dict) -> None:
    """The whole administrator experience. Only reached with a valid admin session."""
    token = st.session_state.admin_token
    if st.session_state.admin_navigation not in ADMIN_PAGES:
        st.session_state.admin_navigation = ADMIN_PAGES[0]

    with st.sidebar:
        st.markdown(
            f"""
            <div class="aqua-brand">
                <div class="aqua-drop">🛡️</div>
                <div class="aqua-name">AquaAI</div>
                <div class="aqua-tagline">ADMIN CONSOLE</div>
            </div>
            <div class="acct-card">
                <div class="acct-avatar">🛡️</div>
                <div>
                    <div class="acct-name">{esc(admin['username'])}</div>
                    <div class="acct-sub">Administrator</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("↩ Admin sign out", key="admin_signout", use_container_width=True):
            sign_out_admin()
        st.markdown('<div class="nav-title">ADMINISTRATION</div>', unsafe_allow_html=True)
        for index, page_name in enumerate(ADMIN_PAGES):
            active = st.session_state.admin_navigation == page_name
            if st.button(
                page_name,
                key=f"admin_nav_button_{index}",
                type="primary" if active else "secondary",
                use_container_width=True,
            ):
                st.session_state.admin_navigation = page_name
                st.rerun()

    notice = st.session_state.pop("auth_notice", None)
    if notice:
        st.toast(notice, icon="✅")

    page_name = st.session_state.admin_navigation
    try:
        if page_name == "📊 Admin Overview":
            render_admin_overview(token)
        elif page_name == "👥 User Management":
            render_user_management(token)
        elif page_name == "📈 Usage Analytics":
            render_usage_analytics(token)
        elif page_name == "🛡️ Security Monitoring":
            render_security_monitoring(token)
        elif page_name == "🔑 Admin Profile":
            render_admin_profile(token, admin)
    except PermissionError:
        # The server no longer recognises this admin session (expired, revoked,
        # signed out elsewhere): drop it and return to the login screen.
        st.session_state.admin_token = None
        st.session_state.auth_notice = "Your administrator session has ended. Please sign in again."
        st.rerun()


# ----------------------------------------------------------------
# LOGIN GATE — nothing below renders until the user is signed in
# ----------------------------------------------------------------
# 1) a valid administrator session  -> the separate Admin Console only
# 2) a signed-in member             -> the AquaAI dashboard below
# 3) anyone else                    -> the login screen
active_admin = load_active_admin()
if active_admin is not None:
    render_admin_app(active_admin)
    st.stop()

if not is_signed_in:
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"],
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="collapsedControl"] { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    _notice = st.session_state.pop("auth_notice", None)
    if _notice:
        st.toast(_notice, icon="✅")
    hero(
        "Welcome to AquaAI",
        "Sign in or create a free account to open your AI-powered water intelligence dashboard.",
    )
    render_auth_panel()
    st.stop()


# ----------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------
NAV_ITEMS = [
    ("🏠 Overview", "Overview"),
    ("👤 My Dashboard", "My Dashboard"),
    ("🤖 AI Predictions", "AI Predictions"),
    ("🔮 Forecast", "Forecast"),
    ("🚨 Risk Monitor", "Risk Monitor"),
    ("🏙️ City Analytics", "City Analytics"),
    ("🏠 Household Simulator", "Household Simulator"),
    ("💧 Leakage Intelligence", "Leakage Intelligence"),
    ("♻️ Sustainability", "Sustainability"),
    ("🧠 Explainable AI", "Explainable AI"),
    ("📑 Reports", "Reports"),
    ("⚙️ Account", "Account"),
]

with st.sidebar:
    st.markdown(
        """
        <div class="aqua-brand">
            <div class="aqua-drop">💧</div>
            <div class="aqua-name">AquaAI</div>
            <div class="aqua-tagline">SMART WATER INTELLIGENCE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Account card (only reachable when signed in, thanks to the login gate).
    st.markdown(
        f"""
        <div class="acct-card">
            <div class="acct-avatar">{esc(active_user['full_name'].strip()[:1].upper())}</div>
            <div>
                <div class="acct-name">{esc(active_user['full_name'])}</div>
                <div class="acct-sub">{esc(active_user['city'])} · {active_user['household_size']} people</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("↩ Sign out", key="acct_signout", use_container_width=True):
        sign_out_user()

    st.markdown('<div class="nav-title">MAIN DASHBOARD</div>', unsafe_allow_html=True)

    for index, (page_name, _) in enumerate(NAV_ITEMS):
        active = st.session_state.aqua_navigation == page_name

        # The active page uses the "primary" button style, which the CSS
        # highlights (a wrapper div around a Streamlit widget can't do that).
        clicked = st.button(
            page_name,
            key=f"aqua_nav_button_{index}",
            type="primary" if active else "secondary",
            use_container_width=True,
        )

        if clicked:
            st.session_state.aqua_navigation = page_name
            st.rerun()

    st.divider()

    dark_value = st.toggle(
        "🌙 Dark mode",
        value=st.session_state.dark_mode,
        key="sidebar_dark_mode",
    )

    if dark_value != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_value
        st.rerun()

    high_risk_count = int((df["shortage_risk"] == "HIGH").sum())

    st.markdown(
        f"""
        <div class="sidebar-status">
            <b style="color:white;">🟢 SYSTEM ONLINE</b>
            <br><br>
            🤖 AI Model: Random Forest<br>
            📊 Records: {len(df):,}<br>
            🎯 R² Score: {r2:.3f}<br>
            🚨 High Risk: {high_risk_count:,}<br>
            💧 Water Engine: Active
        </div>
        <div class="sidebar-footer">
            AquaAI v2.0<br>
            AI • Analytics • Sustainability
        </div>
        """,
        unsafe_allow_html=True,
    )

page = st.session_state.aqua_navigation

# Only real member pages can ever be shown here; a tampered navigation value
# (including an admin page name) falls back to Overview. Admin pages exist
# solely in render_admin_app(), behind a server-validated admin session.
if page not in {name for name, _ in NAV_ITEMS}:
    page = "🏠 Overview"
    st.session_state.aqua_navigation = page

if st.session_state.get("last_logged_page") != page:
    log_page_view(active_user["id"], page)
    st.session_state.last_logged_page = page

_notice = st.session_state.pop("auth_notice", None)
if _notice:
    st.toast(_notice, icon="✅")


# =================================================================
# 1. OVERVIEW
# =================================================================
if page == "🏠 Overview":
    hero(
        "AquaAI",
        "AI-powered household water intelligence. Predict demand, "
        "monitor shortage risk and discover water-saving opportunities.",
    )

    left, center, right = st.columns([1, 2, 1])
    with center:
        st.markdown(
            """
            <style>
            .get-started-wrap .stButton > button {
                min-height: 58px !important;
                border-radius: 999px !important;
                font-size: 20px !important;
                font-weight: 800 !important;
                background: linear-gradient(135deg,#0799f5,#276cf5,#7b2ff7) !important;
                box-shadow: 0 14px 32px rgba(43,108,245,.28) !important;
            }
            </style>
            <div class="get-started-wrap">
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Get Started",
            key="overview_get_started",
            use_container_width=True,
        ):
            st.session_state.aqua_navigation = "🤖 AI Predictions"
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    dashboard_status(len(df), r2, high_risk_count)

    metric_cards(
        [
            ("Households", f"{df['household_id'].nunique():,}"),
            ("Avg AI Demand", f"{df['predicted_demand_lpd'].mean():.0f} L/day"),
            ("Avg Available", f"{df['total_available_lpd'].mean():.0f} L/day"),
            ("Saving Potential", f"{df['potential_flush_saving_lpd'].mean():,.0f} L/day"),
            ("High Risk Records", f"{high_risk_count:,}"),
        ]
    )

    section("📈 Water Demand Trend")
    daily = df.groupby("date")[
        ["actual_consumption_lpd", "predicted_demand_lpd", "total_available_lpd"]
    ].mean()

    st.line_chart(
        daily.rename(
            columns={
                "actual_consumption_lpd": "Actual",
                "predicted_demand_lpd": "AI Prediction",
                "total_available_lpd": "Available",
            }
        )
    )

    section("🚨 Risk Distribution")
    risk_counts = (
        df["shortage_risk"]
        .value_counts()
        .reindex(["LOW", "MEDIUM", "HIGH"])
        .fillna(0)
    )
    st.bar_chart(risk_counts)

    section("🏙️ City Snapshot")
    city_snapshot = (
        df.groupby("city")
        .agg(
            demand_lpd=("predicted_demand_lpd", "mean"),
            available_lpd=("total_available_lpd", "mean"),
            leakage_lpd=("leakage_lpd", "mean"),
        )
        .round(1)
    )
    st.dataframe(city_snapshot, use_container_width=True)


# =================================================================
# 2. AI PREDICTIONS
# =================================================================
elif page == "🤖 AI Predictions":
    hero(
        "AI Predictions",
        "Random Forest-powered water-demand prediction.",
    )

    metric_cards(
        [
            ("R² Score", f"{r2:.3f}"),
            ("MAE", f"{mae:.1f} L/day"),
            ("RMSE", f"{rmse:.1f} L/day"),
        ]
    )

    section("📊 Actual vs Predicted")
    st.scatter_chart(evaluation, x="actual", y="predicted")

    section("🧠 Feature Importance")
    st.dataframe(importance, use_container_width=True, hide_index=True)

    section("📋 Prediction Sample")
    columns = [
        "date",
        "household_id",
        "city",
        "actual_consumption_lpd",
        "predicted_demand_lpd",
        "total_available_lpd",
        "water_balance_lpd",
        "shortage_risk",
    ]
    st.dataframe(df[columns].head(100), use_container_width=True, hide_index=True)


# =================================================================
# 3. FORECAST
# =================================================================
elif page == "🔮 Forecast":
    hero(
        "Forecast",
        "Estimate upcoming household water demand using recent "
        "AI-predicted demand patterns.",
    )

    days = st.selectbox("Forecast horizon", [7, 30, 90])

    daily = (
        df.groupby("date")["predicted_demand_lpd"]
        .mean()
        .sort_index()
    )
    last_14 = daily.tail(14)

    if len(last_14) >= 2:
        x = np.arange(len(last_14))
        trend = float(np.polyfit(x, last_14.to_numpy(), 1)[0])
    else:
        trend = 0.0

    baseline = float(last_14.mean()) if len(last_14) else float(daily.mean())

    future_dates = pd.date_range(
        daily.index.max() + pd.Timedelta(days=1),
        periods=days,
    )

    forecast = np.maximum(
        150,
        baseline + trend * np.arange(1, days + 1),
    )

    forecast_df = pd.DataFrame(
        {"forecast_lpd": forecast},
        index=future_dates,
    )

    metric_cards(
        [
            ("Forecast Days", str(days)),
            ("Starting Level", f"{baseline:.0f} L/day"),
            ("Average Forecast", f"{forecast.mean():.0f} L/day"),
        ]
    )

    section("🔮 Forecast Trend")
    st.line_chart(forecast_df)

    st.dataframe(forecast_df.round(1), use_container_width=True)

    st.info(
        "This is a trend-based projection from recent AI predictions; "
        "it is not a separately trained time-series forecasting model."
    )


# =================================================================
# 4. RISK MONITOR
# =================================================================
elif page == "🚨 Risk Monitor":
    hero(
        "Risk Monitor",
        "Identify records where predicted demand is higher than available water.",
    )

    counts = (
        df["shortage_risk"]
        .value_counts()
        .reindex(["LOW", "MEDIUM", "HIGH"])
        .fillna(0)
    )

    metric_cards(
        [
            ("LOW", f"{int(counts['LOW']):,}"),
            ("MEDIUM", f"{int(counts['MEDIUM']):,}"),
            ("HIGH", f"{int(counts['HIGH']):,}"),
        ]
    )

    section("🚨 Risk Distribution")
    st.bar_chart(counts)

    section("🔥 Priority Households")
    priority = (
        df[df["shortage_risk"] == "HIGH"]
        .sort_values("water_balance_lpd")
        .head(100)
    )

    priority_columns = [
        "household_id",
        "city",
        "household_size",
        "predicted_demand_lpd",
        "total_available_lpd",
        "water_balance_lpd",
        "leakage_lpd",
        "recommendation",
    ]
    st.dataframe(
        priority[priority_columns],
        use_container_width=True,
        hide_index=True,
    )


# =================================================================
# 5. CITY ANALYTICS
# =================================================================
elif page == "🏙️ City Analytics":
    hero(
        "City Analytics",
        "Compare demand, supply, leakage and recycling patterns across cities.",
    )

    city = st.selectbox(
        "Select city",
        ["All"] + sorted(df["city"].unique().tolist()),
    )

    view = df if city == "All" else df[df["city"] == city]

    metric_cards(
        [
            ("Predicted Demand", f"{view['predicted_demand_lpd'].mean():.0f} L/day"),
            ("Available Water", f"{view['total_available_lpd'].mean():.0f} L/day"),
            ("Leakage", f"{view['leakage_lpd'].mean():.1f} L/day"),
            ("Recycled", f"{view['recycled_water_lpd'].mean():.0f} L/day"),
        ]
    )

    section("🏙️ Average Demand by City")
    city_demand = (
        df.groupby("city")["predicted_demand_lpd"]
        .mean()
        .sort_values(ascending=False)
    )
    st.bar_chart(city_demand)

    section("📊 City Summary")
    summary = (
        df.groupby("city")
        .agg(
            households=("household_id", "nunique"),
            demand_lpd=("predicted_demand_lpd", "mean"),
            available_lpd=("total_available_lpd", "mean"),
            leakage_lpd=("leakage_lpd", "mean"),
            recycled_lpd=("recycled_water_lpd", "mean"),
            high_risk_records=(
                "shortage_risk",
                lambda x: int((x == "HIGH").sum()),
            ),
        )
        .round(1)
    )
    st.dataframe(summary, use_container_width=True)


# =================================================================
# 6. HOUSEHOLD SIMULATOR
# =================================================================
elif page == "🏠 Household Simulator":
    hero(
        "Household Simulator",
        "Change household conditions and instantly estimate daily water demand and shortage risk.",
    )

    # Starting values: the original defaults for guests, or the signed-in
    # member's saved household profile.
    sim = {
        "size": 4, "temp": 30, "rainfall": 5.0, "supply": 650, "previous": 500,
        "flushes": 18, "flush": 6, "grey": 100, "recycled": 70,
        "leakage": 10.0, "awareness": 75, "stress": 0.5,
    }
    if is_signed_in:
        saved_flush = int(active_user["flush_litres"])
        sim.update(
            size=int(active_user["household_size"]),
            temp=int(min(50, max(10, round(city_temperature(active_user["city"]))))),
            supply=int(active_user["water_supply_lpd"]),
            previous=int(active_user["previous_consumption_lpd"]),
            flushes=int(active_user["toilet_flushes"]),
            flush=saved_flush if saved_flush in FLUSH_OPTIONS else 6,
            grey=int(active_user["greywater_available_lpd"]),
            recycled=int(active_user["recycled_water_lpd"]),
            leakage=float(active_user["leakage_lpd"]),
            awareness=int(active_user["awareness_score"]),
            stress=float(active_user["water_stress_index"]),
        )
        st.info("🏠 Sliders start from your saved household profile. Change them freely to explore.")

    c1, c2, c3 = st.columns(3)

    with c1:
        size = st.slider("Household size", 2, 12, sim["size"])
        temp = st.slider("Temperature °C", 10, 50, sim["temp"])
        rainfall = st.slider("Rainfall mm", 0.0, 100.0, sim["rainfall"])
        supply = st.slider("Water supply L/day", 200, 1200, sim["supply"])
        previous = st.slider("Previous consumption L/day", 150, 1200, sim["previous"])

    with c2:
        flushes = st.slider("Toilet flushes/day", 1, 40, sim["flushes"])
        flush = st.select_slider(
            "Flush litres",
            options=FLUSH_OPTIONS,
            value=sim["flush"],
        )
        grey = st.slider("Greywater available L/day", 0, 250, sim["grey"])
        recycled = st.slider("Recycled water L/day", 0, 200, sim["recycled"])

    with c3:
        leakage = st.slider("Leakage L/day", 0.0, 100.0, sim["leakage"])
        awareness = st.slider("Awareness score", 0, 100, sim["awareness"])
        stress = st.slider("Water stress index", 0.0, 1.0, sim["stress"])

    today = pd.Timestamp.today()

    row = pd.DataFrame(
        [{
            "household_size": size,
            "temperature_c": temp,
            "rainfall_mm": rainfall,
            "water_supply_lpd": supply,
            "previous_consumption_lpd": previous,
            "toilet_flushes": flushes,
            "flush_litres": flush,
            "greywater_available_lpd": grey,
            "recycled_water_lpd": recycled,
            "leakage_lpd": leakage,
            "awareness_score": awareness,
            "water_stress_index": stress,
            "month": today.month,
            "day_of_week": today.dayofweek,
        }]
    )

    try:
        prediction = float(model.predict(row[FEATURES])[0])
    except Exception:
        st.error("Could not compute a prediction for these inputs. Please adjust the sliders and try again.")
        st.stop()

    available = supply + recycled
    balance = available - prediction

    risk = (
        "HIGH" if balance < -100
        else "MEDIUM" if balance < 0
        else "LOW"
    )

    flush_saving = flushes * max(0, flush - EFFICIENT_FLUSH_LITRES)

    metric_cards(
        [
            ("Predicted Demand", f"{prediction:.0f} L/day"),
            ("Available", f"{available:.0f} L/day"),
            ("Balance", f"{balance:.0f} L/day"),
            ("Risk", risk),
            ("Potential Flush Saving", f"{flush_saving:.0f} L/day"),
        ]
    )

    if risk == "HIGH":
        st.error("🚨 High shortage risk. Reduce freshwater use and inspect leakage.")
    elif risk == "MEDIUM":
        st.warning("⚠️ Medium shortage risk. Reuse water and reduce consumption.")
    else:
        st.success("🟢 Low shortage risk under these conditions.")

    section("💡 Simulator Recommendation")

    recommendations = []
    if leakage > 25:
        recommendations.append("💧 Leakage is relatively high. Inspect taps, pipes and toilet systems.")
    if flush > 6:
        recommendations.append("🚽 Consider a lower-volume efficient flush.")
    if recycled < 50:
        recommendations.append("♻️ Increasing recycled-water use can reduce dependence on fresh supply.")
    if awareness < 60:
        recommendations.append("🧠 Improving water-use awareness may help reduce unnecessary consumption.")

    if recommendations:
        for item in recommendations:
            st.write(item)
    else:
        st.write("✅ Current settings show no additional simulator recommendation.")

    section("💾 Save This Scenario")
    if is_signed_in:
        name_col, button_col = st.columns([3, 1], vertical_alignment="bottom")
        scenario_name = name_col.text_input(
            "Scenario name",
            placeholder="e.g. After fixing the leak",
            max_chars=80,
            key="sim_scenario_name",
        )
        if button_col.button("Save scenario", key="sim_save_scenario", use_container_width=True):
            problem = save_scenario(
                active_user["id"],
                scenario_name,
                {
                    "household_size": size, "temperature_c": temp, "rainfall_mm": rainfall,
                    "water_supply_lpd": supply, "previous_consumption_lpd": previous,
                    "toilet_flushes": flushes, "flush_litres": flush,
                    "greywater_available_lpd": grey, "recycled_water_lpd": recycled,
                    "leakage_lpd": leakage, "awareness_score": awareness,
                    "water_stress_index": stress,
                },
                prediction,
                available,
                balance,
                risk,
            )
            if problem:
                st.error(problem)
            else:
                st.success("Saved! Find it under My Dashboard → Saved scenarios.")
    else:
        st.info("🔐 Sign in from the sidebar to save scenarios and revisit them on your dashboard.")


# =================================================================
# 7. LEAKAGE INTELLIGENCE
# =================================================================
elif page == "💧 Leakage Intelligence":
    hero(
        "Leakage Intelligence",
        "Find where water losses are concentrated and estimate their impact.",
    )

    average_leakage = float(df["leakage_lpd"].mean())
    monthly_loss = average_leakage * 30
    annual_loss = average_leakage * 365

    metric_cards(
        [
            ("Average Leakage", f"{average_leakage:.1f} L/day"),
            ("Monthly Estimate", f"{monthly_loss:,.0f} L"),
            ("Annual Estimate", f"{annual_loss:,.0f} L"),
        ]
    )

    section("🚨 Highest Leakage Records")
    top_leakage = df.nlargest(50, "leakage_lpd")[
        [
            "household_id",
            "city",
            "leakage_lpd",
            "predicted_demand_lpd",
            "water_balance_lpd",
            "shortage_risk",
        ]
    ]
    st.dataframe(top_leakage, use_container_width=True, hide_index=True)

    section("📊 Leakage by City")
    leakage_city = (
        df.groupby("city")["leakage_lpd"]
        .mean()
        .sort_values(ascending=False)
    )
    st.bar_chart(leakage_city)

    section("🔎 Leakage Distribution")
    leakage_trend = (
        df.sort_values("date")
        .set_index("date")["leakage_lpd"]
        .rolling(30, min_periods=1)
        .mean()
    )
    st.line_chart(leakage_trend)


# =================================================================
# 8. SUSTAINABILITY
# =================================================================
elif page == "♻️ Sustainability":
    hero(
        "Sustainability",
        "Measure opportunities from efficient flushing, recycling, reuse and leakage reduction.",
    )

    annual_flush = float(df["annual_flush_saving_litres"].sum())
    recycled = float(df["recycled_water_lpd"].mean())
    flush_saving = float(df["potential_flush_saving_lpd"].mean())
    leakage = float(df["leakage_lpd"].mean())

    metric_cards(
        [
            ("Annual Flush Saving", f"{annual_flush:,.0f} L"),
            ("Average Recycled", f"{recycled:.0f} L/day"),
            ("Potential Flush Saving", f"{flush_saving:.0f} L/day"),
            ("Average Leakage", f"{leakage:.1f} L/day"),
        ]
    )

    section("♻️ Sustainability Opportunity")
    st.progress(
        min(1.0, max(0.0, flush_saving / 100)),
        text=f"Flush-efficiency opportunity: {flush_saving:.0f} L/day average",
    )

    section("🚽 Flush Efficiency")
    flush_table = (
        df.groupby("flush_litres")
        .agg(
            average_saving_lpd=("potential_flush_saving_lpd", "mean"),
            households=("household_id", "count"),
        )
        .round(1)
    )
    st.dataframe(flush_table, use_container_width=True)

    section("♻️ Recycled Water by City")
    recycled_city = (
        df.groupby("city")["recycled_water_lpd"]
        .mean()
        .sort_values(ascending=False)
    )
    st.bar_chart(recycled_city)

    st.info("These are demonstration estimates based on AquaAI's synthetic dataset.")


# =================================================================
# 9. EXPLAINABLE AI
# =================================================================
elif page == "🧠 Explainable AI":
    hero(
        "Explainable AI",
        "Understand which input variables are most influential in AquaAI's Random Forest model.",
    )

    section("🧠 Model Feature Importance")
    top_features = importance.head(14).set_index("feature")["importance"]
    st.bar_chart(top_features)

    st.dataframe(importance, use_container_width=True, hide_index=True)

    section("🔍 Explain a Household Prediction")

    index = st.number_input(
        "Select dataset row",
        min_value=0,
        max_value=max(0, len(df) - 1),
        value=0,
        step=1,
    )

    selected = df.iloc[int(index)]

    st.write(
        f"**Household:** {selected['household_id']}  |  "
        f"**City:** {selected['city']}"
    )

    importance_indexed = importance.set_index("feature")

    contribution_data = pd.DataFrame(
        {
            "feature": FEATURES,
            "value": [selected[f] for f in FEATURES],
            "importance": [
                importance_indexed.loc[f, "importance"]
                for f in FEATURES
            ],
        }
    ).sort_values("importance", ascending=False)

    st.dataframe(
        contribution_data,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Random Forest feature importance indicates relative model influence. "
        "It does not prove that a feature causes water consumption."
    )


# =================================================================
# 10. REPORTS
# =================================================================
elif page == "📑 Reports":
    hero(
        "Reports",
        "Export AquaAI predictions, source data and a compact management report.",
    )

    section("📊 CSV Downloads")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "📊 Download Predictions CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="aquaai_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        raw_data = generate_data()
        st.download_button(
            "🗃️ Download Raw Dataset CSV",
            data=raw_data.to_csv(index=False).encode("utf-8"),
            file_name="aquaai_household_water_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    section("📄 PDF Report")

    if not REPORTLAB_AVAILABLE:
        st.warning(
            "ReportLab is not installed. Install it with: "
            "python3 -m pip install reportlab"
        )
    else:

        @st.cache_data(show_spinner=False)
        def create_pdf(
            record_count: int,
            household_count: int,
            avg_demand: float,
            avg_available: float,
            high_risk_count: int,
            r2_score_: float,
            mae_: float,
            rmse_: float,
            risk_counts: tuple,
        ) -> bytes:
            """Build the summary PDF report. Cached on its scalar inputs so it is
            only rebuilt when the underlying figures actually change, instead of
            on every Streamlit rerun."""
            buffer = BytesIO()

            document = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                rightMargin=36,
                leftMargin=36,
                topMargin=36,
                bottomMargin=36,
            )

            styles = getSampleStyleSheet()

            story = [
                Paragraph("AquaAI Water Management Report", styles["Title"]),
                Spacer(1, 12),
                Paragraph(f"Records analyzed: {record_count:,}", styles["BodyText"]),
                Paragraph(f"Households: {household_count:,}", styles["BodyText"]),
                Paragraph(
                    f"Average predicted demand: {avg_demand:.1f} L/day",
                    styles["BodyText"],
                ),
                Paragraph(
                    f"Average available water: {avg_available:.1f} L/day",
                    styles["BodyText"],
                ),
                Paragraph(
                    f"High-risk records: {high_risk_count:,}",
                    styles["BodyText"],
                ),
                Paragraph(
                    f"Model R²: {r2_score_:.3f} | MAE: {mae_:.1f} | RMSE: {rmse_:.1f}",
                    styles["BodyText"],
                ),
                Spacer(1, 15),
                Paragraph("Risk Summary", styles["Heading2"]),
            ]

            low_count, medium_count, high_count = risk_counts

            table_data = [
                ["Risk", "Records"],
                ["LOW", low_count],
                ["MEDIUM", medium_count],
                ["HIGH", high_count],
            ]

            table = Table(table_data, colWidths=[120, 100])

            table.setStyle(
                TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                    ]
                )
            )

            story.extend(
                [
                    table,
                    Spacer(1, 15),
                    Paragraph(
                        "AquaAI uses a synthetic demonstration dataset. "
                        "Results should not be treated as real-world measurements.",
                        styles["BodyText"],
                    ),
                ]
            )

            document.build(story)
            buffer.seek(0)
            return buffer.getvalue()

        risk_value_counts = df["shortage_risk"].value_counts()

        pdf_bytes = create_pdf(
            record_count=len(df),
            household_count=int(df["household_id"].nunique()),
            avg_demand=float(df["predicted_demand_lpd"].mean()),
            avg_available=float(df["total_available_lpd"].mean()),
            high_risk_count=int((df["shortage_risk"] == "HIGH").sum()),
            r2_score_=float(r2),
            mae_=float(mae),
            rmse_=float(rmse),
            risk_counts=(
                int(risk_value_counts.get("LOW", 0)),
                int(risk_value_counts.get("MEDIUM", 0)),
                int(risk_value_counts.get("HIGH", 0)),
            ),
        )

        st.download_button(
            "📄 Download PDF Report",
            data=pdf_bytes,
            file_name="AquaAI_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    section("📋 Current Model Summary")

    st.dataframe(
        pd.DataFrame(
            {
                "Metric": [
                    "Model",
                    "Trees",
                    "Max Depth",
                    "MAE",
                    "RMSE",
                    "R²",
                ],
                "Value": [
                    "Random Forest Regressor",
                    "300",
                    "15",
                    f"{mae:.2f}",
                    f"{rmse:.2f}",
                    f"{r2:.4f}",
                ],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# =================================================================
# 11. MY DASHBOARD  (personalised; guests see sign-in / register)
# =================================================================
elif page == "👤 My Dashboard":
    render_dashboard_page()


# =================================================================
# 12. ACCOUNT  (profile, security, data; guests see sign-in / register)
# =================================================================
elif page == "⚙️ Account":
    render_account_page()


# ----------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------
st.markdown(
    """
    <br><br>
    <div style="
        text-align:center;
        opacity:.65;
        padding:20px;
        font-size:13px;
    ">
        💧 <b>AquaAI</b> — Smart Water Management Dashboard
        <br>
        AI • Analytics • Sustainability • Water Intelligence
    </div>
    """,
    unsafe_allow_html=True,
)
