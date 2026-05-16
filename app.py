"""
╔══════════════════════════════════════════════════════════════════╗
║   LOAN APPROVAL PREDICTION - ML DASHBOARD                       ║
║   Models: Decision Tree | Random Forest | Gradient Boosting     ║
║   Developed by: SRI VARUN SINGARI                               ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "force_sidebar_visible" not in st.session_state:
    st.session_state.force_sidebar_visible = False

# ─────────────────────────────────────────────
# CUSTOM CSS - Modern Dark Financial Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

    /* ── Root Variables ── */
    :root {
        --bg-dark:       #090e1a;
        --bg-card:       #0f1829;
        --bg-card2:      #131e30;
        --accent-gold:   #f5c842;
        --accent-teal:   #00d4aa;
        --accent-red:    #ff4d6d;
        --accent-blue:   #4da6ff;
        --text-primary:  #e8edf5;
        --text-muted:    #7a8ba0;
        --border:        #1e2d42;
    }

    /* ── Global Reset ── */
    .stApp { background: var(--bg-dark) !important; }
    html, body, [class*="css"] {
        font-family: 'DM Mono', monospace !important;
        color: var(--text-primary);
    }

    /* ── Hide Streamlit Branding ── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1322 0%, #0f1829 100%) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text-primary) !important; }

    /* ── Main Title ── */
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #f5c842 0%, #00d4aa 50%, #4da6ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.15;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        font-family: 'DM Mono', monospace;
        font-size: 0.82rem;
        color: var(--text-muted);
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin-bottom: 2rem;
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 32px rgba(0,0,0,0.4);
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-gold), var(--accent-teal));
        border-radius: 14px 14px 0 0;
    }
    .metric-label {
        font-size: 0.7rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-family: 'Syne', sans-serif;
        font-size: 1.9rem;
        font-weight: 700;
        color: var(--accent-gold);
        line-height: 1;
    }
    .metric-badge {
        display: inline-block;
        font-size: 0.65rem;
        letter-spacing: 0.12em;
        padding: 3px 9px;
        border-radius: 20px;
        margin-top: 0.4rem;
        background: rgba(245,200,66,0.12);
        color: var(--accent-gold);
        border: 1px solid rgba(245,200,66,0.25);
    }

    /* ── Section Headers ── */
    .section-header {
        font-family: 'Syne', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-primary);
        border-left: 3px solid var(--accent-gold);
        padding-left: 0.75rem;
        margin: 1.5rem 0 1rem 0;
        letter-spacing: 0.04em;
    }

    /* ── Result Cards ── */
    .result-approved {
        background: linear-gradient(135deg, rgba(0,212,170,0.12) 0%, rgba(0,212,170,0.04) 100%);
        border: 1px solid rgba(0,212,170,0.35);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-rejected {
        background: linear-gradient(135deg, rgba(255,77,109,0.12) 0%, rgba(255,77,109,0.04) 100%);
        border: 1px solid rgba(255,77,109,0.35);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-title {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 800;
    }
    .result-approved .result-title { color: var(--accent-teal); }
    .result-rejected .result-title { color: var(--accent-red); }

    /* ── Reason Chips ── */
    .reason-box {
        background: var(--bg-card2);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.4rem 0;
        font-size: 0.85rem;
        display: flex;
        align-items: flex-start;
        gap: 0.6rem;
    }
    .reason-good  { border-left: 3px solid var(--accent-teal); }
    .reason-bad   { border-left: 3px solid var(--accent-red); }
    .reason-warn  { border-left: 3px solid var(--accent-gold); }

    /* ── Performance Table ── */
    .perf-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.82rem;
    }
    .perf-table th {
        background: rgba(245,200,66,0.08);
        color: var(--accent-gold);
        text-align: left;
        padding: 10px 14px;
        font-size: 0.7rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        border-bottom: 1px solid var(--border);
    }
    .perf-table td {
        padding: 10px 14px;
        border-bottom: 1px solid rgba(30,45,66,0.5);
        color: var(--text-primary);
    }
    .perf-table tr:hover td { background: rgba(255,255,255,0.02); }
    .best-row td { color: var(--accent-gold) !important; font-weight: 600; }

    /* ── Info Box ── */
    .info-box {
        background: var(--bg-card2);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin: 0.5rem 0;
        font-size: 0.84rem;
        line-height: 1.7;
        color: var(--text-primary);
    }
    .info-box strong { color: var(--accent-gold); }

    /* ── Sidebar Nav Item ── */
    .nav-item {
        padding: 8px 12px;
        border-radius: 8px;
        margin: 3px 0;
        font-size: 0.85rem;
        cursor: pointer;
        transition: background 0.15s;
    }
    .nav-item:hover { background: rgba(245,200,66,0.08); }

    /* ── Confidence Bar ── */
    .conf-track {
        background: var(--border);
        border-radius: 8px;
        height: 10px;
        width: 100%;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    .conf-fill-good  { background: linear-gradient(90deg, #00d4aa, #4da6ff); height:100%; border-radius:8px; }
    .conf-fill-bad   { background: linear-gradient(90deg, #ff4d6d, #ff8c69); height:100%; border-radius:8px; }

    /* ── Streamlit overrides ── */
    .stSlider > div > div { background: var(--border) !important; }
    .stSelectbox label, .stNumberInput label, .stSlider label,
    .stRadio label, div[data-testid="stMarkdownContainer"] p {
        color: var(--text-primary) !important;
        font-size: 0.83rem !important;
    }
    .stButton > button,
    div.stButton button,
    button[kind="primary"],
    button[kind="secondary"],
    [data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #f5c842, #f0a500) !important;
        color: #0a0e1a !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.65rem 1.8rem !important;
        width: 100% !important;
        letter-spacing: 0.06em !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover,
    div.stButton button:hover,
    button[kind="primary"]:hover,
    button[kind="secondary"]:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(245,200,66,0.35) !important;
    }
    div[data-testid="stExpander"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        font-size: 0.75rem;
        color: var(--text-muted);
        border-top: 1px solid var(--border);
        margin-top: 3rem;
        letter-spacing: 0.1em;
    }
    .footer span { color: var(--accent-gold); font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
@st.cache_data
def load_and_preprocess():
    """Load CSV, preprocess, engineer features, encode, return clean df + encoders."""
    try:
        df = pd.read_csv("loan_approval_dataset.csv")
    except FileNotFoundError:
        st.error("⚠️ `loan_approval_dataset.csv` not found. Please place it in the same directory as `app.py`.")
        st.stop()

    # Strip whitespace and normalize column names to lowercase snake_case
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Map common alternative column names from external datasets to the
    # canonical names expected by this app.
    rename_map = {
        "applicant_id": "loan_id",
        "annual_income": "income_annum",
        "credit_score": "cibil_score",
        "loan_amount": "loan_amount",
        "loan_status": "loan_status",
        "education": "education",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # Fail fast with a helpful message when absolutely critical columns are missing.
    required = ["loan_amount", "income_annum"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"⚠️ Missing required columns in CSV: {', '.join(missing)}. Detected columns: {', '.join(df.columns)}")
        st.stop()

    # Provide sensible defaults for optional features expected by the model
    # so the app can still run on datasets with fewer columns.
    defaults = {
        "self_employed": "No",
        "education": "Not Graduate",
        "loan_status": "Rejected",
        "no_of_dependents": 0,
        "loan_term": 10,
        "cibil_score": df.get("cibil_score", pd.Series([650]*len(df))),
        "residential_assets_value": 0,
        "commercial_assets_value": 0,
        "luxury_assets_value": 0,
        "bank_asset_value": 0,
    }

    for col, val in defaults.items():
        if col not in df.columns:
            # If default is a Series (e.g., cibil_score mapping), assign directly
            df[col] = val if not isinstance(val, pd.Series) else val

    # Ensure numeric defaults have the right dtype
    num_cols = ["no_of_dependents", "loan_term", "cibil_score",
                "residential_assets_value", "commercial_assets_value",
                "luxury_assets_value", "bank_asset_value"]
    for nc in num_cols:
        if nc in df.columns:
            df[nc] = pd.to_numeric(df[nc], errors="coerce").fillna(0)

    # Drop loan_id (non-predictive identifier)
    if "loan_id" in df.columns:
        df.drop(columns=["loan_id"], inplace=True)

    # Handle missing values
    df.dropna(inplace=True)

    # ── Feature Engineering ──
    # loan_income_ratio: how many times loan exceeds annual income
    df["loan_income_ratio"] = df["loan_amount"] / (df["income_annum"] + 1)

    # ── Label Encoding ──
    le_edu      = LabelEncoder()
    le_self     = LabelEncoder()
    le_status   = LabelEncoder()

    # Fit encoders on training dataset and record a safe default for unseen labels
    le_edu.fit(df["education"])
    edu_default = df["education"].mode().iloc[0] if not df["education"].mode().empty else le_edu.classes_[0]
    df["education_enc"] = le_edu.transform(df["education"])

    le_self.fit(df["self_employed"])
    self_default = df["self_employed"].mode().iloc[0] if not df["self_employed"].mode().empty else le_self.classes_[0]
    df["self_employed_enc"] = le_self.transform(df["self_employed"])

    le_status.fit(df["loan_status"])
    status_default = df["loan_status"].mode().iloc[0] if not df["loan_status"].mode().empty else le_status.classes_[0]
    df["loan_status_enc"] = le_status.transform(df["loan_status"])

    encoders = {
        "education": {"encoder": le_edu, "default": edu_default},
        "self_employed": {"encoder": le_self, "default": self_default},
        "loan_status": {"encoder": le_status, "default": status_default}
    }

    # Auto-encode any other object/string columns (e.g., marital_status)
    for col in df.select_dtypes(include="object").columns:
        if col not in ["education", "self_employed", "loan_status"]:
            le = LabelEncoder()
            try:
                df[col + "_enc"] = le.fit_transform(df[col].astype(str))
                encoders[col] = {"encoder": le, "default": df[col].mode().iloc[0] if not df[col].mode().empty else str(le.classes_[0])}
            except Exception:
                # skip columns that cannot be encoded
                continue

    return df, encoders


@st.cache_resource
def train_models(df):
    """Train DT, RF, GB models. Return trained models + metrics dict."""
    # Build feature columns dynamically from available numeric and encoded features
    df = df.copy()
    # Drop identifier-like columns if any slipped through
    id_cols = [c for c in df.columns if ("id" in c and not c.endswith("_enc")) or c.endswith("_id")]
    if id_cols:
        df.drop(columns=id_cols, inplace=True, errors="ignore")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # remove target if present
    if "loan_status_enc" in num_cols:
        num_cols.remove("loan_status_enc")
    # include any *_enc columns (categorical encodings)
    enc_cols = [c for c in df.columns if c.endswith("_enc") and c != "loan_status_enc"]
    # prefer a stable ordering: numeric cols first then encoded cols
    feature_cols = [c for c in num_cols if c not in enc_cols] + enc_cols
    target_col = "loan_status_enc"

    X = df[feature_cols].fillna(0)
    y = df[target_col]

    # Quick importance pass to select top features and reduce overfitting
    rf_tmp = RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1, class_weight="balanced")
    rf_tmp.fit(X, y)
    importances = rf_tmp.feature_importances_
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    top_k = min(10, len(feature_cols))
    selected_features = [f for f, _ in feat_imp[:top_k]]

    # Use selected features for final training (reduces noise and overfitting)
    X = X[selected_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Decision Tree":      DecisionTreeClassifier(random_state=42, max_depth=6, class_weight="balanced"),
        "Random Forest":      RandomForestClassifier(n_estimators=200, random_state=42, max_depth=8, class_weight="balanced", n_jobs=-1),
        "Gradient Boosting":  GradientBoostingClassifier(n_estimators=200, random_state=42, learning_rate=0.05, max_depth=3)
    }

    results = {}
    trained  = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc  = accuracy_score(y_test,  model.predict(X_test))
        variance  = train_acc - test_acc
        results[name] = {
            "model":        model,
            "train_acc":    train_acc,
            "test_acc":     test_acc,
            "variance":     variance,
            "X_train":      X_train,
            "X_test":       X_test,
            "y_train":      y_train,
            "y_test":       y_test,
        }
        trained[name] = model
    best_model_name = max(results, key=lambda k: results[k]["test_acc"])

    # Return selected feature list instead of all features
    return results, trained, best_model_name, selected_features, X_train, X_test, y_train, y_test


def _safe_label_transform(encoders, key, value):
    """Transform a single label value using stored encoder dict.
    If the value wasn't seen during training, map to the encoder's default.
    """
    info = encoders.get(key)
    if info is None:
        raise KeyError(f"Encoder for '{key}' not found")
    le = info["encoder"]
    default = info.get("default", le.classes_[0])
    if value in le.classes_:
        return int(le.transform([value])[0])
    else:
        return int(le.transform([default])[0])


# ─────────────────────────────────────────────
# INTELLIGENT REASONING ENGINE
# ─────────────────────────────────────────────
def generate_reasoning(prediction_label, confidence, cibil_score, loan_income_ratio,
                        loan_term, income_annum, loan_amount,
                        residential_assets_value, commercial_assets_value,
                        luxury_assets_value, bank_asset_value):
    """Generate dynamic, intelligent reasons for loan decision."""
    total_assets = residential_assets_value + commercial_assets_value + luxury_assets_value + bank_asset_value
    asset_to_loan = total_assets / (loan_amount + 1)

    reasons_good = []
    reasons_bad  = []
    reasons_warn = []

    # CIBIL Score Analysis
    if cibil_score >= 750:
        reasons_good.append("✅ Excellent CIBIL score ({}) — signals strong credit discipline and repayment history.".format(cibil_score))
    elif cibil_score >= 650:
        reasons_warn.append("⚠️ Moderate CIBIL score ({}) — acceptable but leaves room for credit improvement.".format(cibil_score))
    else:
        reasons_bad.append("❌ Low CIBIL score ({}) — indicates poor credit history or past defaults.".format(cibil_score))

    # Loan-Income Ratio
    if loan_income_ratio < 3:
        reasons_good.append("✅ Healthy loan-income ratio ({:.2f}x) — loan is proportionate to income.".format(loan_income_ratio))
    elif loan_income_ratio < 5:
        reasons_warn.append("⚠️ Elevated loan-income ratio ({:.2f}x) — moderate repayment burden.".format(loan_income_ratio))
    else:
        reasons_bad.append("❌ High loan burden ({:.2f}x income) — loan amount significantly exceeds annual earnings.".format(loan_income_ratio))

    # Asset Coverage
    if asset_to_loan >= 2:
        reasons_good.append("✅ Strong asset coverage ({:.2f}x loan amount) — solid collateral backing.".format(asset_to_loan))
    elif asset_to_loan >= 1:
        reasons_warn.append("⚠️ Adequate asset coverage ({:.2f}x loan amount) — borderline collateral position.".format(asset_to_loan))
    else:
        reasons_bad.append("❌ Insufficient asset coverage ({:.2f}x loan amount) — weak collateral position.".format(asset_to_loan))

    # Loan Term Risk
    if loan_term <= 6:
        reasons_good.append("✅ Short loan term ({} years) — reduces long-term repayment risk.".format(loan_term))
    elif loan_term <= 12:
        reasons_warn.append("⚠️ Medium loan term ({} years) — manageable but requires consistent income.".format(loan_term))
    else:
        reasons_bad.append("❌ Long loan term ({} years) — increases exposure to income disruption and interest risk.".format(loan_term))

    # Income Level
    if income_annum >= 5000000:
        reasons_good.append("✅ High annual income (₹{:,.0f}) — strong repayment capacity.".format(income_annum))
    elif income_annum >= 2000000:
        reasons_warn.append("⚠️ Moderate annual income (₹{:,.0f}) — repayment feasible with discipline.".format(income_annum))
    else:
        reasons_bad.append("❌ Low annual income (₹{:,.0f}) — may struggle with EMI commitments.".format(income_annum))

    return reasons_good, reasons_bad, reasons_warn


# ─────────────────────────────────────────────
# VISUALIZATION HELPERS
# ─────────────────────────────────────────────
def set_plot_style():
    plt.rcParams.update({
        "figure.facecolor":  "#0f1829",
        "axes.facecolor":    "#0f1829",
        "axes.edgecolor":    "#1e2d42",
        "axes.labelcolor":   "#7a8ba0",
        "xtick.color":       "#7a8ba0",
        "ytick.color":       "#7a8ba0",
        "grid.color":        "#1e2d42",
        "grid.linestyle":    "--",
        "grid.alpha":        0.5,
        "text.color":        "#e8edf5",
        "font.family":       "monospace",
    })


def plot_feature_importance(model, feature_cols, model_name):
    set_plot_style()
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]
    sorted_feat = [feature_cols[i] for i in indices]
    sorted_imp  = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#f5c842" if i == 0 else "#4da6ff" if i < 3 else "#1e2d42" for i in range(len(sorted_feat))]
    bars = ax.barh(sorted_feat[::-1], sorted_imp[::-1], color=colors[::-1], height=0.65)
    ax.set_xlabel("Importance Score", fontsize=9)
    ax.set_title(f"Feature Importance — {model_name}", fontsize=10, color="#e8edf5", pad=12)
    ax.spines[["top","right","left","bottom"]].set_visible(False)
    ax.tick_params(labelsize=8)
    for bar, val in zip(bars, sorted_imp[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=7.5, color="#7a8ba0")
    plt.tight_layout()
    return fig


def plot_model_comparison(results):
    set_plot_style()
    names     = list(results.keys())
    train_acc = [results[n]["train_acc"]*100 for n in names]
    test_acc  = [results[n]["test_acc"]*100  for n in names]

    x   = np.arange(len(names))
    w   = 0.32
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - w/2, train_acc, w, label="Train Acc", color="#4da6ff",  alpha=0.85)
    b2 = ax.bar(x + w/2, test_acc,  w, label="Test Acc",  color="#f5c842",  alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(60, 105)
    ax.set_ylabel("Accuracy (%)", fontsize=9)
    ax.set_title("Model Performance Comparison", fontsize=10, color="#e8edf5", pad=12)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(fontsize=8, facecolor="#0f1829", edgecolor="#1e2d42")
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}%", ha="center", fontsize=7.5, color="#e8edf5")
    plt.tight_layout()
    return fig


def plot_approval_distribution(df):
    set_plot_style()
    counts = df["loan_status"].value_counts()
    colors = ["#00d4aa", "#ff4d6d"]
    fig, ax = plt.subplots(figsize=(5, 4))
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index,
        autopct="%1.1f%%", colors=colors,
        startangle=140, wedgeprops=dict(width=0.6, edgecolor="#090e1a", linewidth=3)
    )
    for t in texts:    t.set(color="#e8edf5", fontsize=9)
    for at in autotexts: at.set(color="#090e1a", fontsize=9, fontweight="bold")
    ax.set_title("Loan Approval Distribution", fontsize=10, color="#e8edf5", pad=12)
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df):
    set_plot_style()
    num_cols = ["no_of_dependents","income_annum","loan_amount","loan_term",
                "cibil_score","residential_assets_value","commercial_assets_value",
                "luxury_assets_value","bank_asset_value","loan_income_ratio","loan_status_enc"]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, ax=ax, annot=True, fmt=".2f", linewidths=0.4,
                cmap=sns.diverging_palette(220, 20, as_cmap=True),
                annot_kws={"size": 7}, linecolor="#0f1829",
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", fontsize=10, color="#e8edf5", pad=12)
    ax.tick_params(labelsize=7.5, rotation=30)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# LOAD DATA + TRAIN
# ─────────────────────────────────────────────
with st.spinner("🔄 Loading dataset and training models..."):
    df, encoders = load_and_preprocess()
    results, trained_models, best_model_name, feature_cols, X_train, X_test, y_train, y_test = train_models(df)

best_model    = results[best_model_name]["model"]
best_test_acc = results[best_model_name]["test_acc"]


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem 0;'>
        <div style='font-size:2.8rem;'>🏦</div>
        <div style='font-family:Syne,sans-serif; font-size:1.1rem; font-weight:800;
                    background:linear-gradient(135deg,#f5c842,#00d4aa);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    background-clip:text;'>LOAN AI</div>
        <div style='font-size:0.65rem; color:#7a8ba0; letter-spacing:0.15em; text-transform:uppercase;'>
            Prediction System v2.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🔮 Predict", "📊 Visualizations", "📋 Model Info"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Best model callout
    st.markdown(f"""
    <div style='background:rgba(245,200,66,0.07); border:1px solid rgba(245,200,66,0.2);
                border-radius:10px; padding:0.9rem 1rem; font-size:0.78rem;'>
        <div style='color:#7a8ba0; font-size:0.65rem; letter-spacing:0.12em; text-transform:uppercase;
                    margin-bottom:0.3rem;'>Best Model</div>
        <div style='color:#f5c842; font-weight:600; font-size:0.9rem;'>🏆 {best_model_name}</div>
        <div style='color:#e8edf5; margin-top:0.3rem;'>Accuracy: {best_test_acc*100:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#7a8ba0;'>
        <b style='color:#e8edf5;'>Models Trained:</b><br>
        • Decision Tree<br>
        • Random Forest<br>
        • Gradient Boosting<br><br>
        <b style='color:#e8edf5;'>Features:</b> 12 engineered<br>
        <b style='color:#e8edf5;'>Dataset:</b> loan_approval_dataset.csv
    </div>
    """, unsafe_allow_html=True)

components.html("""
<style>
    #restore-sidebar {
        width:100%;
        padding:0.85rem 1rem;
        border:none;
        border-radius:12px;
        background: linear-gradient(135deg, #f5c842, #f0a500);
        color: #0a0e1a;
        font-weight:800;
        font-size:0.95rem;
        cursor:pointer;
        box-shadow: 0 10px 28px rgba(0,0,0,0.12);
    }
    #restore-sidebar:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 32px rgba(0,0,0,0.2);
    }
    .restore-sidebar-wrap {
        margin:1rem 0;
        max-width:360px;
    }
</style>
<div class='restore-sidebar-wrap'>
    <button id='restore-sidebar'>Open sidebar</button>
</div>
<script>
(function() {
    const parentDoc = window.parent.document;
    const expandSidebar = () => {
        const selectors = [
            '[data-testid="collapsedSidebar"] button',
            'button[aria-label="Expand sidebar"]',
            'button[title*="Expand"]',
            'button[data-testid="collapsedSidebarButton"]',
            '[data-testid="stSidebar"] button'
        ];
        for (const sel of selectors) {
            const btn = parentDoc.querySelector(sel);
            if (btn) {
                btn.click();
                return;
            }
        }
        const sidebar = parentDoc.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {
            sidebar.style.position='relative';
            sidebar.style.left='0';
            sidebar.style.marginLeft='0';
            sidebar.style.transform='translateX(0%)';
            sidebar.style.width='320px';
            sidebar.style.visibility='visible';
            sidebar.style.opacity='1';
            sidebar.style.display='block';
            sidebar.style.zIndex='1000';
        }
    };
    const button = document.getElementById('restore-sidebar');
    if (button) {
        button.onclick = expandSidebar;
    }
})();
</script>
""", height=140, scrolling=False)


# ─────────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────────
if page == "🏠 Dashboard":
    st.markdown('<div class="hero-title">Loan Approval Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">AI/ML Decision Engine · Decision Tree · Random Forest · Gradient Boosting</div>', unsafe_allow_html=True)

    # ── Top Metric Cards ──
    total      = len(df)
    approved   = (df["loan_status"] == "Approved").sum()
    rejected   = total - approved
    appr_pct   = approved / total * 100

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Total Applications</div>
            <div class='metric-value'>{total:,}</div>
            <div class='metric-badge'>DATASET SIZE</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='metric-card' style='border-top-color:#00d4aa;'>
            <div class='metric-label'>Approved Loans</div>
            <div class='metric-value' style='color:#00d4aa;'>{approved:,}</div>
            <div class='metric-badge' style='background:rgba(0,212,170,0.1);color:#00d4aa;border-color:rgba(0,212,170,0.25);'>APPROVED</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='metric-card' style='border-top-color:#ff4d6d;'>
            <div class='metric-label'>Rejected Loans</div>
            <div class='metric-value' style='color:#ff4d6d;'>{rejected:,}</div>
            <div class='metric-badge' style='background:rgba(255,77,109,0.1);color:#ff4d6d;border-color:rgba(255,77,109,0.25);'>REJECTED</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class='metric-card' style='border-top-color:#4da6ff;'>
            <div class='metric-label'>Best Model Accuracy</div>
            <div class='metric-value' style='color:#4da6ff;'>{best_test_acc*100:.1f}%</div>
            <div class='metric-badge' style='background:rgba(77,166,255,0.1);color:#4da6ff;border-color:rgba(77,166,255,0.25);'>{best_model_name.upper()}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Dataset Preview ──
    st.markdown('<div class="section-header">🗃️ Dataset Preview</div>', unsafe_allow_html=True)
    with st.expander("View raw dataset (first 20 rows)", expanded=False):
        raw_cols = [c for c in df.columns if not c.endswith("_enc")]
        st.dataframe(
            df[raw_cols].head(20).style.background_gradient(cmap="Blues", subset=["cibil_score","income_annum"]),
            use_container_width=True
        )


# ─────────────────────────────────────────────
# PAGE: PREDICT
# ─────────────────────────────────────────────
elif page == "🔮 Predict":
    st.markdown('<div class="hero-title">Loan Eligibility Check</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Enter applicant details below · AI will assess in real-time</div>', unsafe_allow_html=True)

    with st.form("prediction_form"):
        st.markdown('<div class="section-header">👤 Applicant Profile</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            no_of_dependents = st.number_input("No. of Dependents", min_value=0, max_value=10, value=2, step=1)
            education        = st.selectbox("Education", ["Graduate", "Not Graduate"])
        with c2:
            self_employed    = st.selectbox("Self Employed", ["No", "Yes"])
            loan_term        = st.slider("Loan Term (Years)", min_value=1, max_value=30, value=10)
        with c3:
            cibil_score      = st.slider("CIBIL Score", min_value=300, max_value=900, value=650)
            st.markdown(f"<div style='font-size:0.75rem; color:#7a8ba0; margin-top:-0.5rem;'>Credit tier: {'🟢 Excellent' if cibil_score>=750 else '🟡 Fair' if cibil_score>=600 else '🔴 Poor'}</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-header">💰 Financial Details</div>', unsafe_allow_html=True)
        f1, f2 = st.columns(2)
        with f1:
            income_annum = st.number_input("Annual Income (₹)", min_value=0, value=5000000, step=100000,
                                           help="Gross annual income before tax")
            loan_amount  = st.number_input("Loan Amount Requested (₹)", min_value=0, value=15000000, step=100000)
        with f2:
            residential_assets_value = st.number_input("Residential Assets (₹)", min_value=0, value=5000000, step=100000)
            commercial_assets_value  = st.number_input("Commercial Assets (₹)", min_value=0, value=3000000, step=100000)

        a1, a2 = st.columns(2)
        with a1:
            luxury_assets_value  = st.number_input("Luxury Assets (₹)", min_value=0, value=2000000, step=100000)
        with a2:
            bank_asset_value     = st.number_input("Bank/Savings Assets (₹)", min_value=0, value=1000000, step=100000)

        # Auto-computed
        loan_income_ratio = loan_amount / (income_annum + 1) if income_annum > 0 else 0
        st.markdown(f"""
        <div class='info-box' style='margin-top:0.5rem;'>
            <strong>Auto-Computed:</strong> Loan-Income Ratio = {loan_income_ratio:.2f}x
            &nbsp;·&nbsp; Total Assets = ₹{(residential_assets_value+commercial_assets_value+luxury_assets_value+bank_asset_value):,}
        </div>""", unsafe_allow_html=True)

        submitted = st.form_submit_button("🔮 Predict Loan Status")

    # ── Validation + Prediction ──
    if submitted:
        errors = []
        if income_annum < 0:    errors.append("Annual income cannot be negative.")
        if loan_amount <= 0:    errors.append("Loan amount must be greater than zero.")
        if cibil_score < 300 or cibil_score > 900: errors.append("CIBIL score must be between 300-900.")

        if errors:
            for e in errors:
                st.error(f"⚠️ {e}")
        else:
            edu_enc  = _safe_label_transform(encoders, "education", education)
            self_enc = _safe_label_transform(encoders, "self_employed", self_employed)

            # Build input vector based on trained feature_cols
            form_map = {
                "no_of_dependents": no_of_dependents,
                "education": education,
                "self_employed": self_employed,
                "income_annum": income_annum,
                "loan_amount": loan_amount,
                "loan_term": loan_term,
                "cibil_score": cibil_score,
                "residential_assets_value": residential_assets_value,
                "commercial_assets_value": commercial_assets_value,
                "luxury_assets_value": luxury_assets_value,
                "bank_asset_value": bank_asset_value,
                "loan_income_ratio": loan_income_ratio
            }

            input_vals = []
            for feat in feature_cols:
                # encoded categorical
                if feat.endswith("_enc"):
                    orig = feat[:-4]
                    if orig in form_map:
                        v = _safe_label_transform(encoders, orig, form_map[orig])
                    else:
                        info = encoders.get(orig)
                        if info:
                            v = int(info["encoder"].transform([info["default"]])[0])
                        else:
                            v = 0
                    input_vals.append(v)
                else:
                    if feat in form_map:
                        input_vals.append(float(form_map[feat]))
                    else:
                        # fallback to dataset median when feature isn't part of the form
                        if feat in df.columns:
                            input_vals.append(float(df[feat].median()))
                        else:
                            input_vals.append(0.0)

            input_data = np.array([input_vals])

            prediction  = best_model.predict(input_data)[0]
            proba       = best_model.predict_proba(input_data)[0]
            pred_label  = encoders["loan_status"]["encoder"].inverse_transform([prediction])[0]
            confidence  = max(proba) * 100
            is_approved = pred_label.strip() == "Approved"


            # ── Intelligent Reasoning ──
            st.markdown('<div class="section-header">🧠 AI Reasoning & Analysis</div>', unsafe_allow_html=True)

            reasons_good, reasons_bad, reasons_warn = generate_reasoning(
                pred_label, confidence, cibil_score, loan_income_ratio,
                loan_term, income_annum, loan_amount,
                residential_assets_value, commercial_assets_value,
                luxury_assets_value, bank_asset_value
            )

            for r in reasons_good:
                st.markdown(f"<div class='reason-box reason-good'>{r}</div>", unsafe_allow_html=True)
            for r in reasons_warn:
                st.markdown(f"<div class='reason-box reason-warn'>{r}</div>", unsafe_allow_html=True)
            for r in reasons_bad:
                st.markdown(f"<div class='reason-box reason-bad'>{r}</div>", unsafe_allow_html=True)

            # Apply conservative override: if many strong negative reasons, force manual review/reject
            if len(reasons_bad) >= 3:
                is_approved = False

            # ── Summary Stats ──
            st.markdown('<div class="section-header">📈 Applicant Financial Snapshot</div>', unsafe_allow_html=True)
            s1, s2, s3, s4 = st.columns(4)
            total_assets = residential_assets_value + commercial_assets_value + luxury_assets_value + bank_asset_value
            for col, label, val, color in [
                (s1, "Annual Income",     f"₹{income_annum/1e5:.1f}L",         "#4da6ff"),
                (s2, "CIBIL Score",       str(cibil_score),                     "#f5c842" if cibil_score>=700 else "#ff4d6d"),
                (s3, "Loan/Income Ratio", f"{loan_income_ratio:.2f}x",          "#00d4aa" if loan_income_ratio<4 else "#ff4d6d"),
                (s4, "Total Assets",      f"₹{total_assets/1e5:.1f}L",          "#4da6ff"),
            ]:
                col.markdown(f"""
                <div class='metric-card' style='border-top-color:{color};'>
                    <div class='metric-label'>{label}</div>
                    <div class='metric-value' style='color:{color}; font-size:1.4rem;'>{val}</div>
                </div>""", unsafe_allow_html=True)

            # Final result header (after reasoning & overrides)
            final_label = 'Approved' if is_approved else 'Rejected'
            if is_approved:
                st.markdown(f"""
                <div class='result-approved'>
                    <div class='result-title'>✅ LOAN APPROVED</div>
                    <div style='color:#7a8ba0; font-size:0.85rem; margin-top:0.5rem;'>
                        Model: <strong style='color:#e8edf5;'>{best_model_name}</strong>
                        &nbsp;·&nbsp; Accuracy: <strong style='color:#e8edf5;'>{best_test_acc*100:.2f}%</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-rejected'>
                    <div class='result-title'>❌ LOAN REJECTED</div>
                    <div style='color:#7a8ba0; font-size:0.85rem; margin-top:0.5rem;'>
                        Model: <strong style='color:#e8edf5;'>{best_model_name}</strong>
                        &nbsp;·&nbsp; Accuracy: <strong style='color:#e8edf5;'>{best_test_acc*100:.2f}%</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Confidence ──
            st.markdown(f"""
            <div style='margin:1rem 0 0.3rem 0; font-size:0.8rem; color:#7a8ba0;'>
                Prediction Confidence: <strong style='color:#e8edf5;'>{confidence:.1f}%</strong>
            </div>
            <div class='conf-track'>
                <div class='{"conf-fill-good" if is_approved else "conf-fill-bad"}' style='width:{confidence}%;'></div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: VISUALIZATIONS
# ─────────────────────────────────────────────
elif page == "📊 Visualizations":
    st.markdown('<div class="hero-title">Data Visualizations</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Explore feature importance, distributions, and model metrics</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎯 Feature Importance", "🏆 Model Comparison", "🍩 Approval Distribution", "🔥 Correlation Heatmap"]
    )

    with tab1:
        st.markdown('<div class="section-header">🎯 Feature Importance</div>', unsafe_allow_html=True)
        model_choice = st.selectbox("Select Model", list(trained_models.keys()), key="fi_model")
        st.pyplot(plot_feature_importance(trained_models[model_choice], feature_cols, model_choice), use_container_width=True)
        st.markdown("""
        <div class='info-box'>
            <strong>Insight:</strong> CIBIL score and loan-income ratio are consistently the top drivers
            across all models — confirming that creditworthiness and debt burden are the primary determinants of loan approval.
        </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-header">🏆 Model Performance Comparison</div>', unsafe_allow_html=True)
        st.pyplot(plot_model_comparison(results), use_container_width=True)
        st.markdown(f"""
        <div class='info-box'>
            <strong>Best Model:</strong> {best_model_name} achieved {best_test_acc*100:.2f}% test accuracy.
            Random Forest and Gradient Boosting typically outperform single Decision Trees due to ensemble learning —
            averaging over many trees reduces variance and overfitting.
        </div>""", unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="section-header">🍩 Loan Approval Distribution</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns([1, 1.2])
        with col_a:
            st.pyplot(plot_approval_distribution(df), use_container_width=True)
        with col_b:
            counts = df["loan_status"].value_counts()
            for status, count in counts.items():
                pct   = count / len(df) * 100
                color = "#00d4aa" if "Approved" in status else "#ff4d6d"
                st.markdown(f"""
                <div class='metric-card' style='border-top-color:{color}; margin-bottom:1rem;'>
                    <div class='metric-label'>{status}</div>
                    <div class='metric-value' style='color:{color};'>{count:,}</div>
                    <div class='conf-track' style='margin-top:0.5rem;'>
                        <div style='width:{pct}%; background:{color}; height:100%; border-radius:8px;'></div>
                    </div>
                    <div style='font-size:0.75rem; color:#7a8ba0; margin-top:0.3rem;'>{pct:.1f}% of total</div>
                </div>""", unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="section-header">🔥 Feature Correlation Heatmap</div>', unsafe_allow_html=True)
        st.pyplot(plot_correlation_heatmap(df), use_container_width=True)
        st.markdown("""
        <div class='info-box'>
            <strong>Key Observations:</strong>
            CIBIL score shows strong positive correlation with loan approval. Income, asset values,
            and loan amount are naturally correlated. The engineered loan_income_ratio captures
            the relative debt burden — a critical risk signal.
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: MODEL INFO
# ─────────────────────────────────────────────
elif page == "📋 Model Info":
    st.markdown('<div class="hero-title">Project Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Architecture · Methodology · Feature Engineering</div>', unsafe_allow_html=True)

    # Dataset
    st.markdown('<div class="section-header">🗃️ Dataset</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        <strong>Source:</strong> loan_approval_dataset.csv<br>
        <strong>Target Variable:</strong> loan_status (Approved / Rejected)<br>
        <strong>Features:</strong> 13 raw columns → 12 model features after engineering<br>
        <strong>Preprocessing:</strong> Label Encoding for categorical variables (education, self_employed, loan_status),
        removal of loan_id (non-predictive ID), and null value elimination.
    </div>""", unsafe_allow_html=True)

    # Feature Engineering
    st.markdown('<div class="section-header">⚙️ Feature Engineering</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        <strong>loan_income_ratio</strong> = loan_amount ÷ income_annum<br><br>
        This engineered feature captures the relative debt burden — a borrower requesting 10× their annual
        income is far riskier than one requesting 2×. This ratio is consistently one of the top predictors
        across all three models, validating the engineering decision.
    </div>""", unsafe_allow_html=True)

    # Models
    st.markdown('<div class="section-header">🤖 Models Used</div>', unsafe_allow_html=True)
    for model_name, description, color in [
        ("Decision Tree", "A single tree that recursively splits data on the feature yielding the highest information gain. Interpretable but prone to overfitting on deep trees. Configured with max_depth=8.", "#4da6ff"),
        ("Random Forest", "An ensemble of 150 decision trees trained on random subsets (bagging). Aggregates votes to reduce variance and improve generalization. Typically the most robust performer.", "#f5c842"),
        ("Gradient Boosting", "Sequential ensemble where each tree corrects the errors of the previous one (boosting). Uses learning_rate=0.1 with 150 estimators for a smooth bias-variance trade-off.", "#00d4aa"),
    ]:
        train_a = results[model_name]["train_acc"] * 100
        test_a  = results[model_name]["test_acc"]  * 100
        var     = results[model_name]["variance"]   * 100
        tag     = "🏆 BEST MODEL" if model_name == best_model_name else ""
        st.markdown(f"""
        <div class='info-box' style='border-left:3px solid {color}; margin-bottom:0.8rem;'>
            <strong style='color:{color};'>{model_name}</strong>
            {'&nbsp;<span style="background:rgba(245,200,66,0.15); color:#f5c842; font-size:0.7rem; padding:2px 8px; border-radius:4px; border:1px solid rgba(245,200,66,0.3);">' + tag + '</span>' if tag else ''}<br>
            {description}<br><br>
            Train Acc: <strong>{train_a:.2f}%</strong> &nbsp;·&nbsp;
            Test Acc: <strong>{test_a:.2f}%</strong> &nbsp;·&nbsp;
            Variance: <strong style='color:{"#ff4d6d" if var > 10 else "#f5c842" if var > 5 else "#00d4aa"}'>{var:.2f}%</strong>
        </div>""", unsafe_allow_html=True)

    # Variance Analysis
    st.markdown('<div class="section-header">📉 Variance Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        <strong>Variance</strong> = Train Accuracy − Test Accuracy<br><br>
        • <strong style='color:#00d4aa;'>Low variance (&lt;5%)</strong>: Model generalizes well — not overfitting.<br>
        • <strong style='color:#f5c842;'>Medium variance (5-10%)</strong>: Slight overfitting — acceptable in practice.<br>
        • <strong style='color:#ff4d6d;'>High variance (&gt;10%)</strong>: Model memorized training data — poor generalization.<br><br>
        Random Forest and Gradient Boosting inherently reduce variance through ensemble averaging, making them
        preferred over a single Decision Tree for real-world deployment.
    </div>""", unsafe_allow_html=True)

    # Pipeline
    st.markdown('<div class="section-header">🔧 Preprocessing Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        <strong>1. Load CSV</strong> → strip whitespace from columns and string values<br>
        <strong>2. Drop loan_id</strong> → non-predictive identifier removed<br>
        <strong>3. Drop nulls</strong> → ensures clean data for all models<br>
        <strong>4. Engineer loan_income_ratio</strong> → loan_amount ÷ income_annum<br>
        <strong>5. Label Encode</strong> → education, self_employed → numeric<br>
        <strong>6. Train-Test Split</strong> → 80% train / 20% test (stratified by loan_status)<br>
        <strong>7. Fit Models</strong> → DT, RF, GB trained on training set<br>
        <strong>8. Evaluate</strong> → Compare test accuracy, select best model automatically
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class='footer'>
    🏦 Loan Approval Prediction System &nbsp;·&nbsp; ML Dashboard v2.0<br>
    Decision Tree &nbsp;|&nbsp; Random Forest &nbsp;|&nbsp; Gradient Boosting<br><br>
    Developed by <span>SRI VARUN SINGARI</span>
</div>
""", unsafe_allow_html=True)
