"""
Streamlit dashboard for the Predictive Maintenance project.

Run from the project root:

    streamlit run app/dashboard.py

What the dashboard shows
------------------------
1. Overview          - frozen test-set results of the final model
2. Predict a machine - enter sensor readings, get a failure probability
3. Batch scoring     - upload a CSV, score every row, download the results
4. Model insights    - feature importance and a threshold explorer
5. Data exploration  - the saved EDA charts from reports/figures/

The dashboard never retrains or retunes the model on its own. It uses the
saved model (models/random_forest_final.joblib) and the frozen threshold
stored in models/model_metadata.json. The threshold explorer is for
understanding the trade-off only - the frozen threshold stays 0.30.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths: make `src` importable and run everything from the project root,
# so the relative paths used in src/ and scripts/ keep working.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.data_processing import (  # noqa: E402
    TARGET_COLUMN,
    clean_and_encode,
    load_and_prepare_data,
)
from src.features import add_engineered_features  # noqa: E402
from src.predict import METADATA_PATH, MODEL_PATH, predict  # noqa: E402

DATA_PATH = Path("data/raw/ai4i2020.csv")
FIGURES_DIR = Path("reports/figures")

# Raw sensor columns a user must provide (names match the AI4I 2020 CSV)
SENSOR_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TYPE_VALUES = ["H", "L", "M"]
TYPE_COLUMNS = [f"Type_{t}" for t in TYPE_VALUES]

# Display-only band below the frozen threshold, used to flag machines that
# are "worth watching". It does NOT change the model's prediction.
WATCH_FRACTION = 0.5

# Colours (reference data-viz palette). Status colours always appear with a
# text label, never on their own.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_CRITICAL = "#d03b3b"
GRID = "rgba(128,128,128,0.18)"


# ===========================================================================
# Helper functions (pure Python, no Streamlit - these are unit-tested)
# ===========================================================================
def prepare_input(df):
    """
    Turn user data into the cleaned/encoded format that src.predict expects.

    Accepts either:
    - raw AI4I-style rows with a `Type` column (L / M / H), or
    - already-encoded rows with Type_H / Type_L / Type_M columns.

    Extra columns (UDI, Product ID, Machine failure, TWF, ...) are ignored.

    Returns
    -------
    pd.DataFrame with the 5 sensor columns + Type_H, Type_L, Type_M.

    Raises
    ------
    ValueError if required columns are missing or Type has unknown values.
    """
    missing = [c for c in SENSOR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    out = df[SENSOR_COLUMNS].copy()

    if "Type" in df.columns:
        types = df["Type"].astype(str).str.strip().str.upper()
        unknown = sorted(set(types) - set(TYPE_VALUES))
        if unknown:
            raise ValueError(
                f"Unknown Type value(s): {', '.join(unknown)} "
                f"(expected L, M or H)"
            )
        # Build all three dummy columns explicitly, so a file that only
        # contains "L" machines still gets Type_H and Type_M columns.
        for t in TYPE_VALUES:
            out[f"Type_{t}"] = types == t
    elif all(c in df.columns for c in TYPE_COLUMNS):
        for c in TYPE_COLUMNS:
            out[c] = df[c].astype(bool)
    else:
        raise ValueError(
            "Provide a `Type` column (L/M/H) or the columns "
            "Type_H, Type_L, Type_M."
        )

    bad = out[SENSOR_COLUMNS].apply(pd.to_numeric, errors="coerce").isna()
    if bad.any().any():
        cols = [c for c in SENSOR_COLUMNS if bad[c].any()]
        raise ValueError(f"Non-numeric or empty values in: {', '.join(cols)}")
    out[SENSOR_COLUMNS] = out[SENSOR_COLUMNS].apply(pd.to_numeric)
    return out


def risk_level(probability, threshold):
    """
    Map a failure probability to a display label.

    - "Failure predicted": probability >= threshold (the model's decision)
    - "Watch":             probability >= threshold * WATCH_FRACTION
    - "Normal":            everything below that
    """
    if probability >= threshold:
        return "Failure predicted"
    if probability >= threshold * WATCH_FRACTION:
        return "Watch"
    return "Normal"


def score_dataframe(df, model, metadata):
    """Prepare, predict and add probability / prediction / risk columns."""
    X = prepare_input(df)
    result = predict(X, model=model, metadata=metadata)
    threshold = result["threshold"]
    scored = df.copy()
    scored["failure_probability"] = np.round(result["probability"], 4)
    scored["predicted_failure"] = result["prediction"]
    scored["risk_level"] = [
        risk_level(p, threshold) for p in result["probability"]
    ]
    return scored


def metrics_at_threshold(y_true, y_proba, threshold):
    """Precision, recall, F1 and confusion counts at one threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {"threshold": threshold, "precision": precision, "recall": recall,
            "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def threshold_sweep(y_true, y_proba, thresholds=None):
    """metrics_at_threshold for a range of thresholds, as a DataFrame."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    return pd.DataFrame(
        [metrics_at_threshold(y_true, y_proba, t) for t in thresholds]
    )


# ===========================================================================
# Streamlit app
# ===========================================================================
def main():
    import plotly.graph_objects as go
    import streamlit as st

    from src.predict import load_metadata, load_model

    st.set_page_config(
        page_title="Predictive Maintenance Dashboard",
        page_icon="🛠️",
        layout="wide",
    )

    # ---------------- cached loaders ----------------
    @st.cache_resource(show_spinner="Loading model...")
    def get_model():
        return load_model(MODEL_PATH)

    @st.cache_data
    def get_metadata():
        return load_metadata(METADATA_PATH)

    @st.cache_data(show_spinner="Scoring the held-out test set...")
    def get_test_scores(_model, feature_names):
        """Probabilities on the frozen test split (needs the raw CSV)."""
        _, _, X_test, _, _, y_test = load_and_prepare_data(str(DATA_PATH))
        X_test = add_engineered_features(X_test)[feature_names]
        proba = _model.predict_proba(X_test)[:, 1]
        return X_test, y_test.to_numpy(), proba

    @st.cache_data
    def get_raw_data():
        return pd.read_csv(DATA_PATH)

    def base_layout(fig, height=340):
        fig.update_layout(
            height=height,
            margin=dict(l=10, r=10, t=30, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            hoverlabel=dict(font_size=13),
        )
        fig.update_xaxes(gridcolor=GRID, zeroline=False)
        fig.update_yaxes(gridcolor=GRID, zeroline=False)
        return fig

    def status_badge(label):
        icon, colour = {
            "Failure predicted": ("⛔", STATUS_CRITICAL),
            "Watch": ("⚠️", STATUS_WARNING),
            "Normal": ("✅", STATUS_GOOD),
        }[label]
        st.markdown(
            f"<div style='padding:14px 18px;border-radius:10px;"
            f"border-left:6px solid {colour};background:rgba(128,128,128,0.08);"
            f"font-size:1.25rem;font-weight:600'>{icon} {label}</div>",
            unsafe_allow_html=True,
        )

    # ---------------- header & sidebar ----------------
    st.title("🛠️ Predictive Maintenance Dashboard")
    st.caption(
        "Class-weighted Random Forest trained on the AI4I 2020 synthetic "
        "benchmark. Educational portfolio project - results show methodology, "
        "not guaranteed performance in a real factory."
    )

    metadata = get_metadata() if Path(METADATA_PATH).exists() else None
    model = None
    model_error = None
    if Path(MODEL_PATH).exists():
        try:
            model = get_model()
        except Exception as exc:  # e.g. scikit-learn version mismatch
            model_error = str(exc)

    data_available = DATA_PATH.exists()

    with st.sidebar:
        st.header("Status")
        st.write(("✅" if model is not None else "❌") + " Trained model")
        st.write(("✅" if metadata is not None else "❌") + " Model metadata")
        st.write(("✅" if data_available else "➖") + " Raw dataset "
                 "(optional, for test-set charts)")

        if metadata is not None:
            st.divider()
            st.subheader("Frozen model")
            st.write(f"**Type:** {metadata['model_type']}")
            st.write(f"**Trees:** {metadata['model_params']['n_estimators']}")
            st.write(f"**Threshold:** {metadata['threshold']:.2f}")
            st.write(f"**Features:** {metadata['n_features']}")
            st.write(f"**Trained on:** {metadata['trained_on_rows']:,} rows")

        if model is None:
            st.divider()
            st.subheader("Train the model")
            if model_error:
                st.error(f"Could not load the model: {model_error}")
            if data_available:
                st.write("The dataset was found. Train and save the frozen "
                         "model (the safety check against the frozen test "
                         "results still applies).")
                if st.button("Train & save model", type="primary"):
                    from scripts.save_final_model import main as save_main
                    with st.spinner("Training..."):
                        try:
                            save_main()
                        except SystemExit as exc:
                            st.error(str(exc))
                        else:
                            st.cache_resource.clear()
                            st.cache_data.clear()
                            st.rerun()
            else:
                st.info(f"Put the AI4I 2020 CSV at `{DATA_PATH}`, then run "
                        "`python scripts/save_final_model.py`.")

    if metadata is None:
        st.error(f"`{METADATA_PATH}` was not found - the dashboard needs it.")
        st.stop()

    threshold = float(metadata["threshold"])
    frozen = metadata["frozen_test_results"]
    feature_names = metadata["feature_names"]

    tab_overview, tab_single, tab_batch, tab_insights, tab_eda = st.tabs(
        ["📊 Overview", "🔧 Predict a machine", "📁 Batch scoring",
         "🧠 Model insights", "🔍 Data exploration"]
    )

    # =====================================================================
    # 1. OVERVIEW
    # =====================================================================
    with tab_overview:
        st.subheader("Frozen test-set results")
        st.caption(f"Evaluated once on {frozen['total_rows']:,} untouched "
                   f"test rows at threshold {threshold:.2f}. Never re-tuned.")

        cols = st.columns(5)
        for col, (label, key) in zip(cols, [
            ("Precision", "precision"), ("Recall", "recall"), ("F1", "f1"),
            ("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"),
        ]):
            col.metric(label, f"{frozen[key]:.3f}")

        total_failures = frozen["failures_detected"] + frozen["failures_missed"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Failures detected",
                  f"{frozen['failures_detected']} / {total_failures}")
        c2.metric("Failures missed", frozen["failures_missed"])
        c3.metric("False alarms", frozen["false_alarms"])

        left, right = st.columns([1, 1])
        with left:
            st.markdown("**Confusion matrix (test set)**")
            tn = (frozen["total_rows"] - total_failures
                  - frozen["false_alarms"])
            z = [[tn, frozen["false_alarms"]],
                 [frozen["failures_missed"], frozen["failures_detected"]]]
            text = [[f"TN<br><b>{z[0][0]:,}</b>", f"FP<br><b>{z[0][1]}</b>"],
                    [f"FN<br><b>{z[1][0]}</b>", f"TP<br><b>{z[1][1]}</b>"]]
            fig = go.Figure(go.Heatmap(
                z=np.log10(np.array(z) + 1),  # log so small cells stay visible
                x=["Predicted: no failure", "Predicted: failure"],
                y=["Actual: no failure", "Actual: failure"],
                text=text, texttemplate="%{text}", textfont=dict(size=16),
                colorscale=[[0, "#cde2fb"], [1, "#1c5cab"]],
                showscale=False, xgap=2, ygap=2,
                customdata=z,
                hovertemplate="%{y}<br>%{x}<br>Count: %{customdata:,}"
                              "<extra></extra>",
            ))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(base_layout(fig, 320), width="stretch")
        with right:
            st.markdown("**What this means**")
            st.write(
                f"- Out of **{total_failures}** real failures in the test set, "
                f"the model caught **{frozen['failures_detected']}** and missed "
                f"**{frozen['failures_missed']}**.\n"
                f"- It raised **{frozen['false_alarms']}** false alarms across "
                f"{tn + frozen['false_alarms']:,} healthy machines.\n"
                f"- A machine is flagged when its failure probability is "
                f"**≥ {threshold:.2f}**. A lower threshold than 0.5 was chosen "
                f"on the validation set because missing a failure is usually "
                f"more costly than an extra inspection.\n"
                f"- The dataset is synthetic, so these numbers show the method "
                f"works - not how it would perform in a real plant."
            )

    # =====================================================================
    # 2. SINGLE PREDICTION
    # =====================================================================
    with tab_single:
        if model is None:
            st.warning("Train the model first (see the sidebar).")
        else:
            st.subheader("Enter the machine's current readings")
            c1, c2, c3 = st.columns(3)
            with c1:
                m_type = st.selectbox(
                    "Product quality type", ["L", "M", "H"],
                    help="L = low, M = medium, H = high quality variant")
                air = st.number_input("Air temperature [K]", 290.0, 310.0,
                                      300.0, 0.1)
            with c2:
                process = st.number_input("Process temperature [K]", 300.0,
                                          320.0, 310.0, 0.1)
                rpm = st.number_input("Rotational speed [rpm]", 1000, 3000,
                                      1500, 10)
            with c3:
                torque = st.number_input("Torque [Nm]", 0.0, 80.0, 40.0, 0.5)
                wear = st.number_input("Tool wear [min]", 0, 260, 100, 1)

            row = pd.DataFrame([{
                "Type": m_type,
                "Air temperature [K]": air,
                "Process temperature [K]": process,
                "Rotational speed [rpm]": rpm,
                "Torque [Nm]": torque,
                "Tool wear [min]": wear,
            }])
            scored = score_dataframe(row, model, metadata)
            prob = float(scored["failure_probability"].iloc[0])
            label = scored["risk_level"].iloc[0]

            st.divider()
            r1, r2 = st.columns([1, 2])
            with r1:
                st.metric("Failure probability", f"{prob:.1%}")
                status_badge(label)
                st.caption(f"Decision threshold: {threshold:.0%}. "
                           f"'Watch' = above {threshold * WATCH_FRACTION:.0%} "
                           f"(display only).")
                feats = add_engineered_features(prepare_input(row)).iloc[0]
                st.markdown("**Engineered features**")
                st.write(f"- Power: {feats['Power']:,.0f} W")
                st.write(f"- Temp diff: {feats['Temp_diff']:.1f} K")
                st.write(f"- Wear × Torque: {feats['Wear_Torque']:,.0f}")
            with r2:
                st.markdown("**What if one reading changes?**")
                what = st.selectbox(
                    "Vary this reading (others stay fixed)",
                    ["Tool wear [min]", "Torque [Nm]",
                     "Rotational speed [rpm]", "Process temperature [K]"])
                ranges = {
                    "Tool wear [min]": np.arange(0, 255, 5),
                    "Torque [Nm]": np.arange(5, 76, 1.0),
                    "Rotational speed [rpm]": np.arange(1150, 2900, 25),
                    "Process temperature [K]": np.arange(305, 314.1, 0.2),
                }
                grid = pd.concat([row] * len(ranges[what]), ignore_index=True)
                grid[what] = ranges[what]
                curve = predict(prepare_input(grid), model=model,
                                metadata=metadata)["probability"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=grid[what], y=curve, mode="lines",
                    line=dict(color=BLUE, width=2), name="Probability",
                    hovertemplate=f"{what}: %{{x}}<br>"
                                  "Probability: %{y:.1%}<extra></extra>"))
                fig.add_hline(y=threshold, line=dict(color=STATUS_CRITICAL,
                              width=1.5, dash="dash"),
                              annotation_text=f"threshold {threshold:.2f}",
                              annotation_position="top left")
                fig.add_trace(go.Scatter(
                    x=[row[what].iloc[0]], y=[prob], mode="markers",
                    marker=dict(size=11, color=BLUE,
                                line=dict(color="white", width=2)),
                    name="Current", hovertemplate="Current reading<br>"
                    "Probability: %{y:.1%}<extra></extra>"))
                fig.update_layout(showlegend=False, hovermode="x unified")
                fig.update_yaxes(title="Failure probability",
                                 tickformat=".0%", range=[0, 1.02])
                fig.update_xaxes(title=what)
                st.plotly_chart(base_layout(fig, 360),
                                width="stretch")

    # =====================================================================
    # 3. BATCH SCORING
    # =====================================================================
    with tab_batch:
        if model is None:
            st.warning("Train the model first (see the sidebar).")
        else:
            st.subheader("Score many machines at once")
            st.write(
                "Upload a CSV with the columns "
                + ", ".join(f"`{c}`" for c in SENSOR_COLUMNS)
                + " and either `Type` (L/M/H) or `Type_H`, `Type_L`, `Type_M`. "
                "The original AI4I 2020 CSV works as-is."
            )
            template = pd.DataFrame([
                {"Machine ID": "M-001", "Type": "L",
                 "Air temperature [K]": 298.1, "Process temperature [K]": 308.6,
                 "Rotational speed [rpm]": 1551, "Torque [Nm]": 42.8,
                 "Tool wear [min]": 0},
                {"Machine ID": "M-002", "Type": "M",
                 "Air temperature [K]": 302.5, "Process temperature [K]": 310.9,
                 "Rotational speed [rpm]": 1320, "Torque [Nm]": 62.0,
                 "Tool wear [min]": 215},
            ])
            st.download_button("Download a template CSV",
                               template.to_csv(index=False),
                               "machines_template.csv", "text/csv")

            uploaded = st.file_uploader("CSV file", type=["csv"])
            if uploaded is not None:
                try:
                    df_in = pd.read_csv(uploaded)
                    scored = score_dataframe(df_in, model, metadata)
                except Exception as exc:
                    st.error(f"Could not score this file: {exc}")
                else:
                    counts = scored["risk_level"].value_counts()
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Machines scored", f"{len(scored):,}")
                    c2.metric("⛔ Failure predicted",
                              int(counts.get("Failure predicted", 0)))
                    c3.metric("⚠️ Watch", int(counts.get("Watch", 0)))
                    c4.metric("✅ Normal", int(counts.get("Normal", 0)))

                    show_only = st.checkbox("Show only flagged machines "
                                            "(Failure predicted or Watch)")
                    view = scored.sort_values("failure_probability",
                                              ascending=False)
                    if show_only:
                        view = view[view["risk_level"] != "Normal"]
                    st.dataframe(
                        view, width="stretch", hide_index=True,
                        column_config={
                            "failure_probability": st.column_config
                            .ProgressColumn("Failure probability",
                                            format="%.2f", min_value=0.0,
                                            max_value=1.0),
                        },
                    )
                    st.download_button(
                        "Download scored CSV",
                        scored.to_csv(index=False),
                        "scored_machines.csv", "text/csv", type="primary")

                    if TARGET_COLUMN in scored.columns:
                        m = metrics_at_threshold(scored[TARGET_COLUMN],
                                                 scored["failure_probability"],
                                                 threshold)
                        st.info(
                            f"This file has a `{TARGET_COLUMN}` column, so "
                            f"here is how the model did on it: precision "
                            f"{m['precision']:.3f}, recall {m['recall']:.3f}, "
                            f"F1 {m['f1']:.3f}. If the file contains training "
                            f"rows, these numbers are optimistic - the frozen "
                            f"test results on the Overview tab are the honest "
                            f"ones.")

    # =====================================================================
    # 4. MODEL INSIGHTS
    # =====================================================================
    with tab_insights:
        if model is None:
            st.warning("Train the model first (see the sidebar).")
        else:
            st.subheader("Which readings matter most?")
            imp = (pd.Series(model.feature_importances_, index=feature_names)
                   .sort_values())
            fig = go.Figure(go.Bar(
                x=imp.values, y=imp.index, orientation="h",
                marker=dict(color=BLUE, cornerradius=4),
                hovertemplate="%{y}<br>Importance: %{x:.3f}<extra></extra>"))
            fig.update_xaxes(title="Mean decrease in impurity")
            fig.update_layout(bargap=0.35)
            st.plotly_chart(base_layout(fig, 380), width="stretch")
            st.caption("Random Forest impurity-based importance. Engineered "
                       "features (Power, Temp_diff, Wear_Torque) are included.")

            st.divider()
            st.subheader("Threshold explorer")
            if not data_available:
                st.info(f"Add the dataset at `{DATA_PATH}` to explore how "
                        "precision and recall change with the threshold.")
            else:
                st.caption(
                    "For understanding only. The frozen threshold was chosen "
                    "on the validation set and stays at "
                    f"{threshold:.2f}; this chart uses the test set, so it must "
                    "not be used to pick a new threshold.")
                _, y_test, proba = get_test_scores(model, feature_names)
                sweep = threshold_sweep(y_test, proba)
                fig = go.Figure()
                for name, colour in [("precision", BLUE), ("recall", ORANGE),
                                     ("f1", AQUA)]:
                    fig.add_trace(go.Scatter(
                        x=sweep["threshold"], y=sweep[name], mode="lines",
                        name=name.capitalize() if name != "f1" else "F1",
                        line=dict(color=colour, width=2),
                        hovertemplate="%{y:.3f}"))
                fig.add_vline(x=threshold, line=dict(color="gray", width=1.5,
                              dash="dash"),
                              annotation_text=f"frozen {threshold:.2f}")
                fig.update_layout(hovermode="x unified",
                                  legend=dict(orientation="h", y=1.12, x=0))
                fig.update_xaxes(title="Decision threshold")
                fig.update_yaxes(title="Score", range=[0, 1.02])
                st.plotly_chart(base_layout(fig, 380),
                                width="stretch")

                t = st.slider("Try a threshold", 0.05, 0.95, threshold, 0.05)
                m = metrics_at_threshold(y_test, proba, t)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Failures caught", f"{m['tp']} / {m['tp'] + m['fn']}")
                c2.metric("Failures missed", m["fn"])
                c3.metric("False alarms", m["fp"])
                c4.metric("F1", f"{m['f1']:.3f}")

                st.markdown("**Probability distribution on the test set**")
                fig = go.Figure()
                for cls, name, colour in [(0, "Actual: no failure", BLUE),
                                          (1, "Actual: failure", ORANGE)]:
                    fig.add_trace(go.Histogram(
                        x=proba[y_test == cls], name=name,
                        marker=dict(color=colour,
                                    line=dict(color="white", width=1)),
                        xbins=dict(start=0, end=1, size=0.05),
                        opacity=0.85,
                        hovertemplate="Probability %{x}<br>Machines: %{y}"
                                      "<extra>" + name + "</extra>"))
                fig.add_vline(x=threshold, line=dict(color="gray", width=1.5,
                              dash="dash"))
                fig.update_layout(barmode="overlay",
                                  legend=dict(orientation="h", y=1.12, x=0))
                fig.update_yaxes(title="Machines (log scale)", type="log")
                fig.update_xaxes(title="Predicted failure probability")
                st.plotly_chart(base_layout(fig, 340),
                                width="stretch")

    # =====================================================================
    # 5. DATA EXPLORATION
    # =====================================================================
    with tab_eda:
        st.subheader("Exploratory data analysis")
        if data_available:
            raw = get_raw_data()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{len(raw):,}")
            c2.metric("Columns", raw.shape[1])
            c3.metric("Failures", int(raw[TARGET_COLUMN].sum()))
            c4.metric("Failure rate", f"{raw[TARGET_COLUMN].mean():.2%}")

            modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
            names = {"TWF": "Tool wear", "HDF": "Heat dissipation",
                     "PWF": "Power", "OSF": "Overstrain", "RNF": "Random"}
            if all(m in raw.columns for m in modes):
                counts = raw[modes].sum().sort_values()
                fig = go.Figure(go.Bar(
                    x=counts.values, y=[names[m] + f" ({m})" for m in
                                        counts.index],
                    orientation="h", marker=dict(color=BLUE, cornerradius=4),
                    text=counts.values, textposition="outside",
                    hovertemplate="%{y}<br>Rows: %{x}<extra></extra>"))
                fig.update_xaxes(title="Rows with this failure mode")
                st.markdown("**Failure modes** (not used as model inputs - "
                            "they would leak the answer)")
                st.plotly_chart(base_layout(fig, 280),
                                width="stretch")

            with st.expander("Preview raw data"):
                st.dataframe(raw.head(200), width="stretch",
                             hide_index=True)
            # clean_and_encode is the same step used for training
            with st.expander("Preview cleaned & encoded data"):
                st.dataframe(clean_and_encode(raw).head(200),
                             width="stretch", hide_index=True)

        figures = sorted(FIGURES_DIR.glob("*.png"))
        if figures:
            st.markdown("**Saved EDA charts** (from the exploration notebook)")
            for i in range(0, len(figures), 2):
                cols = st.columns(2)
                for col, f in zip(cols, figures[i:i + 2]):
                    col.image(str(f),
                              caption=f.stem.replace("_", " ").capitalize(),
                              width="stretch")
        elif not data_available:
            st.info("No EDA figures or dataset found.")


if __name__ == "__main__":
    main()
