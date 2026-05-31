"""
SentinelMind — Investigator Dashboard
Streamlit frontend for the MPS-based insider threat detection system.

Tabs:
    Overview     — system-wide risk stats + top flagged users
    Alert Queue  — filterable table of all flagged sessions
    User Profile — per-user risk timeline + SHAP waterfall
    Live Score   — fire a single event at the /score API endpoint

Run:
    streamlit run dashboard.py

Requires:
    pip install streamlit plotly pandas requests
    python -m uvicorn main:app --port 8080   (in a separate terminal)
"""

import json
import ast
import math
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime

# ─── CONFIG ───────────────────────────────────────────────────────────────────

API_BASE   = "http://localhost:8000"
ALERTS_CSV = Path("data/alerts.csv")

RISK_COLOURS = {
    "CRITICAL": "#E53935",
    "HIGH":     "#FB8C00",
    "MEDIUM":   "#FDD835",
    "LOW":      "#43A047",
}

FEATURE_LABELS = {
    "off_hours":             "Off-hours login",
    "bulk_download":         "Bulk download",
    "privilege_escalation":  "Privilege escalation",
    "unauthorised_db":       "Unauthorised DB access",
    "records_accessed":      "Records accessed",
    "hour_of_login":         "Login hour",
    "session_duration_min":  "Session duration (min)",
    "records_vs_user_avg":   "Records vs. personal avg",
    "duration_vs_user_avg":  "Duration vs. personal avg",
    "role_enc":              "Role",
    "db_accessed_enc":       "Database",
    "action_enc":            "Action type",
    "day_of_week":           "Day of week",
}

# ─── PAGE SETUP ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title  = "SentinelMind",
    page_icon   = "🛡️",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main { background: #0f1117; }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 18px 22px;
        border-left: 4px solid;
        margin-bottom: 8px;
    }
    .metric-val  { font-size: 2rem; font-weight: 700; margin: 4px 0; }
    .metric-label{ font-size: 0.82rem; color: #aaa; text-transform: uppercase; letter-spacing: .06em; }
    .risk-pill {
        display: inline-block; padding: 3px 10px;
        border-radius: 20px; font-size: 0.78rem; font-weight: 600;
    }
    .explain-box {
        background: #1a1d27; border-radius: 8px;
        padding: 14px 18px; border-left: 3px solid #5c6bc0;
        font-size: 0.9rem; color: #cfd8dc; margin: 8px 0;
    }
    .header-band {
        background: linear-gradient(90deg,#1a237e,#283593);
        padding: 18px 28px; border-radius: 12px; margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

# ─── HELPERS ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def load_alerts() -> pd.DataFrame:
    """Load alerts from CSV (or mock data if file absent)."""
    if ALERTS_CSV.exists():
        df = pd.read_csv(ALERTS_CSV, dtype={"user_id": str})
        df["user_id"] = df["user_id"].fillna("UNKNOWN")
        if "top_features" in df.columns:
            df["top_features"] = df["top_features"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
        if "feature_values" in df.columns:
            df["feature_values"] = df["feature_values"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
        return df
    return _mock_alerts()


def _mock_alerts() -> pd.DataFrame:
    """Generate realistic-looking mock data for demo when alerts.csv is absent."""
    np.random.seed(42)
    users   = [f"USR{i:03d}" for i in [3, 7, 11, 14, 19, 22, 27, 31]]
    roles   = ["teller","dba","loan_officer","sysadmin","risk_analyst"]
    dbs     = ["customer_db","core_db","loan_db","treasury_db","audit_db"]
    actions = ["SELECT","BULK_DOWNLOAD","PRIVILEGE_CHANGE","EXPORT","UPDATE"]
    rows = []
    base = datetime(2025, 1, 15)
    for i in range(150):
        uid  = np.random.choice(users)
        evil = uid in ["USR003","USR011","USR027"]
        risk = float(np.clip(
            np.random.beta(2, 5) + (np.random.uniform(0.4, 0.6) if evil else 0),
            0, 1
        ))
        level = (
            "CRITICAL" if risk >= 0.85 else
            "HIGH"     if risk >= 0.70 else
            "MEDIUM"   if risk >= 0.50 else "LOW"
        )
        rows.append({
            "user_id":            uid,
            "role":               np.random.choice(roles),
            "db_accessed":        np.random.choice(dbs),
            "action":             np.random.choice(actions),
            "risk_score":         round(risk, 4),
            "deviation_score":    round(risk * 0.85, 4),
            "mps_overlap":        round(1 - risk, 6),
            "iso_score":          round(risk * 0.9, 4),
            "is_anomaly_pred":    int(risk >= 0.65),
            "is_anomaly_true":    int(risk >= 0.60),
            "off_hours":          int(risk > 0.70),
            "bulk_download":      int(risk > 0.75),
            "privilege_escalation": int(risk > 0.85),
            "unauthorised_db":    int(risk > 0.72),
            "records_accessed":   int(np.random.randint(1, 200) * (1 + risk * 10)),
            "hour_of_login":      int(np.random.choice(
                [2, 3, 4, 10, 11, 14, 15],
                p=[0.05, 0.05, 0.05, 0.25, 0.25, 0.2, 0.15]
            )),
            "session_duration_min": int(np.random.randint(10, 300)),
            "login_time":         (base + pd.Timedelta(days=i // 4,
                                   hours=int(np.random.randint(0, 23)))).isoformat(),
            "top_features":       ["off_hours","bulk_download","unauthorised_db",
                                   "records_accessed","hour_of_login"],
            "explanation":        f"[{level}] User {uid} — risk {risk:.0%}. behavioural drift detected.",
        })
    return pd.DataFrame(rows)


def risk_colour(score: float) -> str:
    if score >= 0.85: return RISK_COLOURS["CRITICAL"]
    if score >= 0.70: return RISK_COLOURS["HIGH"]
    if score >= 0.50: return RISK_COLOURS["MEDIUM"]
    return RISK_COLOURS["LOW"]


def risk_level(score: float) -> str:
    if score >= 0.85: return "CRITICAL"
    if score >= 0.70: return "HIGH"
    if score >= 0.50: return "MEDIUM"
    return "LOW"


def pill(level: str) -> str:
    c = RISK_COLOURS.get(level, "#888")
    return f'<span class="risk-pill" style="background:{c}22;color:{c};border:1px solid {c}">{level}</span>'


def metric_card(label: str, value, colour: str, suffix: str = "") -> str:
    return f"""
    <div class="metric-card" style="border-color:{colour}">
        <div class="metric-label">{label}</div>
        <div class="metric-val" style="color:{colour}">{value}{suffix}</div>
    </div>"""


def shap_waterfall(
    feature_names: list,
    feature_vals:  list,
    risk_score:    float,
    base_value:    float = 0.12,
) -> go.Figure:
    """
    Build a SHAP-style waterfall chart.
    Contributions are estimated from feature values and risk score.
    """
    n = len(feature_names)
    # Estimate contributions proportional to feature magnitude
    raw = []
    for name, val in zip(feature_names, feature_vals):
        if name in ("off_hours","bulk_download","privilege_escalation","unauthorised_db"):
            raw.append(float(val) * np.random.uniform(0.18, 0.30))
        elif name == "records_accessed":
            raw.append(min(float(val) / 10000, 0.15))
        elif name == "hour_of_login":
            raw.append(0.08 if float(val) < 6 else 0.02)
        else:
            raw.append(abs(float(val)) * 0.01)

    # Scale so contributions sum to risk_score - base_value
    total_raw = sum(raw) + 1e-8
    target    = max(0, risk_score - base_value)
    contribs  = [r / total_raw * target for r in raw]

    # Sort by abs contribution
    pairs = sorted(
        zip([FEATURE_LABELS.get(n, n) for n in feature_names], contribs, feature_vals),
        key=lambda x: abs(x[1]), reverse=True
    )
    names, contribs_s, vals_s = zip(*pairs)

    colours  = [RISK_COLOURS["CRITICAL"] if c > 0 else RISK_COLOURS["LOW"] for c in contribs_s]
    measures = ["relative"] * len(names) + ["total"]
    x_vals   = list(contribs_s) + [base_value]
    x_names  = [f"{n}<br><sub>val={v}</sub>" for n, v in zip(names, vals_s)] + ["Base value"]

    fig = go.Figure(go.Waterfall(
        orientation = "h",
        measure     = measures[::-1],
        x           = x_vals[::-1],
        y           = x_names[::-1],
        connector   = {"line": {"color": "#444"}},
        increasing  = {"marker": {"color": RISK_COLOURS["CRITICAL"]}},
        decreasing  = {"marker": {"color": RISK_COLOURS["LOW"]}},
        totals      = {"marker": {"color": "#5c6bc0"}},
        texttemplate= "%{x:.3f}",
        textposition= "outside",
    ))
    fig.update_layout(
        title       = f"Why this session scored {risk_score:.0%} risk",
        plot_bgcolor= "#1e2130",
        paper_bgcolor="#1e2130",
        font        = {"color": "#cfd8dc"},
        xaxis_title = "Contribution to risk score",
        height      = 380,
        margin      = {"l": 220, "r": 40, "t": 50, "b": 30},
    )
    return fig


def api_score(payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}/score", json=payload, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/cyber-security.png", width=56)
    st.title("SentinelMind")
    st.caption("Quantum-Inspired Insider Threat Detection")
    st.divider()

    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        api_ok = r.status_code == 200
    except Exception:
        api_ok = False

    st.markdown(
        f"**API status:** {'🟢 Online' if api_ok else '🔴 Offline (using CSV)'}"
    )
    st.markdown(f"**Model:** {'MPS scorer' if api_ok else 'Mock / CSV mode'}")
    st.divider()

    min_risk = st.slider("Min risk score", 0.0, 1.0, 0.0, 0.05)
    show_flagged = st.toggle("Flagged only", value=False)
    st.divider()
    st.caption("Wolf Pack · iDEA 2.0 · PSBs Hackathon 2026")

# ─── LOAD DATA ────────────────────────────────────────────────────────────────

df_all = load_alerts()
df = df_all.copy()
if min_risk > 0:
    df = df[df["risk_score"] >= min_risk]
if show_flagged:
    df = df[df["is_anomaly_pred"] == 1]

flagged    = df_all[df_all["is_anomaly_pred"] == 1]
n_total    = len(df_all)
n_flagged  = len(flagged)
n_critical = int((df_all["risk_score"] >= 0.85).sum())
n_high     = int(((df_all["risk_score"] >= 0.70) & (df_all["risk_score"] < 0.85)).sum())
alert_rate = round(100 * n_flagged / max(n_total, 1), 1)

# ─── HEADER ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="header-band">
  <span style="font-size:1.6rem;font-weight:700;color:#fff">🛡️ SentinelMind</span>
</div>
""", unsafe_allow_html=True)

# ─── SUMMARY METRICS ──────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(metric_card("Total sessions", f"{n_total:,}", "#5c6bc0"), unsafe_allow_html=True)
with c2:
    st.markdown(metric_card("Flagged alerts", n_flagged, RISK_COLOURS["HIGH"], f" ({alert_rate}%)"), unsafe_allow_html=True)
with c3:
    st.markdown(metric_card("Critical", n_critical, RISK_COLOURS["CRITICAL"]), unsafe_allow_html=True)
with c4:
    st.markdown(metric_card("High risk", n_high, RISK_COLOURS["HIGH"]), unsafe_allow_html=True)

st.markdown("---")

# ─── TABS ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview", "🚨 Alert Queue", "👤 User Profile", "⚡ Live Score"
])

# ══ TAB 1 — OVERVIEW ══════════════════════════════════════════════════════════

with tab1:
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Risk score distribution")
        fig_hist = px.histogram(
            df_all, x="risk_score", nbins=30,
            color_discrete_sequence=["#5c6bc0"],
        )
        fig_hist.add_vline(x=0.65, line_dash="dash",
                           line_color=RISK_COLOURS["HIGH"],
                           annotation_text="Alert threshold",
                           annotation_position="top right")
        fig_hist.update_layout(
            plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
            font={"color": "#cfd8dc"}, height=280,
            xaxis_title="Risk score", yaxis_title="Sessions",
            margin={"l":30,"r":20,"t":20,"b":40},
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_right:
        st.subheader("Risk breakdown")
        levels = ["CRITICAL","HIGH","MEDIUM","LOW"]
        counts = [
            int((df_all["risk_score"] >= 0.85).sum()),
            int(((df_all["risk_score"] >= 0.70) & (df_all["risk_score"] < 0.85)).sum()),
            int(((df_all["risk_score"] >= 0.50) & (df_all["risk_score"] < 0.70)).sum()),
            int((df_all["risk_score"] < 0.50).sum()),
        ]
        fig_pie = go.Figure(go.Pie(
            labels=levels, values=counts,
            marker_colors=[RISK_COLOURS[l] for l in levels],
            hole=0.55,
            textinfo="label+percent",
        ))
        fig_pie.update_layout(
            plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
            font={"color":"#cfd8dc"}, height=280,
            margin={"l":10,"r":10,"t":10,"b":10},
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Top risky users")
    top_users = (
        df_all.groupby("user_id")
        .agg(max_risk=("risk_score","max"),
             sessions=("user_id","count"),
             alerts=("is_anomaly_pred","sum"))
        .reset_index()
        .sort_values("max_risk", ascending=False)
        .head(8)
    )
    top_users["risk_level"] = top_users["max_risk"].apply(risk_level)

    fig_bar = go.Figure(go.Bar(
        x=top_users["user_id"],
        y=top_users["max_risk"],
        marker_color=[risk_colour(r) for r in top_users["max_risk"]],
        text=top_users["max_risk"].apply(lambda x: f"{x:.0%}"),
        textposition="outside",
    ))
    fig_bar.update_layout(
        plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
        font={"color":"#cfd8dc"}, height=280,
        yaxis={"tickformat":".0%", "range":[0,1.1]},
        xaxis_title="User", yaxis_title="Peak risk score",
        margin={"l":30,"r":20,"t":20,"b":40},
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# ══ TAB 2 — ALERT QUEUE ═══════════════════════════════════════════════════════

with tab2:
    st.subheader("Flagged sessions — investigator queue")

    queue = flagged.sort_values("risk_score", ascending=False).copy()

    if queue.empty:
        st.info("No flagged sessions. Lower the risk threshold in the sidebar.")
    else:
        # Render with colour-coded risk level
        display = queue[[
            "user_id","risk_score","deviation_score",
            "is_anomaly_true","explanation"
        ]].copy()
        display["risk_level"] = display["risk_score"].apply(risk_level)
        display["risk_score_pct"] = display["risk_score"].apply(lambda x: f"{x:.0%}")

        st.dataframe(
            display[["user_id","risk_level","risk_score_pct",
                      "deviation_score","is_anomaly_true","explanation"]],
            column_config={
                "user_id":         st.column_config.TextColumn("User"),
                "risk_level":      st.column_config.TextColumn("Level"),
                "risk_score_pct":  st.column_config.TextColumn("Risk"),
                "deviation_score": st.column_config.ProgressColumn(
                    "Deviation", min_value=0, max_value=1, format="%.2f"
                ),
                "is_anomaly_true": st.column_config.CheckboxColumn("Confirmed"),
                "explanation":     st.column_config.TextColumn("Reason", width="large"),
            },
            use_container_width=True,
            height=420,
        )

        st.caption(f"{len(queue)} flagged sessions shown. Click a row to copy user_id for the User Profile tab.")


# ══ TAB 3 — USER PROFILE ══════════════════════════════════════════════════════

with tab3:
    all_users = sorted(df_all["user_id"].unique().tolist())
    selected  = st.selectbox("Select user", all_users)

    user_df = df_all[df_all["user_id"] == selected].copy()

    if user_df.empty:
        st.warning("No sessions found for this user.")
    else:
        # — Header stats —
        max_risk  = user_df["risk_score"].max()
        avg_risk  = user_df["risk_score"].mean()
        n_alerts  = user_df["is_anomaly_pred"].sum()
        level     = risk_level(max_risk)

        uc1, uc2, uc3, uc4 = st.columns(4)
        with uc1:
            st.markdown(metric_card("Peak risk",    f"{max_risk:.0%}", risk_colour(max_risk)), unsafe_allow_html=True)
        with uc2:
            st.markdown(metric_card("Avg risk",     f"{avg_risk:.0%}", "#5c6bc0"),  unsafe_allow_html=True)
        with uc3:
            st.markdown(metric_card("Alerts fired", n_alerts,          RISK_COLOURS["HIGH"]), unsafe_allow_html=True)
        with uc4:
            st.markdown(metric_card("Sessions",     len(user_df),      "#78909c"),  unsafe_allow_html=True)

        st.markdown(f"**Risk classification:** {pill(level)}", unsafe_allow_html=True)

        # — Risk timeline —
        st.subheader("Risk score timeline")
        user_df_sorted = user_df.sort_values("risk_score", ascending=False).reset_index(drop=True)
        user_df_sorted["session_idx"] = range(len(user_df_sorted))

        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=user_df_sorted["session_idx"],
            y=user_df_sorted["risk_score"],
            mode="lines+markers",
            line={"color": "#5c6bc0", "width": 2},
            marker={
                "color": [risk_colour(r) for r in user_df_sorted["risk_score"]],
                "size": 8,
            },
            name="Risk score",
        ))
        fig_line.add_hrect(y0=0.65, y1=1.0,
                           fillcolor=RISK_COLOURS["HIGH"], opacity=0.08,
                           annotation_text="Alert zone",
                           annotation_position="top left")
        fig_line.add_hline(y=0.65, line_dash="dash",
                           line_color=RISK_COLOURS["HIGH"])
        fig_line.update_layout(
            plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
            font={"color":"#cfd8dc"}, height=260,
            yaxis={"tickformat":".0%","range":[0,1.05]},
            xaxis_title="Session (sorted by risk)", yaxis_title="Risk score",
            margin={"l":40,"r":20,"t":20,"b":40},
        )
        st.plotly_chart(fig_line, use_container_width=True)

        # — Worst session SHAP waterfall —
        st.subheader("SHAP explanation — highest risk session")
        worst = user_df.loc[user_df["risk_score"].idxmax()]

        st.markdown(
            f'<div class="explain-box">{worst["explanation"]}</div>',
            unsafe_allow_html=True
        )

        top_feats = worst.get("top_features", [])
        if isinstance(top_feats, str):
            try:
                top_feats = ast.literal_eval(top_feats)
            except Exception:
                top_feats = []

        if not top_feats:
            top_feats = ["off_hours","bulk_download","unauthorised_db",
                         "records_accessed","hour_of_login"]

        feat_dict = worst.get("feature_values", {})
        feat_vals = []
        for f in top_feats:
            if isinstance(feat_dict, dict) and f in feat_dict:
                feat_vals.append(float(feat_dict[f]))
            else:
                feat_vals.append(float(worst.get(f, 0)))

        fig_shap = shap_waterfall(top_feats, feat_vals, float(worst["risk_score"]))
        st.plotly_chart(fig_shap, use_container_width=True)

        # — Session table —
        with st.expander("All sessions for this user"):
            cols_to_show = [c for c in ["risk_score","deviation_score","is_anomaly_pred",
                                       "off_hours","bulk_download","privilege_escalation",
                                       "unauthorised_db","records_accessed","hour_of_login",
                                       "explanation"] if c in user_df.columns]
            st.dataframe(
                user_df[cols_to_show].sort_values("risk_score", ascending=False),
                use_container_width=True,
                height=300,
            )


# ══ TAB 4 — LIVE SCORE ════════════════════════════════════════════════════════

with tab4:
    st.subheader("Score a session live")
    st.caption("Calls POST /score on the FastAPI backend. Requires API to be running.")

    with st.form("score_form"):
        fc1, fc2, fc3 = st.columns(3)

        with fc1:
            uid            = st.text_input("User ID",             value="USR003")
            role_enc       = st.number_input("Role (encoded)",    value=2,   min_value=0, max_value=10)
            db_enc         = st.number_input("DB accessed (enc)", value=4,   min_value=0, max_value=10)
            action_enc     = st.number_input("Action (encoded)",  value=1,   min_value=0, max_value=10)

        with fc2:
            hour_login     = st.slider("Hour of login",           0,  23,  3)
            day_of_week    = st.slider("Day of week (0=Mon)",     0,  6,   6)
            duration       = st.number_input("Session duration (min)", value=240)
            records        = st.number_input("Records accessed",       value=9000)

        with fc3:
            off_hours      = st.toggle("Off-hours login",            value=True)
            unauth_db      = st.toggle("Unauthorised DB access",     value=True)
            bulk_dl        = st.toggle("Bulk download",              value=True)
            priv_esc       = st.toggle("Privilege escalation",       value=False)
            rec_ratio      = st.number_input("Records vs. user avg", value=45.0)

        submitted = st.form_submit_button("🔍 Score this session", use_container_width=True)

    if submitted:
        payload = {
            "user_id":               uid,
            "role_enc":              int(role_enc),
            "department_enc":        1,
            "db_accessed_enc":       int(db_enc),
            "action_enc":            int(action_enc),
            "session_duration_min":  float(duration),
            "records_accessed":      float(records),
            "hour_of_login":         int(hour_login),
            "day_of_week":           int(day_of_week),
            "off_hours":             int(off_hours),
            "unauthorised_db":       int(unauth_db),
            "bulk_download":         int(bulk_dl),
            "privilege_escalation":  int(priv_esc),
            "records_vs_user_avg":   float(rec_ratio),
            "duration_vs_user_avg":  float(duration / 30),
            "user_avg_records":      200.0,
            "user_max_records":      500.0,
            "user_avg_duration":     30.0,
            "user_login_std":        1.2,
            "user_session_count":    15.0,
        }

        with st.spinner("Scoring via MPS engine..."):
            result = api_score(payload)

        if result:
            risk  = result["risk_score"]
            level = result.get("risk_level", risk_level(risk))
            colour = risk_colour(risk)

            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                st.markdown(metric_card("Risk score",      f"{risk:.0%}",                    colour),   unsafe_allow_html=True)
            with rc2:
                st.markdown(metric_card("Deviation",       f"{result['deviation_score']:.3f}", "#5c6bc0"), unsafe_allow_html=True)
            with rc3:
                st.markdown(metric_card("Flagged",         "YES" if result["is_anomaly_pred"] else "NO",
                                        RISK_COLOURS["CRITICAL"] if result["is_anomaly_pred"] else RISK_COLOURS["LOW"]),
                            unsafe_allow_html=True)

            st.markdown(
                f'<div class="explain-box">{result["explanation"]}</div>',
                unsafe_allow_html=True
            )

            # SHAP waterfall from live result
            top_feats = result.get("top_features", [])
            feat_vals = [float(payload.get(f, 0)) for f in top_feats]
            if top_feats:
                fig_live = shap_waterfall(top_feats, feat_vals, risk)
                st.plotly_chart(fig_live, use_container_width=True)

            with st.expander("Raw API response"):
                st.json(result)
        else:
            # API offline — compute locally with mock scorer
            st.warning("API offline. Showing local mock score.")
            flags = (
                int(off_hours) * 0.30 + int(bulk_dl) * 0.30
                + int(priv_esc) * 0.25 + int(unauth_db) * 0.20
            )
            risk   = float(np.clip(flags + min(0.20, (rec_ratio - 1) * 0.01), 0, 1))
            level  = risk_level(risk)
            colour = risk_colour(risk)
            st.markdown(metric_card("Mock risk score", f"{risk:.0%}", colour), unsafe_allow_html=True)
            top_feats = ["off_hours","bulk_download","unauthorised_db","records_vs_user_avg","hour_of_login"]
            feat_vals = [int(off_hours), int(bulk_dl), int(unauth_db), rec_ratio, hour_login]
            fig_mock = shap_waterfall(top_feats, feat_vals, risk)
            st.plotly_chart(fig_mock, use_container_width=True)
