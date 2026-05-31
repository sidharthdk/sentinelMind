"""
SentinelMind — FastAPI Backend
Exposes the MPS scorer as a REST API.

Endpoints:
    POST /score          — score a single privileged-user session
    GET  /alerts         — return all flagged sessions from alerts.csv
    GET  /alerts/{user}  — return alerts for a specific user
    GET  /stats          — dashboard summary stats
    GET  /health         — liveness check

Run locally:
    uvicorn main:app --reload --port 8000

Then open:
    http://localhost:8000/docs   (auto-generated Swagger UI)
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─── LOGGING ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger("sentinelmind")

# ─── PATHS ────────────────────────────────────────────────────────────────────

MODEL_PATH  = Path("models/mps_scorer.pkl")
ALERTS_PATH = Path("data/alerts.csv")

# ─── APP ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SentinelMind API",
    description=(
        "Quantum-inspired insider threat detection for privileged banking users. "
        "AI-CSPARC · PSBs Hackathon 2026 · Team Wolf Pack"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── SCORER SINGLETON ─────────────────────────────────────────────────────────

_scorer = None

def get_scorer():
    """
    Load scorer once on first call, reuse thereafter.
    Falls back to a mock scorer if model file is missing
    so the API stays alive during development.
    """
    global _scorer
    if _scorer is not None:
        return _scorer

    if MODEL_PATH.exists():
        log.info(f"Loading MPS scorer from {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            _scorer = pickle.load(f)
        log.info("Scorer loaded.")
    else:
        log.warning(
            f"Model file not found at {MODEL_PATH}. "
            "Using mock scorer — run mps_scorer.py to train a real model."
        )
        _scorer = _MockScorer()

    return _scorer


class _MockScorer:
    """
    Stub scorer used when models/mps_scorer.pkl doesn't exist yet.
    Returns plausible-looking scores so the API and dashboard
    can be developed and demoed before the model is trained.
    """
    is_fitted = True
    anomaly_threshold = 0.65

    def score_event(self, event: dict) -> dict:
        flags = (
            event.get("off_hours", 0) * 0.30
            + event.get("bulk_download", 0) * 0.30
            + event.get("privilege_escalation", 0) * 0.25
            + event.get("unauthorised_db", 0) * 0.20
        )
        ratio_bump = min(0.20, (event.get("records_vs_user_avg", 1) - 1) * 0.01)
        risk = float(np.clip(flags + ratio_bump, 0.0, 1.0))

        level = (
            "CRITICAL" if risk >= 0.85 else
            "HIGH"     if risk >= 0.70 else
            "MEDIUM"   if risk >= 0.50 else
            "LOW"
        )

        reasons = []
        if event.get("off_hours"):            reasons.append("off-hours login")
        if event.get("bulk_download"):        reasons.append("bulk download")
        if event.get("privilege_escalation"): reasons.append("privilege escalation")
        if event.get("unauthorised_db"):      reasons.append("unauthorised DB access")
        if not reasons:                       reasons.append("minor behavioural drift")

        uid = event.get("user_id", "UNKNOWN")
        top = ["off_hours","bulk_download","privilege_escalation",
               "unauthorised_db","records_vs_user_avg"]

        return {
            "user_id":         uid,
            "risk_score":      round(risk, 4),
            "deviation_score": round(risk * 0.85, 4),
            "mps_overlap":     round(1.0 - risk, 6),
            "iso_score":       round(risk * 0.9, 4),
            "is_anomaly_pred": int(risk >= self.anomaly_threshold),
            "top_features":    top,
            "feature_values":  {k: event.get(k, 0) for k in top},
            "explanation": (
                f"[{level}] User {uid} — risk {risk:.0%}. "
                + "; ".join(reasons) + ". (mock scorer)"
            ),
        }


# ─── SCHEMAS ──────────────────────────────────────────────────────────────────

class SessionEvent(BaseModel):
    """
    A single privileged-user session to score.
    All fields map directly to feature_extractor.py output columns.
    """
    user_id:               str   = Field(..., example="USR003")
    role_enc:              int   = Field(0,   example=2,    description="LabelEncoded role")
    department_enc:        int   = Field(0,   example=1)
    db_accessed_enc:       int   = Field(0,   example=4,    description="LabelEncoded DB name")
    action_enc:            int   = Field(0,   example=1,    description="LabelEncoded action type")
    session_duration_min:  float = Field(30,  example=240,  description="Session length in minutes")
    records_accessed:      float = Field(10,  example=9000, description="Number of records queried")
    hour_of_login:         int   = Field(10,  example=3,    description="Hour of login (0–23)")
    day_of_week:           int   = Field(1,   example=6,    description="0=Monday … 6=Sunday")
    off_hours:             int   = Field(0,   example=1,    description="1 if login outside work hours")
    unauthorised_db:       int   = Field(0,   example=1,    description="1 if DB outside role access")
    bulk_download:         int   = Field(0,   example=1,    description="1 if bulk volume detected")
    privilege_escalation:  int   = Field(0,   example=0,    description="1 if privilege change recorded")
    records_vs_user_avg:   float = Field(1.0, example=45.0, description="Ratio to user's own baseline")
    duration_vs_user_avg:  float = Field(1.0, example=8.2)
    user_avg_records:      float = Field(50,  example=200)
    user_max_records:      float = Field(100, example=500)
    user_avg_duration:     float = Field(30,  example=30)
    user_login_std:        float = Field(1.0, example=1.2)
    user_session_count:    float = Field(10,  example=15)


class ScoreResponse(BaseModel):
    user_id:         str
    risk_score:      float         = Field(..., description="0.0 (safe) → 1.0 (critical)")
    risk_level:      str           = Field(..., description="LOW / MEDIUM / HIGH / CRITICAL")
    deviation_score: float
    mps_overlap:     float
    iso_score:       float
    is_anomaly_pred: int           = Field(..., description="1 = flagged as threat")
    top_features:    list[str]
    feature_values:  dict
    explanation:     str
    scored_at:       str


class AlertRecord(BaseModel):
    user_id:         str
    risk_score:      float
    risk_level:      str
    deviation_score: float
    is_anomaly_pred: int
    is_anomaly_true: int
    explanation:     str
    top_features:    list[str]


class StatsResponse(BaseModel):
    total_sessions:    int
    total_alerts:      int
    alert_rate_pct:    float
    critical_count:    int
    high_count:        int
    medium_count:      int
    top_risky_users:   list[dict]
    model_loaded:      bool
    scorer_type:       str


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def risk_level(score: float) -> str:
    if score >= 0.85: return "CRITICAL"
    if score >= 0.70: return "HIGH"
    if score >= 0.50: return "MEDIUM"
    return "LOW"


def load_alerts() -> pd.DataFrame:
    if not ALERTS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(ALERTS_PATH, dtype={"user_id": str})
    df["user_id"] = df["user_id"].fillna("UNKNOWN")
    # Normalise top_features column (stored as string repr of list)
    if "top_features" in df.columns:
        df["top_features"] = df["top_features"].apply(
            lambda x: json.loads(x.replace("'", '"')) if isinstance(x, str) else x
        )
    return df


# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["system"])
def health():
    """Liveness probe — returns 200 if the API is running."""
    return {
        "status":    "ok",
        "service":   "SentinelMind API",
        "timestamp": datetime.utcnow().isoformat(),
        "model_ready": MODEL_PATH.exists(),
    }


@app.post("/score", response_model=ScoreResponse, tags=["inference"])
def score_session(event: SessionEvent):
    """
    Score a single privileged-user session.

    Pass a session event and receive:
    - `risk_score`      — 0.0 to 1.0
    - `risk_level`      — LOW / MEDIUM / HIGH / CRITICAL
    - `explanation`     — human-readable reason the session was flagged
    - `top_features`    — the 5 behavioural features that drove the score
    - `is_anomaly_pred` — 1 if the session should be investigated

    Powered by a quantum-inspired Matrix Product State (MPS) scorer
    fused with an IsolationForest cross-validator.
    """
    scorer = get_scorer()

    try:
        result = scorer.score_event(event.model_dump())
    except Exception as e:
        log.error(f"Scoring error: {e}")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")

    log.info(
        f"Scored {event.user_id} → risk={result['risk_score']:.3f} "
        f"[{risk_level(result['risk_score'])}]"
    )

    return ScoreResponse(
        **result,
        risk_level=risk_level(result["risk_score"]),
        scored_at=datetime.utcnow().isoformat(),
    )


@app.get("/alerts", response_model=list[AlertRecord], tags=["alerts"])
def get_alerts(
    flagged_only: bool  = Query(True,  description="Return only flagged sessions"),
    min_risk:     float = Query(0.0,   description="Minimum risk score filter (0.0–1.0)"),
    limit:        int   = Query(100,   description="Max number of records to return"),
    user_id:      Optional[str] = Query(None, description="Filter by user_id"),
):
    """
    Return scored sessions from the last scoring run.

    Use `flagged_only=true` for the investigator queue.
    Use `min_risk=0.7` to see only HIGH and CRITICAL alerts.
    """
    df = load_alerts()
    if df.empty:
        return []

    if flagged_only:
        df = df[df["is_anomaly_pred"] == 1]
    if user_id:
        df = df[df["user_id"] == user_id]
    if min_risk > 0:
        df = df[df["risk_score"] >= min_risk]

    df = df.sort_values("risk_score", ascending=False).head(limit)

    records = []
    for _, row in df.iterrows():
        feats = row.get("top_features", [])
        if isinstance(feats, str):
            try:
                feats = json.loads(feats.replace("'", '"'))
            except Exception:
                feats = []

        records.append(AlertRecord(
            user_id         = str(row.get("user_id", "")),
            risk_score      = round(float(row.get("risk_score", 0)), 4),
            risk_level      = risk_level(float(row.get("risk_score", 0))),
            deviation_score = round(float(row.get("deviation_score", 0)), 4),
            is_anomaly_pred = int(row.get("is_anomaly_pred", 0)),
            is_anomaly_true = int(row.get("is_anomaly_true", -1)),
            explanation     = str(row.get("explanation", "")),
            top_features    = feats,
        ))

    return records


@app.get("/alerts/{user_id}", response_model=list[AlertRecord], tags=["alerts"])
def get_user_alerts(user_id: str):
    """
    Return all scored sessions for a specific user.
    Useful for the per-user timeline in the investigator dashboard.
    """
    df = load_alerts()
    if df.empty:
        raise HTTPException(status_code=404, detail="No alert data found.")

    user_df = df[df["user_id"] == user_id]
    if user_df.empty:
        raise HTTPException(status_code=404, detail=f"No sessions found for user {user_id}.")

    user_df = user_df.sort_values("risk_score", ascending=False)

    records = []
    for _, row in user_df.iterrows():
        feats = row.get("top_features", [])
        if isinstance(feats, str):
            try:
                feats = json.loads(feats.replace("'", '"'))
            except Exception:
                feats = []

        records.append(AlertRecord(
            user_id         = str(row.get("user_id", "")),
            risk_score      = round(float(row.get("risk_score", 0)), 4),
            risk_level      = risk_level(float(row.get("risk_score", 0))),
            deviation_score = round(float(row.get("deviation_score", 0)), 4),
            is_anomaly_pred = int(row.get("is_anomaly_pred", 0)),
            is_anomaly_true = int(row.get("is_anomaly_true", -1)),
            explanation     = str(row.get("explanation", "")),
            top_features    = feats,
        ))

    return records


@app.get("/stats", response_model=StatsResponse, tags=["dashboard"])
def get_stats():
    """
    Aggregated statistics for the investigator dashboard header.

    Returns total session count, alert counts by risk level,
    and the top 5 highest-risk users.
    """
    df = load_alerts()

    if df.empty:
        return StatsResponse(
            total_sessions=0, total_alerts=0, alert_rate_pct=0.0,
            critical_count=0, high_count=0, medium_count=0,
            top_risky_users=[], model_loaded=MODEL_PATH.exists(),
            scorer_type="mock" if not MODEL_PATH.exists() else "mps",
        )

    flagged  = df[df["is_anomaly_pred"] == 1]
    total    = len(df)
    n_alerts = len(flagged)

    df["risk_level_col"] = df["risk_score"].apply(risk_level)

    top_users = (
        flagged.groupby("user_id")["risk_score"]
        .max()
        .reset_index()
        .sort_values("risk_score", ascending=False)
        .head(5)
        .rename(columns={"risk_score": "max_risk"})
        .assign(max_risk=lambda x: x["max_risk"].round(4))
        .assign(risk_level=lambda x: x["max_risk"].apply(risk_level))
        .to_dict("records")
    )

    return StatsResponse(
        total_sessions  = total,
        total_alerts    = n_alerts,
        alert_rate_pct  = round(100 * n_alerts / max(total, 1), 2),
        critical_count  = int((df["risk_level_col"] == "CRITICAL").sum()),
        high_count      = int((df["risk_level_col"] == "HIGH").sum()),
        medium_count    = int((df["risk_level_col"] == "MEDIUM").sum()),
        top_risky_users = top_users,
        model_loaded    = MODEL_PATH.exists(),
        scorer_type     = "mock" if not MODEL_PATH.exists() else "mps",
    )
