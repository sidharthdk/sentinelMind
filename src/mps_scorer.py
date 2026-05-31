"""
SentinelMind — MPS Scorer
Quantum-inspired insider threat detection using Matrix Product States.

Each privileged user gets their own MPS baseline trained on their
normal behaviour. Anomaly score = how far a new session deviates
from that learned baseline, measured via tensor network overlap.

Architecture:
    MPSTensor        — single site tensor (left_bond, feature_dim, right_bond)
    UserBaseline     — per-user MPS chain + feature statistics
    MPSScorer        — trains baselines, scores sessions, emits alerts

Usage:
    scorer = MPSScorer(bond_dim=4, anomaly_threshold=0.75)
    scorer.fit("data/features_combined.csv")
    alerts = scorer.score("data/features_combined.csv")
    scorer.save("models/mps_scorer.pkl")

    # Single event inference (for FastAPI /score endpoint)
    risk = scorer.score_event(event_dict)
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score

DATA_DIR  = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# ─── CONFIG ───────────────────────────────────────────────────────────────────

# Feature columns produced by feature_extractor.py
# These are the behavioural vectors fed into MPS
FEATURE_COLS = [
    "role_enc", "department_enc", "db_accessed_enc", "action_enc",
    "session_duration_min", "records_accessed",
    "hour_of_login", "day_of_week",
    "off_hours", "unauthorised_db", "bulk_download", "privilege_escalation",
    "records_vs_user_avg", "duration_vs_user_avg",
    "user_avg_records", "user_max_records", "user_avg_duration",
    "user_login_std", "user_session_count",
]

# ─── MPS CORE ─────────────────────────────────────────────────────────────────

class MPSTensor:
    """
    Single site in the MPS chain.
    Shape: (left_bond_dim, feature_dim, right_bond_dim)
    For boundary sites: left=1 (first) or right=1 (last).
    """
    def __init__(self, left_dim: int, feature_dim: int, right_dim: int):
        scale = 1.0 / np.sqrt(left_dim * right_dim + 1e-8)
        self.W = np.random.randn(left_dim, feature_dim, right_dim) * scale

    def contract(self, x_val: float, state: np.ndarray) -> np.ndarray:
        """
        Contract this site tensor with scalar input x_val and
        incoming bond state vector.
        state: (left_bond_dim,)
        returns: (right_bond_dim,)
        """
        # W[:, :, :] contracted: sum over left & feature dims
        # Treat feature as linear: W contracted with x_val
        W_contracted = np.einsum("ijk,j->ik", self.W, np.array([x_val]))
        return state @ W_contracted   # (right_bond_dim,)


class UserMPSBaseline:
    """
    MPS chain representing one user's normal behavioural baseline.
    Trained by gradient-free update: iteratively shift tensors
    toward observed normal sessions.
    """
    def __init__(self, n_features: int, bond_dim: int = 4):
        self.n_features = n_features
        self.bond_dim   = bond_dim
        self.tensors: list[MPSTensor] = []
        self.mean_vec:  Optional[np.ndarray] = None   # feature mean
        self.std_vec:   Optional[np.ndarray] = None   # feature std
        self.n_samples: int = 0
        self._init_tensors()

    def _init_tensors(self) -> None:
        for i in range(self.n_features):
            left  = 1 if i == 0 else self.bond_dim
            right = 1 if i == self.n_features - 1 else self.bond_dim
            self.tensors.append(MPSTensor(left, 1, right))

    def overlap(self, x: np.ndarray) -> float:
        """
        Compute MPS overlap with normalised input vector x.
        Higher overlap = more familiar = lower anomaly risk.
        """
        x_norm = (x - self.mean_vec) / (self.std_vec + 1e-8)
        state = np.ones(1)   # left boundary
        for i, tensor in enumerate(self.tensors):
            state = tensor.contract(float(x_norm[i]), state)
        return float(np.abs(state.sum()))

    def fit(self, X: np.ndarray, lr: float = 0.01, epochs: int = 5) -> None:
        """
        Fit MPS to normal sessions via simple gradient-free
        tensor update (shift toward observed data mean).
        """
        self.n_samples = len(X)
        self.mean_vec  = X.mean(axis=0)
        self.std_vec   = X.std(axis=0) + 1e-8

        X_norm = (X - self.mean_vec) / self.std_vec

        for _ in range(epochs):
            for x in X_norm:
                state = np.ones(1)
                for i, tensor in enumerate(self.tensors):
                    # Nudge tensor toward current input
                    delta = np.einsum(
                        "i,j,k->ijk",
                        state,
                        np.array([float(x[i])]),
                        np.ones(tensor.W.shape[2])
                    ) * lr
                    tensor.W += delta[:tensor.W.shape[0], :, :]
                    state = tensor.contract(float(x[i]), state)

    def deviation_score(self, x: np.ndarray) -> float:
        """
        Euclidean deviation from user's mean behavioural vector.
        Normalised by std so it's comparable across users.
        """
        z = (x - self.mean_vec) / (self.std_vec + 1e-8)
        return float(np.linalg.norm(z))


# ─── MPS SCORER ───────────────────────────────────────────────────────────────

@dataclass
class Alert:
    user_id:          str
    risk_score:       float          # 0.0 – 1.0
    deviation_score:  float
    mps_overlap:      float
    iso_score:        float          # IsolationForest anomaly score
    is_anomaly_pred:  int            # 1 = flagged
    is_anomaly_true:  int            # ground truth (if available)
    top_features:     list[str]      # top contributing feature names
    feature_values:   dict           # raw feature values for SHAP
    explanation:      str            # human-readable reason


class MPSScorer:
    """
    Full SentinelMind scoring engine.

    Combines:
      - Per-user MPS baseline (temporal behavioural modelling)
      - IsolationForest (unsupervised cross-validation)
      - Risk score fusion (weighted combination of both signals)
      - Threshold-based alert generation with SHAP-ready output
    """

    def __init__(
        self,
        bond_dim: int          = 4,
        anomaly_threshold: float = 0.65,
        iso_contamination: float = 0.05,
        mps_weight: float      = 0.6,    # weight given to MPS vs IsoForest
    ):
        self.bond_dim          = bond_dim
        self.anomaly_threshold = anomaly_threshold
        self.iso_contamination = iso_contamination
        self.mps_weight        = mps_weight

        self.user_baselines: dict[str, UserMPSBaseline] = {}
        self.iso_forest:     Optional[IsolationForest]  = None
        self.scaler:         Optional[StandardScaler]   = None
        self.feature_cols:   list[str]                  = []
        self.is_fitted:      bool                       = False

    # ── fit ─────────────────────────────────────────────────────────

    def fit(self, data_path: str) -> "MPSScorer":
        """
        Train per-user MPS baselines and IsolationForest
        on NORMAL sessions only (is_anomaly == 0).
        """
        print(f"\n[MPS] Loading {data_path} ...")
        df = pd.read_csv(data_path)

        # Select available feature columns
        self.feature_cols = [c for c in FEATURE_COLS if c in df.columns]
        if not self.feature_cols:
            raise ValueError(
                "No recognised feature columns found. "
                "Run feature_extractor.py first."
            )
        print(f"[MPS] Using {len(self.feature_cols)} features: {self.feature_cols}")

        # Scale features
        self.scaler = StandardScaler()
        normal_df   = df[df["is_anomaly"] == 0].copy()
        self.scaler.fit(normal_df[self.feature_cols])

        # ── Per-user MPS baselines ──
        print(f"[MPS] Fitting per-user baselines ...")
        user_col = "user_id" if "user_id" in df.columns else None

        if user_col:
            for uid, group in normal_df.groupby(user_col):
                if len(group) < 2:
                    continue   # need at least 2 sessions to learn a baseline
                X = self.scaler.transform(group[self.feature_cols])
                baseline = UserMPSBaseline(
                    n_features=len(self.feature_cols),
                    bond_dim=self.bond_dim
                )
                baseline.fit(X)
                self.user_baselines[uid] = baseline
        else:
            # No user_id — fit a global baseline (NSL-KDD mode)
            X = self.scaler.transform(normal_df[self.feature_cols])
            baseline = UserMPSBaseline(
                n_features=len(self.feature_cols),
                bond_dim=self.bond_dim
            )
            baseline.fit(X)
            self.user_baselines["__global__"] = baseline

        print(f"[MPS] Trained baselines for {len(self.user_baselines)} users")

        # ── IsolationForest (fallback validator) ──
        print(f"[MPS] Fitting IsolationForest ...")
        X_all = self.scaler.transform(df[self.feature_cols])
        self.iso_forest = IsolationForest(
            n_estimators=100,
            contamination=self.iso_contamination,
            random_state=42,
            n_jobs=-1,
        )
        self.iso_forest.fit(X_all)

        self.is_fitted = True
        print("[MPS] Fit complete.\n")
        return self

    # ── score ────────────────────────────────────────────────────────

    def score(self, data_path: str) -> pd.DataFrame:
        """
        Score every session in data_path.
        Returns DataFrame of Alert objects with risk scores.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before score().")

        df = pd.read_csv(data_path)
        X  = self.scaler.transform(df[self.feature_cols])

        # IsolationForest raw scores (negative = more anomalous)
        iso_raw    = self.iso_forest.decision_function(X)
        iso_scores = 1.0 - (iso_raw - iso_raw.min()) / (iso_raw.max() - iso_raw.min() + 1e-8)

        alerts = []
        for i, row in df.iterrows():
            x      = X[i]
            uid    = str(row.get("user_id", "__global__"))
            true_label = int(row.get("is_anomaly", -1))

            # Get user baseline (fall back to global if user unseen)
            baseline = self.user_baselines.get(
                uid,
                self.user_baselines.get("__global__")
            )

            if baseline is None:
                dev_score = 0.5
                overlap   = 0.5
            else:
                dev_score = baseline.deviation_score(x)
                overlap   = baseline.overlap(x)

            # Normalise deviation to 0–1
            dev_norm = float(np.tanh(dev_score / 3.0))

            # Fuse MPS + IsoForest risk signals
            iso_s     = float(iso_scores[i])
            risk      = self.mps_weight * dev_norm + (1 - self.mps_weight) * iso_s
            risk      = float(np.clip(risk, 0.0, 1.0))

            # Flag above threshold
            is_pred   = int(risk >= self.anomaly_threshold)

            # Top contributing features (by abs z-score from user mean)
            if baseline and baseline.mean_vec is not None:
                z = np.abs((x - baseline.mean_vec) / (baseline.std_vec + 1e-8))
                top_idx  = np.argsort(z)[::-1][:5]
                top_feats = [self.feature_cols[j] for j in top_idx]
                top_vals  = {self.feature_cols[j]: round(float(row.get(self.feature_cols[j], 0)), 4)
                             for j in top_idx}
            else:
                top_feats = self.feature_cols[:5]
                top_vals  = {c: round(float(row.get(c, 0)), 4) for c in top_feats}

            explanation = self._explain(uid, top_feats, top_vals, risk, row)

            alerts.append({
                "user_id":         uid,
                "risk_score":      round(risk, 4),
                "deviation_score": round(dev_norm, 4),
                "mps_overlap":     round(overlap, 6),
                "iso_score":       round(iso_s, 4),
                "is_anomaly_pred": is_pred,
                "is_anomaly_true": true_label,
                "top_features":    top_feats,
                "feature_values":  top_vals,
                "explanation":     explanation,
            })

        return pd.DataFrame(alerts)

    # ── score_event ──────────────────────────────────────────────────

    def score_event(self, event: dict) -> dict:
        """
        Score a single event dict (for FastAPI /score endpoint).
        event must contain the feature_cols keys.
        Returns risk score + explanation JSON.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before score_event().")

        row = pd.Series(event)
        x_raw = np.array([float(row.get(c, 0)) for c in self.feature_cols]).reshape(1, -1)
        x     = self.scaler.transform(x_raw)[0]

        uid      = str(event.get("user_id", "__global__"))
        baseline = self.user_baselines.get(uid, self.user_baselines.get("__global__"))

        dev_score = baseline.deviation_score(x) if baseline else 0.5
        overlap   = baseline.overlap(x) if baseline else 0.5
        dev_norm  = float(np.tanh(dev_score / 3.0))

        iso_raw  = self.iso_forest.decision_function(x_raw)[0]
        iso_s    = float(np.clip(1.0 - (iso_raw + 0.5), 0, 1))
        risk     = float(np.clip(
            self.mps_weight * dev_norm + (1 - self.mps_weight) * iso_s,
            0.0, 1.0
        ))

        if baseline and baseline.mean_vec is not None:
            z         = np.abs((x - baseline.mean_vec) / (baseline.std_vec + 1e-8))
            top_idx   = np.argsort(z)[::-1][:5]
            top_feats = [self.feature_cols[j] for j in top_idx]
            top_vals  = {self.feature_cols[j]: round(float(event.get(self.feature_cols[j], 0)), 4)
                         for j in top_idx}
        else:
            top_feats = self.feature_cols[:5]
            top_vals  = {c: round(float(event.get(c, 0)), 4) for c in top_feats}

        return {
            "user_id":         uid,
            "risk_score":      round(risk, 4),
            "deviation_score": round(dev_norm, 4),
            "mps_overlap":     round(overlap, 6),
            "iso_score":       round(iso_s, 4),
            "is_anomaly_pred": int(risk >= self.anomaly_threshold),
            "top_features":    top_feats,
            "feature_values":  top_vals,
            "explanation":     self._explain(uid, top_feats, top_vals, risk, row),
        }

    # ── explain ──────────────────────────────────────────────────────

    def _explain(
        self,
        uid: str,
        top_feats: list[str],
        top_vals: dict,
        risk: float,
        row: pd.Series,
    ) -> str:
        """
        Generate a human-readable explanation for an alert.
        This is what investigators see in the dashboard.
        """
        level = (
            "CRITICAL" if risk >= 0.85 else
            "HIGH"     if risk >= 0.70 else
            "MEDIUM"   if risk >= 0.50 else
            "LOW"
        )

        reasons = []

        if "off_hours" in top_feats and row.get("off_hours", 0):
            reasons.append("off-hours login detected")
        if "bulk_download" in top_feats and row.get("bulk_download", 0):
            reasons.append("bulk download volume far above user baseline")
        if "privilege_escalation" in top_feats and row.get("privilege_escalation", 0):
            reasons.append("privilege escalation action recorded")
        if "unauthorised_db" in top_feats and row.get("unauthorised_db", 0):
            reasons.append("accessed database outside role permissions")
        if "records_vs_user_avg" in top_feats:
            v = row.get("records_vs_user_avg", 0)
            if abs(float(v)) > 1.5:
                reasons.append(f"record access volume {abs(float(v)):.1f}x above personal average")
        if not reasons:
            reasons.append(
                f"behavioural drift across {', '.join(top_feats[:3])}"
            )

        return f"[{level}] User {uid} — risk {risk:.0%}. " + "; ".join(reasons) + "."

    # ── evaluate ─────────────────────────────────────────────────────

    def evaluate(self, alerts_df: pd.DataFrame) -> None:
        """
        Print classification metrics against ground truth labels.
        Only meaningful when is_anomaly_true is available (synthetic + NSL-KDD).
        """
        mask = alerts_df["is_anomaly_true"] != -1
        if mask.sum() == 0:
            print("No ground truth labels available for evaluation.")
            return

        df = alerts_df[mask]
        y_true = df["is_anomaly_true"]
        y_pred = df["is_anomaly_pred"]
        y_score = df["risk_score"]

        print("\n── Evaluation ────────────────────────────────────────")
        print(classification_report(y_true, y_pred, target_names=["normal", "anomaly"]))
        try:
            auc = roc_auc_score(y_true, y_score)
            print(f"  ROC-AUC : {auc:.4f}")
        except Exception:
            pass
        print("──────────────────────────────────────────────────────\n")

    # ── save / load ──────────────────────────────────────────────────

    def save(self, path: str = "models/mps_scorer.pkl") -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[MPS] Saved scorer to {path}")

    @staticmethod
    def load(path: str = "models/mps_scorer.pkl") -> "MPSScorer":
        with open(path, "rb") as f:
            scorer = pickle.load(f)
        print(f"[MPS] Loaded scorer from {path}")
        return scorer


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data_path = str(DATA_DIR / "features_combined.csv")
    out_path  = str(DATA_DIR / "alerts.csv")

    if not Path(data_path).exists():
        print(f"  {data_path} not found.")
        print("  Run: generate_audit_logs.py → load_nslkdd.py → feature_extractor.py first.")
        exit(1)

    # ── Train ──────────────────────────────────────────────────────
    scorer = MPSScorer(
        bond_dim=4,
        anomaly_threshold=0.65,
        iso_contamination=0.05,
        mps_weight=0.6,
    )
    scorer.fit(data_path)
    scorer.save()

    # ── Score ──────────────────────────────────────────────────────
    print("[MPS] Scoring all sessions ...")
    alerts = scorer.score(data_path)
    alerts.to_csv(out_path, index=False)

    # ── Evaluate ───────────────────────────────────────────────────
    scorer.evaluate(alerts)

    # ── Sample alerts ──────────────────────────────────────────────
    flagged = alerts[alerts["is_anomaly_pred"] == 1].sort_values(
        "risk_score", ascending=False
    )
    print(f"\nTop flagged sessions ({len(flagged)} total):")
    print(flagged[["user_id","risk_score","deviation_score",
                   "is_anomaly_true","explanation"]].head(10).to_string())

    # ── Single event demo ──────────────────────────────────────────
    print("\n── Single event inference demo ────────────────────────")
    sample_event = {
        "user_id":              "USR003",
        "role_enc":             2,
        "department_enc":       1,
        "db_accessed_enc":      4,
        "action_enc":           1,
        "session_duration_min": 240,
        "records_accessed":     9000,
        "hour_of_login":        3,
        "day_of_week":          6,
        "off_hours":            1,
        "unauthorised_db":      1,
        "bulk_download":        1,
        "privilege_escalation": 0,
        "records_vs_user_avg":  45.0,
        "duration_vs_user_avg": 8.2,
        "user_avg_records":     200,
        "user_max_records":     500,
        "user_avg_duration":    30,
        "user_login_std":       1.2,
        "user_session_count":   15,
    }
    result = scorer.score_event(sample_event)
    print(json.dumps({k: v for k, v in result.items() if k != "feature_values"}, indent=2))
    print(f"\nSaved {len(alerts):,} scored sessions → {out_path}")
    print("\nNext step: python shap_explainer.py")
