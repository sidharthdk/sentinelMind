"""
SentinelMind — Synthetic Banking Audit Log Generator
Generates realistic privileged user behaviour logs for insider threat detection.
Produces: audit_logs.csv with normal + injected anomalous sessions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import uuid

random.seed(42)
np.random.seed(42)

# ─── CONFIG ───────────────────────────────────────────────────────────────────

NUM_USERS = 40
NUM_DAYS = 30
ANOMALOUS_USER_IDS = [3, 11, 27]   # users who will exhibit fraud behaviour

ROLES = {
    "teller":          {"db_access": ["customer_db"],                          "max_records": 50,   "work_hours": (9, 17)},
    "loan_officer":    {"db_access": ["loan_db", "customer_db"],               "max_records": 100,  "work_hours": (9, 18)},
    "dba":             {"db_access": ["core_db", "audit_db", "customer_db"],   "max_records": 500,  "work_hours": (8, 20)},
    "risk_analyst":    {"db_access": ["loan_db", "treasury_db"],               "max_records": 200,  "work_hours": (9, 18)},
    "branch_manager":  {"db_access": ["customer_db", "loan_db"],               "max_records": 150,  "work_hours": (8, 18)},
    "sysadmin":        {"db_access": ["core_db", "audit_db", "treasury_db",
                                       "customer_db", "loan_db"],              "max_records": 1000, "work_hours": (0, 23)},
}

ALL_DBS = ["core_db", "audit_db", "customer_db", "loan_db", "treasury_db"]

ACTIONS = ["SELECT", "EXPORT", "UPDATE", "DELETE", "LOGIN", "LOGOUT",
           "PRIVILEGE_CHANGE", "BULK_DOWNLOAD", "SCHEMA_ACCESS"]

# ─── USER SETUP ───────────────────────────────────────────────────────────────

def create_users(n):
    role_list = list(ROLES.keys())
    users = []
    for i in range(n):
        role = random.choice(role_list)
        users.append({
            "user_id": f"USR{i:03d}",
            "role": role,
            "department": random.choice(["Mumbai_Branch", "Delhi_HQ", "Chennai_Branch",
                                          "Pune_Branch", "Kolkata_Branch"]),
        })
    return users

# ─── NORMAL SESSION GENERATOR ─────────────────────────────────────────────────

def normal_session(user, date):
    role_cfg = ROLES[user["role"]]
    wh = role_cfg["work_hours"]

    # Login time — gaussian around midday of work hours
    mean_hour = (wh[0] + wh[1]) / 2
    login_hour = int(np.clip(np.random.normal(mean_hour, 1.5), wh[0], wh[1] - 1))
    login_minute = random.randint(0, 59)
    login_time = date.replace(hour=login_hour, minute=login_minute, second=0)
    session_duration = random.randint(20, 240)  # minutes
    logout_time = login_time + timedelta(minutes=session_duration)

    # DB accessed — only from allowed DBs
    db = random.choice(role_cfg["db_access"])

    # Action — mostly SELECT, occasional UPDATE
    action = random.choices(
        ["SELECT", "UPDATE", "LOGIN", "LOGOUT", "EXPORT"],
        weights=[60, 15, 10, 10, 5]
    )[0]

    records = random.randint(1, role_cfg["max_records"])

    return {
        "event_id": str(uuid.uuid4())[:8],
        "user_id": user["user_id"],
        "role": user["role"],
        "department": user["department"],
        "login_time": login_time,
        "logout_time": logout_time,
        "session_duration_min": session_duration,
        "db_accessed": db,
        "action": action,
        "records_accessed": records,
        "off_hours": 0,
        "unauthorised_db": 0,
        "bulk_download": 0,
        "privilege_escalation": 0,
        "is_anomaly": 0,
    }

# ─── ANOMALY INJECTORS ────────────────────────────────────────────────────────

def inject_off_hours_login(user, date):
    """Login at 2-4am, outside any normal work window."""
    event = normal_session(user, date)
    login_time = date.replace(hour=random.randint(2, 4), minute=random.randint(0, 59))
    event.update({
        "login_time": login_time,
        "logout_time": login_time + timedelta(minutes=random.randint(30, 90)),
        "off_hours": 1,
        "is_anomaly": 1,
    })
    return event

def inject_bulk_download(user, date):
    """Accesses 5-20x normal record volume."""
    event = normal_session(user, date)
    max_normal = ROLES[user["role"]]["max_records"]
    event.update({
        "records_accessed": random.randint(max_normal * 5, max_normal * 20),
        "action": "BULK_DOWNLOAD",
        "bulk_download": 1,
        "is_anomaly": 1,
    })
    return event

def inject_unauthorised_db(user, date):
    """Accesses a DB outside their role's allowed list."""
    event = normal_session(user, date)
    allowed = ROLES[user["role"]]["db_access"]
    forbidden = [db for db in ALL_DBS if db not in allowed]
    if not forbidden:
        forbidden = ["core_db"]
    event.update({
        "db_accessed": random.choice(forbidden),
        "unauthorised_db": 1,
        "is_anomaly": 1,
    })
    return event

def inject_privilege_escalation(user, date):
    """Performs PRIVILEGE_CHANGE or SCHEMA_ACCESS."""
    event = normal_session(user, date)
    event.update({
        "action": random.choice(["PRIVILEGE_CHANGE", "SCHEMA_ACCESS", "DELETE"]),
        "db_accessed": random.choice(["core_db", "audit_db"]),
        "privilege_escalation": 1,
        "unauthorised_db": 1,
        "is_anomaly": 1,
    })
    return event

ANOMALY_INJECTORS = [
    inject_off_hours_login,
    inject_bulk_download,
    inject_unauthorised_db,
    inject_privilege_escalation,
]

# ─── MAIN GENERATION LOOP ─────────────────────────────────────────────────────

def generate_logs(users, num_days):
    logs = []
    start_date = datetime(2025, 1, 1)

    for day_offset in range(num_days):
        date = start_date + timedelta(days=day_offset)
        is_weekend = date.weekday() >= 5

        for idx, user in enumerate(users):
            # Skip weekends with 80% probability (banks do have weekend shifts)
            if is_weekend and random.random() < 0.80:
                continue

            # Normal sessions: 1-3 per day
            num_sessions = random.randint(1, 3)
            for _ in range(num_sessions):
                logs.append(normal_session(user, date))

            # Inject anomalies for flagged users
            if idx in ANOMALOUS_USER_IDS:
                # Anomaly probability increases over time (simulates behavioural drift)
                anomaly_prob = 0.05 + (day_offset / num_days) * 0.40
                if random.random() < anomaly_prob:
                    injector = random.choice(ANOMALY_INJECTORS)
                    logs.append(injector(user, date))

    return logs

# ─── RUN & SAVE ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating users...")
    users = create_users(NUM_USERS)

    print(f"Generating audit logs for {NUM_DAYS} days, {NUM_USERS} users...")
    logs = generate_logs(users, NUM_DAYS)

    df = pd.DataFrame(logs)
    df = df.sort_values("login_time").reset_index(drop=True)

    # Derived features useful for MPS/ML pipeline
    df["hour_of_login"] = df["login_time"].dt.hour
    df["day_of_week"] = df["login_time"].dt.dayofweek
    df["login_time"] = df["login_time"].astype(str)
    df["logout_time"] = df["logout_time"].astype(str)

    output_path = "data/audit_logs.csv"
    import os; os.makedirs("data", exist_ok=True)
    df.to_csv(output_path, index=False)

    total = len(df)
    anomalies = df["is_anomaly"].sum()
    print(f"\n Done. {total} events written to {output_path}")
    print(f"   Normal events   : {total - anomalies}")
    print(f"   Anomalous events: {anomalies} ({100*anomalies/total:.1f}%)")
    print(f"   Anomalous users : USR{ANOMALOUS_USER_IDS[0]:03d}, USR{ANOMALOUS_USER_IDS[1]:03d}, USR{ANOMALOUS_USER_IDS[2]:03d}")
    print(f"\nColumns: {list(df.columns)}")
