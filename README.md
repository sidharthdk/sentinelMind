# SentinelMind

### Quantum-Inspired Insider Threat Detection for Public Sector Banks

**AI-Driven Early Warning System for Internal & Privileged User Fraud**

**Team Wolf Pack | iDEA 2.0 PSBs Hackathon 2026**

---

## Overview

SentinelMind is a quantum-inspired behavioral monitoring system designed to detect insider threats and privileged user fraud in banking environments before significant damage occurs.

The system builds behavioral baselines for individual users and roles using **Matrix Product States (MPS)**, a tensor-network technique inspired by quantum physics. It continuously monitors activity across banking systems and identifies deviations that may indicate malicious intent, credential misuse, privilege abuse, or emerging insider threats.

Unlike traditional anomaly detection systems that provide only risk scores, SentinelMind generates **SHAP-based explanations** for every alert, enabling investigators to understand exactly why a user was flagged.

---

## Problem Statement

This project addresses:

**PS1: AI-Driven Early Warning System for Internal & Privileged User Fraud**

Public sector banks generate enormous volumes of audit logs from:

* Core Banking Systems (CBS)
* Treasury Operations
* Loan Origination Platforms
* Customer Databases
* Administrative Systems

Traditional rule-based monitoring systems struggle to detect subtle behavioral drift that often precedes fraud or privilege abuse.

SentinelMind addresses this challenge through behavioral modeling, anomaly detection, and explainable AI.

---

## Key Features

### Quantum-Inspired Behavioral Modeling

* Uses Matrix Product States (MPS) to model user activity sequences.
* Learns individual and role-specific behavior patterns.
* Detects behavioral drift in real time.

### Explainable AI

* SHAP explanations for every anomaly.
* Evidence-grade feature attribution.
* Transparent decision-making process.

### Role-Aware Detection

Supports multiple banking roles:

* Branch Tellers
* Relationship Managers
* System Administrators
* IT / Backend Support Staff

### Early Fraud Detection

Identifies:

* Unusual login behavior
* Off-hour system access
* Transaction volume spikes
* Suspicious data downloads
* Rare role-action combinations
* Behavioral drift over time

### Interactive Dashboard

* Risk scoring visualization
* Alert investigation workflow
* SHAP explanation views
* User behavioral analytics

---

## System Architecture

```text
Audit Logs
      │
      ▼
Feature Engineering
      │
      ▼
Behavioral Sequence Generation
      │
      ▼
Matrix Product States (MPS)
Behavior Modeling
      │
      ▼
Anomaly Scoring
      │
      ▼
SHAP Explainability Engine
      │
      ▼
Risk Dashboard & Alerts
```

---

## Technology Stack

| Component           | Technology                     |
| ------------------- | ------------------------------ |
| Language            | Python 3.10+                   |
| Behavioral Modeling | TensorNetwork (MPS)            |
| Explainability      | SHAP                           |
| Machine Learning    | Scikit-learn                   |
| Dashboard           | Streamlit                      |
| Data Processing     | Pandas, NumPy                  |
| Visualization       | Plotly                         |
| Data Generation     | Custom Synthetic Log Generator |

---

## Project Structure

```text
sentinelmind/
│
├── data/
│   ├── synthetic logs
│   └── sample datasets
│
├── models/
│   ├── trained MPS models
│   └── SHAP explainers
│
├── notebooks/
│   ├── exploratory analysis
│   ├── experiments
│   └── SHAP visualizations
│
├── app.py
├── generate_synthetic_logs.py
├── train_mps_baseline.py
├── evaluate.py
├── requirements.txt
└── README.md
```

---

## Dataset

### Synthetic Banking Activity Dataset

To ensure privacy and regulatory compliance, SentinelMind uses a custom synthetic data generator.

### Simulated Attributes

#### Authentication Behavior

* Login timestamps
* Session duration
* Failed login attempts
* Off-hour access activity

#### Transaction Behavior

* Transaction frequency
* Transaction volume
* Approval patterns

#### Data Access Behavior

* Account lookups
* Report generation
* Bulk downloads
* Customer record access

#### Role-Based Profiles

Each role has unique baseline behavior patterns with injected anomalies for evaluation.

### Dataset Characteristics

| Metric             | Value     |
| ------------------ | --------- |
| Users              | 1,000     |
| Observation Period | 90 Days   |
| Roles              | 4         |
| Injected Anomalies | 5%        |
| Data Type          | Synthetic |

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-team/sentinelmind.git
cd sentinelmind
```

### Create Virtual Environment

Linux / macOS:

```bash
python -m venv venv
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Step 1: Generate Synthetic Logs

```bash
python generate_synthetic_logs.py
```

### Step 2: Train Behavioral Models

```bash
python train_mps_baseline.py
```

### Step 3: Evaluate Performance

```bash
python evaluate.py
```

### Step 4: Launch Dashboard

```bash
streamlit run app.py
```

### Step 5: Open Browser

```text
http://localhost:8501
```

---

## Performance Results

### MPS Behavioral Baseline + Drift Detection

| Metric              | Score       |
| ------------------- | ----------- |
| AUC-ROC             | 0.93        |
| Precision           | 0.87        |
| Recall              | 0.82        |
| F1 Score            | 0.84        |
| False Positive Rate | 4.8%        |
| Detection Lag       | ~2.1 Events |

### Explainability

Every alert includes:

* Feature importance ranking
* SHAP contribution values
* Behavioral deviation explanation
* Risk evidence summary

---

## Example Threat Scenarios

### Scenario 1

A Relationship Manager suddenly downloads hundreds of customer records outside normal working hours.

**Detected Signals**

* Off-hour access
* Data volume spike
* Unusual access frequency

### Scenario 2

A System Administrator begins accessing customer accounts unrelated to their responsibilities.

**Detected Signals**

* Role-action mismatch
* Rare access pattern
* Behavioral drift

### Scenario 3

Compromised privileged credentials initiate abnormal transactions.

**Detected Signals**

* Sequence deviation
* Transaction anomaly
* Elevated risk score

---

## Future Roadmap

### Phase 2

* Multi-system correlation

  * CBS
  * CRM
  * Email systems
  * HR systems

* Real-time streaming ingestion

  * Apache Kafka
  * Amazon Kinesis

* Graph-based insider threat detection

* Hybrid Quantum-Classical architectures

* Federated learning for privacy-preserving deployment

---

## Limitations

* Currently trained on synthetic data only.
* Requires validation using anonymized production audit logs.
* Cross-system behavioral correlation is not yet implemented.
* MPS computational complexity increases with sequence length.
* Current dashboard supports batch CSV processing.

---

## Team Wolf Pack

| Name          | Role                 | Contribution                                                       |
| ------------- | -------------------- | ------------------------------------------------------------------ |
| Sidharth D    | ML & Quantum ML Lead | MPS design, training pipeline, drift detection, architecture       |
| S Sidharthan, Tharshan  | ML & Data Science    | Feature engineering, synthetic data generation, evaluation, SHAP   |
| Vishweshwar R | Backend & Java ML    | Dashboard backend, ingestion pipeline, integrations, documentation |

**Institution:** St. Joseph's Institute of Technology, Chennai

**Track:** AI – CSPARC

**Competition:** iDEA 2.0 PSBs Hackathon Series 2026

---

## Contact

**Team:** Wolf Pack

**Project:** SentinelMind

**Email:** [sidharthdk73@gmail.com](mailto:sidharthdk73@gmail.com)

**GitHub:** https://github.com/sidharthdk

---

## Acknowledgements

* Union Bank of India
* Ministry of Finance, Government of India
* iDEA 2.0 PSBs Hackathon
* Open-source AI and Quantum Computing communities

---

**Building secure, explainable, and future-ready banking systems for India.**
