# Healthcare FWA Intelligence — End-to-End Example (Pre/Post-Pay)

Problem
Detect fraud, waste, and abuse in medical, dental, vision, and pharmacy claims while limiting false positives.

Approach
Hybrid rules + ML + GenAI (synthetic only):
- Feature store with PySpark over claims, providers, members, and simple network joins.
- Supervised models (XGBoost, Logistic Regression) + unsupervised (IsolationForest/LOF).
- Lightweight network features for potential collusion patterns.
- RAG-based “investigator brief” over retrieved evidence.

Data
Synthetic only. Use `data/synthetic_generator.py`. No PHI/PII.

Key Features
- Temporal velocities, after-hours rates, reversal ratios.
- Network co-occurrence signals.
- Billing deltas for upcoding/unbundling patterns.
- Member utilization anomalies.

Models
- Binary classifier for pre-pay triage.
- Unsupervised anomaly score for discovery.
- Calibrated thresholds by line of business.

Results (demo)
- Reviewer time reduced in simulations.
- Improper payment reduction estimated on synthetic baselines.
- Target availability for online scoring: 99.9%.

Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/synthetic_generator.py
jupyter lab
