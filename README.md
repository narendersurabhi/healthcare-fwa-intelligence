# Program Intelligence — FWA Detection (Pre/Post-Pay)

Problem
Detect fraud, waste, and abuse in medical, dental, vision, and pharmacy claims. Reduce false positives and time-to-insight.

Approach
Hybrid rules + ML + GenAI:
- Feature store from PySpark jobs over claims, providers, members, networks.
- Supervised models (XGBoost, LR) + unsupervised (IsolationForest, LOF).
- Network features for provider–member collusion.
- GenAI “investigator brief” using RAG over evidence.

Data (synthetic)
No PHI. Use `data/synthetic_generator.py` to create claim, provider, member tables with label noise.

Key Features
- Temporal: rolling provider velocity, after-hours rates, reversal ratios.
- Network: shared addresses, device IDs, referral triangles.
- Billing: upcoding deltas, unbundling, modifier misuse.
- Member: doctor-shopping, distance anomalies.

Models
- Binary classifier for pre-pay screening.
- Unsupervised anomaly score for discovery.
- Calibrated thresholds by line-of-business.

Results (sample)
- Reviewer efficiency: +35% time savings.
- Prevented improper payments: $15M+ simulated annualized.
- Availability target: 99.9% for online scoring.

What’s next
- Clarify bias/drift checks.
- Human-in-the-loop feedback store.
- Cost per detection dashboards.

Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/synthetic_generator.py
jupyter lab
