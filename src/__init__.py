"""
Phishing website detection - reproduction and critical evaluation package.

Modules:
    config          - shared constants (paths, random seed, column names)
    data_loading    - loading the article dataset and the newer external dataset
    preprocessing   - target conversion, dedup, group-key construction
    models          - model pipeline definitions
    evaluation      - metric computation, threshold analysis, low-prevalence simulation
    experiments     - end-to-end experiment runners used by scripts/run_all_experiments.py
"""
