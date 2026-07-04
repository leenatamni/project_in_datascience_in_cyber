"""Shared configuration constants for the phishing detection reproduction project."""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

ARTICLE_DATA_URL = (
    "https://raw.githubusercontent.com/rishy/phishing-websites/"
    "master/Datasets/phising.csv"
)
LOCAL_DATA_PATH = os.path.join(DATA_DIR, "phising.csv")
NEWER_DATA_PATH = os.path.join(DATA_DIR, "newer_phishing_dataset_2020.csv")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 1234
N_CV_SPLITS = 3
TEST_SIZE = 0.25

# ---------------------------------------------------------------------------
# Original article dataset schema (30 features + target, header-less CSV)
# ---------------------------------------------------------------------------
ARTICLE_COLUMNS = [
    "has_ip", "long_url", "short_service", "has_at", "double_slash_redirect",
    "pref_suf", "has_sub_domain", "ssl_state", "long_domain", "favicon",
    "port", "https_token", "req_url", "url_of_anchor", "tag_links", "SFH",
    "submit_to_email", "abnormal_url", "redirect", "mouseover", "right_click",
    "popup", "iframe", "domain_age", "dns_record", "traffic", "page_rank",
    "google_index", "links_to_page", "stats_report", "target",
]

TARGET_COLUMN = "is_phishing"

THRESHOLDS = (0.10, 0.30, 0.50, 0.70, 0.90)
