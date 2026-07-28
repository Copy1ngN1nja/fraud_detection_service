"""Grafana integration for the adversarial-validation monitoring pipeline.

The actual ROC-AUC and feature-importance heatmaps live in Grafana and are
built directly on top of the ``adversarial_validation_results`` Postgres
table that ``feature_validator.py`` writes to (see the provisioned
datasource/dashboard under ``/grafana/provisioning`` at the repo root, wired
up by the root ``docker-compose.yml``). Grafana queries Postgres itself, so
no metric-pushing is required to draw the heatmaps.

This module's job is to "hand off" each freshly computed adversarial
validation result to Grafana as soon as it is available, by creating a
Grafana annotation on the dashboard's timeline via the HTTP Annotations API.
This drops a marker on both heatmaps every time a new batch of transactions
triggered a validation round, without Grafana having to poll for it.
"""

import logging
import os

import requests

logger = logging.getLogger(__name__)

GRAFANA_URL = os.getenv("GRAFANA_URL", "http://grafana:3000")
GRAFANA_API_KEY = os.getenv("GRAFANA_API_KEY", "")
GRAFANA_DASHBOARD_UID = os.getenv("GRAFANA_DASHBOARD_UID", "adversarial-validation")
ANNOTATION_TAGS = ["adversarial-validation", "fraud-detector"]
REQUEST_TIMEOUT_SECONDS = 5


def _format_annotation_text(results: dict) -> str:
    features = ", ".join(
        f"{f['feature']} ({f['importance_pct']:.1f}%)" for f in results["top_features"]
    )
    psi = results.get("psi_target")
    psi_text = f"{psi:.3f}" if psi is not None else "n/a"
    return (
        f"Adversarial validation round: ROC-AUC={results['roc_auc']:.3f}, "
        f"target PSI={psi_text} "
        f"(current={results['n_current']}, historical={results['n_historical']}). "
        f"Top drifting features: {features}"
    )


def push_results_to_grafana(results: dict) -> bool:
    """Push a Grafana annotation marking a completed adversarial-validation round.

    Returns ``True`` if the annotation was created, ``False`` otherwise (e.g.
    Grafana is unreachable or no API key is configured). Never raises -- a
    failure here must not interrupt the scoring pipeline.
    """
    if not GRAFANA_API_KEY:
        logger.info(
            "GRAFANA_API_KEY not set, skipping annotation push. The "
            "ROC-AUC/feature-importance heatmaps are still updated because "
            "Grafana reads them straight from Postgres."
        )
        return False

    payload = {
        "dashboardUID": GRAFANA_DASHBOARD_UID,
        "time": int(results["run_at"].timestamp() * 1000),
        "tags": ANNOTATION_TAGS,
        "text": _format_annotation_text(results),
    }

    try:
        response = requests.post(
            f"{GRAFANA_URL}/api/annotations",
            json=payload,
            headers={"Authorization": f"Bearer {GRAFANA_API_KEY}"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException:
        logger.warning("Failed to push annotation to Grafana at %s", GRAFANA_URL, exc_info=True)
        return False

    logger.info(
        "Pushed adversarial-validation annotation to Grafana (ROC-AUC=%.3f)",
        results["roc_auc"],
    )
    return True
