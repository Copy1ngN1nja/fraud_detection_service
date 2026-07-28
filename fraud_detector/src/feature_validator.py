"""Adversarial-validation feature-drift detector.

Buffers processed (feature-space) transactions as they are scored by the
fraud detection service. Once ``BUFFER_SIZE`` observations have accumulated,
they are flushed to Postgres and used, together with a random sample of the
historical data uploaded through the interface service (see
``interface/app.py``), to train a binary classifier that tries to tell
"current" (freshly scored) transactions apart from "historical" ones -- the
classic adversarial-validation trick for detecting feature/data drift.

The resulting ROC-AUC (how easily the two datasets can be told apart --
close to 0.5 means no drift, close to 1.0 means strong drift), the top-3
most important features driving that separation, and the Population
Stability Index (PSI) between the model's score distribution on the current
batch vs. the historical baseline are persisted to Postgres, where they are
picked up by ``visualizer.py`` / Grafana for monitoring.
"""

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine, text

from scorer import BOOSTER
from visualizer import push_results_to_grafana

logger = logging.getLogger(__name__)

BUFFER_SIZE = int(os.getenv("ADV_VALIDATION_BUFFER_SIZE", "1000"))
TOP_N_FEATURES = 3
MIN_HISTORICAL_ROWS = 10
RANDOM_STATE = 42
PSI_BINS = 10

CURRENT_TABLE = "current_transactions"
HISTORICAL_TABLE = "historical_transactions"
RESULTS_TABLE = "adversarial_validation_results"

# Columns that may be present in either table but are not model features.
NON_FEATURE_COLUMNS = {"id", "created_at", "uploaded_at", "transaction_id"}


def _population_stability_index(expected: np.ndarray, actual: np.ndarray, n_bins: int = PSI_BINS) -> float:
    """PSI between an 'expected' (historical/baseline) and 'actual' (current) distribution.

    Bin edges are derived from the quantiles of ``expected`` (the standard
    approach), so PSI == 0 when the two distributions coincide.
    Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 significant shift.
    """
    eps = 1e-6
    bin_edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    if len(bin_edges) < 3:
        # Baseline distribution is degenerate (e.g. near-constant) -- not enough
        # spread to bin meaningfully.
        return 0.0
    bin_edges[0], bin_edges[-1] = -np.inf, np.inf

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_pct = expected_counts / max(len(expected), 1) + eps
    actual_pct = actual_counts / max(len(actual), 1) + eps

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def _get_engine():
    url = (
        f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:"
        f"{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:"
        f"{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
    )
    return create_engine(url)


def _ensure_results_table(engine) -> None:
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {RESULTS_TABLE} (
                id SERIAL PRIMARY KEY,
                run_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                n_current INT NOT NULL,
                n_historical INT NOT NULL,
                roc_auc FLOAT NOT NULL,
                top1_feature TEXT,
                top1_importance FLOAT,
                top2_feature TEXT,
                top2_importance FLOAT,
                top3_feature TEXT,
                top3_importance FLOAT,
                psi_target FLOAT
            );
        """))
        conn.execute(text(f"ALTER TABLE {RESULTS_TABLE} ADD COLUMN IF NOT EXISTS psi_target FLOAT;"))


class AdversarialFeatureValidator:
    """Accumulates scored transactions and periodically runs adversarial validation."""

    def __init__(self, buffer_size: int = BUFFER_SIZE):
        self.buffer_size = buffer_size
        self._buffer: list[pd.DataFrame] = []
        self._scores: list[float] = []
        self._lock = threading.Lock()
        # Раунд adversarial-валидации (обучение LGBM + запись в Postgres) -
        # тяжёлая операция, гонять её надо не в потоке Kafka-consumer'а,
        # иначе на каждые BUFFER_SIZE сообщений он замирает на время обучения.
        # max_workers=1: раунды выполняются по одному, не конкурируют за CPU.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="adv-validation")

    def observe(self, processed_df: pd.DataFrame, score) -> None:
        """Register a batch of already-preprocessed feature rows plus the
        model score(s) predicted for them (``score`` is a scalar or an
        array-like with the same length as ``processed_df``).

        Once the internal buffer reaches ``self.buffer_size`` rows, an
        adversarial-validation round is scheduled on a background thread
        (this call returns immediately either way, without waiting for it).
        """
        scores = np.atleast_1d(np.asarray(score, dtype=float))
        if len(scores) != len(processed_df):
            raise ValueError("`score` must have the same length as `processed_df`")

        with self._lock:
            self._buffer.append(processed_df.copy())
            self._scores.extend(scores.tolist())
            buffered_rows = sum(len(chunk) for chunk in self._buffer)

            if buffered_rows < self.buffer_size:
                return

            current_df = pd.concat(self._buffer, ignore_index=True)
            current_scores = np.array(self._scores)
            self._buffer = []
            self._scores = []

        self._executor.submit(self._run_adversarial_validation_safe, current_df, current_scores)

    def _run_adversarial_validation_safe(self, current_df: pd.DataFrame, current_scores: np.ndarray) -> None:
        try:
            self._run_adversarial_validation(current_df, current_scores)
        except Exception:
            logger.exception("Adversarial validation round failed")

    def _run_adversarial_validation(self, current_df: pd.DataFrame, current_scores: np.ndarray) -> dict | None:
        engine = _get_engine()

        logger.info("Flushing %d current observations to '%s'...", len(current_df), CURRENT_TABLE)
        current_df.to_sql(CURRENT_TABLE, engine, if_exists="append", index=False)

        historical_df = self._sample_historical_data(engine, n=len(current_df))
        if historical_df is None or len(historical_df) < MIN_HISTORICAL_ROWS:
            logger.warning(
                "Not enough historical data in '%s' to run adversarial validation "
                "(need >= %d rows, got %s). Upload historical data via the interface service.",
                HISTORICAL_TABLE, MIN_HISTORICAL_ROWS, 0 if historical_df is None else len(historical_df),
            )
            return None

        common_cols = [
            col for col in current_df.columns
            if col in historical_df.columns and col not in NON_FEATURE_COLUMNS
        ]
        if not common_cols:
            logger.warning("No common feature columns between current and historical data.")
            return None

        X = pd.concat([current_df[common_cols], historical_df[common_cols]], ignore_index=True)
        y = pd.Series([1] * len(current_df) + [0] * len(historical_df))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y,
        )

        model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=5, random_state=RANDOM_STATE, verbosity=-1,
        )
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)

        importances = pd.Series(model.feature_importances_, index=common_cols)
        total_importance = importances.sum() or 1.0
        top_features = [
            {
                "feature": feature,
                "importance": float(importance),
                "importance_pct": float(importance / total_importance * 100),
            }
            for feature, importance in importances.sort_values(ascending=False).head(TOP_N_FEATURES).items()
        ]

        psi_target = self._compute_target_psi(current_scores, historical_df, common_cols)

        results = {
            "run_at": datetime.now(timezone.utc),
            "n_current": len(current_df),
            "n_historical": len(historical_df),
            "roc_auc": float(roc_auc),
            "top_features": top_features,
            "psi_target": psi_target,
        }

        logger.info(
            "Adversarial validation: ROC-AUC=%.4f, PSI(target)=%s, top features=%s",
            roc_auc, psi_target, [f["feature"] for f in top_features],
        )

        self._persist_results(engine, results)
        push_results_to_grafana(results)

        return results

    def _compute_target_psi(
        self, current_scores: np.ndarray, historical_df: pd.DataFrame, common_cols: list[str],
    ) -> float | None:
        """PSI of the model's score distribution: historical data (scored through
        the same booster) as the expected/baseline population, current batch as
        the actual population."""
        try:
            historical_scores = np.asarray(BOOSTER.predict(historical_df[common_cols]))
        except Exception:
            logger.warning("Could not score historical data for target PSI", exc_info=True)
            return None

        return _population_stability_index(expected=historical_scores, actual=current_scores)

    def _sample_historical_data(self, engine, n: int) -> pd.DataFrame | None:
        try:
            return pd.read_sql(
                text(f"SELECT * FROM {HISTORICAL_TABLE} ORDER BY RANDOM() LIMIT :n"),
                engine, params={"n": n},
            )
        except Exception:
            logger.warning(
                "Could not read historical data from '%s' (has it been uploaded yet?)",
                HISTORICAL_TABLE, exc_info=True,
            )
            return None

    def _persist_results(self, engine, results: dict) -> None:
        _ensure_results_table(engine)
        top = results["top_features"] + [{}] * (TOP_N_FEATURES - len(results["top_features"]))

        with engine.begin() as conn:
            conn.execute(
                text(f"""
                    INSERT INTO {RESULTS_TABLE} (
                        run_at, n_current, n_historical, roc_auc,
                        top1_feature, top1_importance,
                        top2_feature, top2_importance,
                        top3_feature, top3_importance,
                        psi_target
                    ) VALUES (
                        :run_at, :n_current, :n_historical, :roc_auc,
                        :f1, :i1, :f2, :i2, :f3, :i3,
                        :psi_target
                    );
                """),
                {
                    "run_at": results["run_at"],
                    "n_current": results["n_current"],
                    "n_historical": results["n_historical"],
                    "roc_auc": results["roc_auc"],
                    "f1": top[0].get("feature"), "i1": top[0].get("importance"),
                    "f2": top[1].get("feature"), "i2": top[1].get("importance"),
                    "f3": top[2].get("feature"), "i3": top[2].get("importance"),
                    "psi_target": results.get("psi_target"),
                },
            )


# Module-level singleton used by the Kafka consumer service (see app/app.py).
validator = AdversarialFeatureValidator()


def observe_transaction(processed_df: pd.DataFrame, score) -> dict | None:
    """Convenience wrapper around ``validator.observe`` for a single call site."""
    return validator.observe(processed_df, score)
