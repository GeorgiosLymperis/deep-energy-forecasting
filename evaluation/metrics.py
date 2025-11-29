"""
Proper scoring rules, dependence metrics, and statistical comparison utilities
for probabilistic forecasting models.

This module provides:
    - CRPS per marginal and per batch
    - Energy Score per marginal and per batch
    - Variogram Score
    - Quantile Scores (pinball loss)
    - Diebold–Mariano test (with Newey–West correction)
    - ExtraTrees-based realism ROC test

All functions are NumPy-based and work on Monte Carlo scenarios produced by
generative models (NF, VAE, GAN).

Expected Shapes
---------------
- scenarios marginal:          (S, T)
- scenarios batch:             (S, B, T)
- y_true marginal:             (T,)
- y_true batch:                (B, T)
- quantiles marginal:          (99, T)
- quantiles batch:             (99, B, T)
"""

from typing import List, Tuple, Dict
import numpy as np
from scipy.stats import t
from scipy.signal import correlate
from sklearn.ensemble import ExtraTreesClassifier
from itertools import combinations
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc


# =====================================================================
# CRPS
# =====================================================================

def crps_per_marginal(scenarios: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Compute CRPS per marginal (per time step).

    Args:
        scenarios (np.ndarray):  
            Array of shape ``(S, T)`` containing S sampled scenarios.
        y_true (np.ndarray):  
            True observed values, shape ``(T,)``.

    Returns:
        np.ndarray:  
            CRPS values per time step, shape ``(T,)``.
    """
    term1 = np.mean(np.abs(scenarios - y_true[None, :]), axis=0)
    diffs = np.abs(scenarios[:, None, :] - scenarios[None, :, :])
    term2 = 0.5 * np.mean(diffs, axis=(0, 1))
    return term1 - term2


def crps_batch_per_marginal(
    scenarios: np.ndarray,
    y_true: np.ndarray,
) -> List[np.ndarray]:
    """Compute CRPS per marginal over a batch.

    Args:
        scenarios (np.ndarray):  
            Shape ``(S, B, T)`` where S = number of scenarios.
        y_true (np.ndarray):  
            True targets, shape ``(B, T)``.
        calculate_mean (bool):
            Unused (kept for compatibility).

    Returns:
        List[np.ndarray]:
            List of CRPS arrays (each shape ``(T,)``) per batch element.
    """
    crps_list = []
    for i in range(y_true.shape[0]):
        crps_list.append(crps_per_marginal(scenarios[:, i, :], y_true[i, :]))
    return crps_list


# =====================================================================
# ENERGY SCORE
# =====================================================================

def energy_score(scenarios: np.ndarray, y_true: np.ndarray) -> float:
    """Compute the multivariate Energy Score.

    Args:
        scenarios (np.ndarray):  
            Monte Carlo samples, shape ``(S, T)``.
        y_true (np.ndarray):  
            Ground truth values, shape ``(T,)``.

    Returns:
        float: Energy Score.
    """
    term_1 = np.linalg.norm(scenarios - y_true[None, :], axis=1).mean()
    term_2 = np.linalg.norm(
        scenarios[:, None, :] - scenarios[None, :, :], axis=2
    ).mean()
    return term_1 - 0.5 * term_2


def energy_score_per_batch(
    scenarios: np.ndarray,
    y_true: np.ndarray
) -> List[float]:
    """Compute Energy Score over a batch.

    Args:
        scenarios (np.ndarray):  
            Samples of shape ``(S, B, T)``.
        y_true (np.ndarray):  
            Ground truth, shape ``(B, T)``.

    Returns:
        List[float]: Energy Score per batch element.
    """
    scores = []
    for i in range(y_true.shape[0]):
        scores.append(energy_score(scenarios[:, i, :], y_true[i, :]))
    return scores


# =====================================================================
# VARIOGRAM SCORE
# =====================================================================

def variogram_score(
    scenarios: np.ndarray,
    y_true: np.ndarray,
    gamma: float = 0.5
) -> float:
    """Compute Variogram Score for multivariate forecasts.

    Args:
        scenarios (np.ndarray):  
            Samples, shape ``(S, T)``.
        y_true (np.ndarray):  
            True vector, shape ``(T,)``.
        gamma (float):
            Exponent parameter (commonly 0.5 or 1).

    Returns:
        float: Variogram Score.
    """
    diffs_true = np.abs(y_true[None, :] - y_true[:, None]) ** gamma
    diffs_sample = np.abs(
        scenarios[:, :, None] - scenarios[:, None, :]
    ) ** gamma
    expected_diffs = np.mean(diffs_sample, axis=0)
    return float(np.sum((diffs_true - expected_diffs) ** 2))


def variogram_score_per_batch(
    scenarios: np.ndarray,
    y_true: np.ndarray,
    gamma: float = 0.5
) -> List[float]:
    """Compute Variogram Score for a batch.

    Args:
        scenarios (np.ndarray):  
            Shape ``(S, B, T)``.
        y_true (np.ndarray):  
            Shape ``(B, T)``.
        gamma (float):
            Variogram exponent.

    Returns:
        List[float]: Variogram Score per batch element.
    """
    scores = [
        variogram_score(scenarios[:, i, :], y_true[i, :], gamma)
        for i in range(y_true.shape[0])
    ]
    return scores


# =====================================================================
# QUANTILE SCORES (PINBALL LOSS)
# =====================================================================

def quantile_score_per_marginal(
    quantiles: np.ndarray,
    y_true: np.ndarray,
    q: int = 50
) -> np.ndarray:
    """Compute pinball quantile score for one quantile and one example.

    Args:
        quantiles (np.ndarray):  
            Array of shape ``(99, T)`` (1% ... 99% quantiles).
        y_true (np.ndarray):  
            True values, shape ``(T,)``.
        q (int):
            Quantile index (1–99).

    Returns:
        np.ndarray:
            Pinball loss per time step, shape ``(T,)``.
    """
    assert quantiles.shape[0] == 99
    assert 1 <= q <= 99
    diff = quantiles[q - 1, :] - y_true
    return np.where(diff > 0, (1 - q / 100) * diff, -(q / 100) * diff)


def quantile_score_per_batch(
    quantiles: np.ndarray,
    y_true: np.ndarray,
    q: int = 50
) -> List[np.ndarray]:
    """Compute pinball score for a batch.

    Args:
        quantiles (np.ndarray):  
            Shape ``(99, B, T)``.
        y_true (np.ndarray):  
            Shape ``(B, T)``.
        q (int):
            Quantile index (1–99).

    Returns:
        List[np.ndarray]: List of arrays (shape ``(T,)``).
    """
    return [
        quantile_score_per_marginal(quantiles[:, i, :], y_true[i, :], q)
        for i in range(y_true.shape[0])
    ]


def quantile_score_averaged(
    quantiles: np.ndarray,
    y_true: np.ndarray
) -> float:
    """Compute average quantile score over q = 1..99.

    Args:
        quantiles (np.ndarray): Shape ``(99, B, T)``.
        y_true (np.ndarray): Shape ``(B, T)``.

    Returns:
        float: Average quantile score.
    """
    return float(
        np.mean([quantile_score_per_batch(quantiles, y_true, q)
                 for q in range(1, 100)])
    )


def quantile_score_averaged_fast(
    quantiles: np.ndarray,
    y_true: np.ndarray
) -> float:
    """Vectorized quantile score over q=1..99.

    Args:
        quantiles (np.ndarray):  
            Shape ``(99, B, T)``.
        y_true (np.ndarray):  
            Shape ``(B, T)``.

    Returns:
        float: Mean quantile score over all q, batch, and time.
    """
    qs = np.arange(1, 100).reshape(-1, 1, 1) / 100
    diff = quantiles - y_true[None, :, :]
    loss = np.where(diff >= 0, (1 - qs) * diff, -qs * diff)
    return float(loss.mean())


# =====================================================================
# DIEBOLD–MARIANO TEST
# =====================================================================

def diebold_mariano_test(
    errors_g: np.ndarray,
    errors_h: np.ndarray,
    h: int = 1,
    eps: float = 1e-12
) -> Tuple[float, float]:
    """Compute Diebold–Mariano test with Newey–West correction.

    Args:
        errors_g (np.ndarray):
            Errors from model g. Shape: ``(N,)`` or ``(N, H)``.
        errors_h (np.ndarray):
            Errors from model h. Same shape rules as errors_g.
        h (int):
            Newey–West truncation lag (commonly horizon - 1).
        eps (float):
            Numerical stability for variance floor.

    Returns:
        Tuple[float, float]:
            - DM statistic (float)
            - p-value (float)
    """
    eg = np.asarray(errors_g)
    eh = np.asarray(errors_h)

    if eg.ndim > 1:
        lg = np.sum(np.abs(eg), axis=1)
    else:
        lg = np.abs(eg)

    if eh.ndim > 1:
        lh = np.sum(np.abs(eh), axis=1)
    else:
        lh = np.abs(eh)

    d = (lg - lh).ravel()
    Tn = d.size
    d_mean = d.mean()
    d_centered = d - d_mean

    ac = correlate(d_centered, d_centered, mode="full", method="fft") / Tn
    mid = ac.size // 2
    gamma = ac[mid - h: mid + h + 1]

    f_d = gamma[0] + 2 * gamma[1:].sum()
    f_d = max(f_d, eps)

    DM_stat = d_mean / np.sqrt(f_d / Tn)

    corr = np.sqrt((Tn + 1 - 2*h + h*(h - 1)/Tn) / Tn)
    DM_corr = DM_stat * corr

    p_value = float(2 * t.cdf(-abs(DM_corr), df=Tn - 1))
    return float(DM_corr), p_value


def dm_pvalue_matrix(
    loss_dict: Dict[str, np.ndarray],
    labels: List[str],
    h: int = 1
) -> pd.DataFrame:
    """Compute a matrix of pairwise DM p-values.

    Args:
        loss_dict (Dict[str, np.ndarray]):
            Mapping model_name -> error_array.
        labels (List[str]):
            Ordered list of model names.
        h (int):
            Newey–West lag.

    Returns:
        pd.DataFrame:
            NxN matrix of p-values.
    """
    n = len(labels)
    p_values = np.full((n, n), np.nan)

    for m1, m2 in combinations(labels, 2):
        _, pval = diebold_mariano_test(loss_dict[m1], loss_dict[m2], h=h)
        i, j = labels.index(m1), labels.index(m2)
        p_values[i, j] = pval
        p_values[j, i] = pval

    return pd.DataFrame(p_values, index=labels, columns=labels)


# =====================================================================
# CLASSIFIER-BASED REALISM METRICS
# =====================================================================

def fit_trees(X: np.ndarray, y: np.ndarray, **kwargs) -> ExtraTreesClassifier:
    """Fit an ExtraTrees classifier.

    Args:
        X (np.ndarray): Feature matrix, shape ``(N, T)``.
        y (np.ndarray): Labels, shape ``(N,)``.
        **kwargs: Hyperparameters for ExtraTreesClassifier.

    Returns:
        ExtraTreesClassifier: Fitted classifier.
    """
    params = {
        "n_estimators": kwargs.get("n_estimators", 300),
        "max_depth": kwargs.get("max_depth", None),
        "min_samples_leaf": kwargs.get("min_samples_leaf", 1),
        "n_jobs": kwargs.get("n_jobs", -1),
        "random_state": kwargs.get("random_state", 42),
        "class_weight": kwargs.get("class_weight", None),
    }
    clf = ExtraTreesClassifier(**params)
    clf.fit(X, y)
    return clf


def roc_for_real_vs_fake(
    X_real: np.ndarray,
    X_fake: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 42,
    **tree_kwargs
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute ROC curve for distinguishing real from fake scenarios.

    Args:
        X_real (np.ndarray):  
            Real samples, shape ``(N_real, T)``.
        X_fake (np.ndarray):  
            Generated samples, shape ``(N_fake, T)``.
        test_size (float):
            Fraction of samples for test split.
        random_state (int):
            Random seed for train-test split.
        **tree_kwargs:
            Passed to `fit_trees`.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]:
            - fpr: False positive rate array
            - tpr: True positive rate array
            - auc_value: Area under ROC curve
    """
    X = np.vstack([X_real, X_fake])
    y = np.r_[np.ones(len(X_real), dtype=int), np.zeros(len(X_fake), dtype=int)]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    clf = fit_trees(X_train, y_train, **tree_kwargs)
    y_score = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    return fpr, tpr, auc(fpr, tpr)
