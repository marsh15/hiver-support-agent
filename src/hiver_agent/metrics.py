"""All reported numbers live here: intent metrics, escalation metrics,
auto-handle safety, reply-constraint violations, bootstrap CIs, kappa."""
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


def intent_metrics(y_true, y_pred) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "per_class_f1": {l: f for l, f in zip(
            labels, f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0))},
    }


def escalation_metrics(y_true, y_pred) -> dict:
    """Escalate class P/R/F1 + auto-handle safety: of messages we auto-handled,
    the fraction where auto-handling was actually correct (the trust metric)."""
    y_true = np.asarray(y_true, bool)
    y_pred = np.asarray(y_pred, bool)
    tp = int(((y_pred) & (y_true)).sum())
    fp = int((y_pred & ~y_true).sum())
    fn = int((~y_pred & y_true).sum())
    auto_total = int((~y_pred).sum())
    auto_correct = int((~y_pred & ~y_true).sum())
    return {
        "escalate_precision": tp / (tp + fp) if tp + fp else 1.0,
        "escalate_recall": tp / (tp + fn) if tp + fn else 1.0,
        "escalate_f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
        "auto_handle_safety": auto_correct / auto_total if auto_total else 1.0,
        "auto_handle_rate": auto_total / len(y_true) if len(y_true) else 0.0,
        "n_auto_handled": auto_total,
    }


def bootstrap_ci(values: list, stat=np.mean, n=2000, seed=42) -> list:
    """95% CI for any scalar stat over per-example values."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return [float("nan"), float("nan")]
    stats = [stat(arr[rng.integers(0, len(arr), len(arr))]) for _ in range(n)]
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def kappa(a, b) -> float:
    return float(cohen_kappa_score(a, b))


def within_one(a, b) -> float:
    return float(np.mean([abs(x - y) <= 1 for x, y in zip(a, b)]))


def confusion(y_true, y_pred) -> dict[str, dict[str, int]]:
    labels = sorted(set(y_true) | set(y_pred))
    return {t: {p: sum(1 for tt, pp in zip(y_true, y_pred) if tt == t and pp == p)
                for p in labels} for t in labels}
