"""Pure routing core for native-preserving Expert inference.

This module deliberately knows nothing about GT, LCO, NMS, or mmdetection.
The caller supplies the native evidence and the already-computed legacy
action.  Keeping the decision core small makes the P0 regression auditable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NativeEvidence:
    labels: np.ndarray
    scores: np.ndarray
    protected: np.ndarray


def native_evidence(class_scores: np.ndarray, category_ids: np.ndarray,
                    novel_ids: set[int], threshold: float) -> NativeEvidence:
    """Return detector top-1 evidence and the fixed native-novel mask."""
    scores = np.asarray(class_scores)
    categories = np.asarray(category_ids)
    if scores.ndim != 2 or categories.ndim != 1:
        raise ValueError('class_scores must be [M,C] and category_ids [C]')
    if scores.shape[1] != len(categories):
        raise ValueError('category order does not match class_scores')
    top = np.argmax(scores, axis=1)
    labels = categories[top].astype(np.int64, copy=False)
    values = scores[np.arange(len(scores)), top].astype(np.float32, copy=False)
    is_novel = np.asarray([int(x) in novel_ids for x in labels], dtype=bool)
    protected = is_novel & (values >= float(threshold))
    return NativeEvidence(labels=labels, scores=values, protected=protected)


def query_indices(protected: np.ndarray, budget: int,
                  redistribute: bool) -> np.ndarray:
    """Choose stable M100 positions to query under P1 or P2.

    P1 queries the original top-Q positions.  P2 skips protected positions
    and continues down the same M100 priority order until Q calls are used.
    """
    mask = np.asarray(protected, dtype=bool)
    if mask.ndim != 1 or budget < 0:
        raise ValueError('protected must be 1-D and budget non-negative')
    if redistribute:
        return np.flatnonzero(~mask)[:budget].astype(np.int64)
    return np.arange(min(int(budget), len(mask)), dtype=np.int64)


def retain_native(legacy_labels: np.ndarray, legacy_scores: np.ndarray,
                  evidence: NativeEvidence) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replace protected legacy actions with their native label and score."""
    labels = np.asarray(legacy_labels, dtype=np.int64).copy()
    scores = np.asarray(legacy_scores, dtype=np.float32).copy()
    if labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError('legacy labels/scores must be matching 1-D arrays')
    if any(len(x) != len(labels) for x in
           (evidence.labels, evidence.scores, evidence.protected)):
        raise ValueError('native and legacy actions must have the same length')
    labels[evidence.protected] = evidence.labels[evidence.protected]
    scores[evidence.protected] = evidence.scores[evidence.protected]
    source = np.full(len(labels), 'legacy', dtype='<U16')
    source[evidence.protected] = 'native_protected'
    return labels, scores, source
