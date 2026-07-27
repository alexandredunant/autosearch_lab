"""Precision-Recall-Gain (AUPRG) — Flach & Kull, NeurIPS 2015.

Trimmed vendor copy of the reference implementation (MIT License,
Copyright (c) 2016 Meelis Kull, Peter Flach):
https://github.com/meeliskull/prg -- Python_package/prg/prg.py

Only the scalar-area computation is kept (no plotting). AUPRG fixes a known
pathology of ordinary PR-AUC / average precision: because F-scores combine
precision and recall via a harmonic mean while PR-AUC integrates via an
arithmetic mean, PR-AUC can rank one classifier above another even when the
other has strictly better expected F1 at every threshold. AUPRG is computed
in "gain" space (relative to the always-positive baseline) so its area does
correspond to expected F1. Used here as a secondary, informational metric
alongside val_pr_auc (see prepare.py's scalar_metrics()) -- not the search's
decision metric.
"""

from __future__ import annotations

import warnings

import numpy as np


def _precision(tp, fn, fp, tn):
    with np.errstate(divide="ignore", invalid="ignore"):
        return tp / (tp + fp)


def _recall(tp, fn, fp, tn):
    with np.errstate(divide="ignore", invalid="ignore"):
        return tp / (tp + fn)


def _precision_gain(tp, fn, fp, tn):
    n_pos = tp + fn
    n_neg = fp + tn
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = 1.0 - (n_pos / n_neg) * (fp / tp)
    if np.size(gain) > 1:
        gain[tn + fn == 0] = 0
    elif tn + fn == 0:
        gain = 0
    return gain


def _recall_gain(tp, fn, fp, tn):
    n_pos = tp + fn
    n_neg = fp + tn
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = 1.0 - (n_pos / n_neg) * (fn / tp)
    if np.size(gain) > 1:
        gain[tn + fn == 0] = 1
    elif tn + fn == 0:
        gain = 1
    return gain


def _create_segments(labels, pos_scores, neg_scores):
    n = np.size(labels)
    new_order = np.lexsort((neg_scores, -pos_scores))
    labels = labels[new_order]
    pos_scores = pos_scores[new_order]
    neg_scores = neg_scores[new_order]
    segments = {
        "pos_score": np.zeros(n), "neg_score": np.zeros(n),
        "pos_count": np.zeros(n), "neg_count": np.zeros(n),
    }
    j = -1
    for i, label in enumerate(labels):
        if i == 0 or pos_scores[i - 1] != pos_scores[i] or neg_scores[i - 1] != neg_scores[i]:
            j += 1
            segments["pos_score"][j] = pos_scores[i]
            segments["neg_score"][j] = neg_scores[i]
        if label == 0:
            segments["neg_count"][j] += 1
        else:
            segments["pos_count"][j] += 1
    for key in segments:
        segments[key] = segments[key][: j + 1]
    return segments


def _get_point(points, index):
    keys = list(points.keys())
    point = np.zeros(len(keys))
    key_indices = {}
    for i, key in enumerate(keys):
        point[i] = points[key][index]
        key_indices[key] = i
    return point, key_indices


def _insert_point(new_point, key_indices, points, precision_gain=0, recall_gain=0, is_crossing=0):
    for key in key_indices:
        points[key] = np.insert(points[key], 0, new_point[key_indices[key]])
    points["precision_gain"][0] = precision_gain
    points["recall_gain"][0] = recall_gain
    points["is_crossing"][0] = is_crossing
    new_order = np.lexsort((-points["precision_gain"], points["recall_gain"]))
    for key in points:
        points[key] = points[key][new_order]
    return points


def _create_crossing_points(points, n_pos, n_neg):
    n = n_pos + n_neg
    points["is_crossing"] = np.zeros(np.size(points["pos_score"]))
    j = np.amin(np.where(points["recall_gain"] >= 0)[0])
    if points["recall_gain"][j] > 0:
        point_1, key_indices_1 = _get_point(points, j)
        point_2, _ = _get_point(points, j - 1)
        delta = point_1 - point_2
        if delta[key_indices_1["TP"]] > 0:
            alpha = (n_pos * n_pos / n - points["TP"][j - 1]) / delta[key_indices_1["TP"]]
        else:
            alpha = 0.5
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new_point = point_2 + alpha * delta
        new_prec_gain = _precision_gain(
            new_point[key_indices_1["TP"]], new_point[key_indices_1["FN"]],
            new_point[key_indices_1["FP"]], new_point[key_indices_1["TN"]],
        )
        points = _insert_point(new_point, key_indices_1, points, precision_gain=new_prec_gain, is_crossing=1)

    x = points["recall_gain"]
    y = points["precision_gain"]
    temp_y_0 = np.append(y, 0)
    temp_0_y = np.append(0, y)
    temp_1_x = np.append(1, x)
    with np.errstate(invalid="ignore"):
        indices = np.where(np.logical_and(temp_y_0 * temp_0_y < 0, temp_1_x >= 0))[0]
    for i in indices:
        cross_x = x[i - 1] + (-y[i - 1]) / (y[i] - y[i - 1]) * (x[i] - x[i - 1])
        point_1, key_indices_1 = _get_point(points, i)
        point_2, _ = _get_point(points, i - 1)
        delta = point_1 - point_2
        if delta[key_indices_1["TP"]] > 0:
            alpha = (n_pos * n_pos / (n - n_neg * cross_x) - points["TP"][i - 1]) / delta[key_indices_1["TP"]]
        else:
            alpha = (n_neg / n_pos * points["TP"][i - 1] - points["FP"][i - 1]) / delta[key_indices_1["FP"]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new_point = point_2 + alpha * delta
        new_rec_gain = _recall_gain(
            new_point[key_indices_1["TP"]], new_point[key_indices_1["FN"]],
            new_point[key_indices_1["FP"]], new_point[key_indices_1["TN"]],
        )
        points = _insert_point(new_point, key_indices_1, points, recall_gain=new_rec_gain, is_crossing=1)
        i += 1
        indices += 1
        x = points["recall_gain"]
        y = points["precision_gain"]
        temp_y_0 = np.append(y, 0)
        temp_0_y = np.append(0, y)
        temp_1_x = np.append(1, x)
    return points


def _create_prg_curve(labels, pos_scores):
    neg_scores = -pos_scores
    n = np.size(labels)
    n_pos = np.sum(labels)
    n_neg = n - n_pos
    labels = 1 * (labels == 1)
    segments = _create_segments(labels, pos_scores, neg_scores)
    points = {
        "pos_score": np.insert(segments["pos_score"], 0, np.inf),
        "neg_score": np.insert(segments["neg_score"], 0, -np.inf),
        "TP": np.insert(np.cumsum(segments["pos_count"]), 0, 0),
        "FP": np.insert(np.cumsum(segments["neg_count"]), 0, 0),
    }
    points["FN"] = n_pos - points["TP"]
    points["TN"] = n_neg - points["FP"]
    points["precision_gain"] = _precision_gain(points["TP"], points["FN"], points["FP"], points["TN"])
    points["recall_gain"] = _recall_gain(points["TP"], points["FN"], points["FP"], points["TN"])
    points = _create_crossing_points(points, n_pos, n_neg)
    return points


def _calc_auprg(prg_curve) -> float:
    area = 0.0
    recall_gain = prg_curve["recall_gain"]
    precision_gain = prg_curve["precision_gain"]
    for i in range(1, len(recall_gain)):
        if not np.isnan(recall_gain[i - 1]) and recall_gain[i - 1] >= 0:
            width = recall_gain[i] - recall_gain[i - 1]
            height = (precision_gain[i] + precision_gain[i - 1]) / 2
            area += width * height
    return float(area)


def auprg_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Area under the Precision-Recall-Gain curve. NaN if only one class is present."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) != 2:
        return float("nan")
    curve = _create_prg_curve(y_true.astype(float), np.asarray(y_prob, dtype=float))
    return _calc_auprg(curve)
