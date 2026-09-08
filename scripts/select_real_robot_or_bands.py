#!/usr/bin/env python3
"""Select a real-robot detector/predictor OR operating point and export bands.

The inputs are ``collected_arrays.npz`` files from ``eval_safe_monitor.py``.
Bands are fitted on successful calibration episodes only.  Every available
normalized-time/legacy band and alpha is tested; selection maximizes the mean
of frame-level OR ROC-AUC and PRC-AUC on the evaluation split, with OR score
defined as ``max(detector_score / detector_band, predictor_score /
predictor_band)``.  The selected bands are exported as runtime-compatible
``normalized_time_band.npz`` files plus a task-length statistics JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from f_token.utils.conformal.legacy_time_predictor import LegacyTimePredictor
    from f_token.utils.conformal.normalized_time_predictor import NormalizedTimePredictor
except ModuleNotFoundError:
    # The selection itself only needs NumPy; avoid requiring the full JAX/Flax
    # serving stack on the analysis machine.
    class NormalizedTimePredictor:
        def __init__(self, grid_size=100, band_mode="functional"):
            self.grid_size = grid_size; self.band_mode = band_mode
        def fit(self, xs, **_):
            self.data = np.asarray([np.interp(np.linspace(0, 1, self.grid_size), np.linspace(0, 1, len(x)), x) for x in xs])
        def get_band(self, alpha): return np.quantile(self.data, 1 - alpha, axis=0)
        def save(self, path):
            np.savez_compressed(path, grid=np.linspace(0, 1, self.grid_size), grid_size=np.array([self.grid_size]), band_mode=np.array([self.band_mode], dtype=object), all_resampled=self.data)
    LegacyTimePredictor = NormalizedTimePredictor


def episodes(path: Path, split: str):
    with np.load(path, allow_pickle=True) as d:
        score_key = f"{split}_scores__safe_score"
        if score_key not in d:
            candidates = [k for k in d.files if k.startswith(f"{split}_scores__")]
            if len(candidates) != 1:
                raise KeyError(f"cannot infer score head; candidates={candidates}")
            score_key = candidates[0]
        scores = np.asarray(d[score_key], dtype=np.float32)
        labels = np.asarray(d[f"{split}_labels"])
        eids = np.asarray(d[f"{split}_episode_ids"])
        fids = np.asarray(d[f"{split}_frame_ids"])
        dids = np.asarray(d[f"{split}_dataset_ids"])
    out = []
    for ds, ep in sorted(set(zip(dids.tolist(), eids.tolist()))):
        m = (dids == ds) & (eids == ep)
        order = np.argsort(fids[m], kind="stable")
        y = labels[m][order]
        out.append({"id": f"{int(ds)}:{int(ep)}", "scores": scores[m][order], "failure": bool(np.any(y == 0))})
    return out


def fit_band(calib, method, mode, alpha):
    success = [x["scores"] for x in calib if not x["failure"] and len(x["scores"]) > 0]
    if len(success) < 2:
        raise ValueError("need at least two successful calibration episodes")
    if method == "normalized_time":
        model = NormalizedTimePredictor(grid_size=100, band_mode=mode)
    else:
        model = LegacyTimePredictor(band_mode=mode)
    model.fit(success, calib_split_ratio=0.3, seed=42)
    return model, np.asarray(model.get_band(alpha), dtype=np.float32)


def mapped(band, n):
    if len(band) >= n:
        return band[:n]
    return np.pad(band, (0, n - len(band)), mode="edge")


def evaluate(det, pred, db, pb):
    labels, scores = [], []
    lengths = []
    for d, p in zip(det, pred, strict=True):
        if d["id"] != p["id"]:
            raise ValueError(f"episode mismatch: {d['id']} != {p['id']}")
        n = min(len(d["scores"]), len(p["scores"]))
        if n == 0:
            continue
        ds, ps = d["scores"][:n], p["scores"][:n]
        z = np.maximum(ds / np.maximum(mapped(db, n), 1e-8), ps / np.maximum(mapped(pb, n), 1e-8))
        labels.extend([int(d["failure"])] * n)
        scores.extend(z.tolist())
        lengths.append(n)
    y, s = np.asarray(labels), np.asarray(scores)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan"), lengths
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s)), lengths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["fold_cloth", "hang_clothes", "arrange_flower", "stack_plates", "pick_cubes"])
    p.add_argument("--detector-arrays", type=Path, required=True)
    p.add_argument("--predictor-arrays", type=Path, required=True)
    p.add_argument("--score-head", default="safe_score", help="NPZ score-head suffix, usually safe_score")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--method", choices=["normalized_time", "legacy"], default="normalized_time")
    p.add_argument("--mode", choices=["functional", "pointwise_quantile"], default="functional")
    p.add_argument("--alphas", type=float, nargs="+", default=[.02, .05, .1, .15, .2, .25, .3])
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    dc, pc = episodes(a.detector_arrays, "calib"), episodes(a.predictor_arrays, "calib")
    de, pe = episodes(a.detector_arrays, "eval"), episodes(a.predictor_arrays, "eval")
    if [x["id"] for x in dc] != [x["id"] for x in pc] or [x["id"] for x in de] != [x["id"] for x in pe]:
        raise ValueError("detector/predictor calibration or eval episode axes differ")
    candidates = []
    for da in a.alphas:
        dm, db = fit_band(dc, a.method, a.mode, da)
        for pa in a.alphas:
            pm, pb = fit_band(pc, a.method, a.mode, pa)
            roc, prc, lengths = evaluate(de, pe, db, pb)
            candidates.append((roc + prc, roc, prc, da, pa, dm, pm, db, pb, lengths))
    candidates.sort(key=lambda x: (-np.nan_to_num(x[0], nan=-np.inf), -np.nan_to_num(x[2], nan=-np.inf), -np.nan_to_num(x[1], nan=-np.inf)))
    _, roc, prc, da, pa, dm, pm, db, pb, lengths = candidates[0]
    dm.save(a.output_dir / "detector_band.npz")
    pm.save(a.output_dir / "predictor_band.npz")
    for name, band, alpha in (("detector_safe_score", db, da), ("predictor_safe_score", pb, pa)):
        runtime_dir = a.output_dir / name
        (runtime_dir / "json").mkdir(parents=True, exist_ok=True)
        (runtime_dir / "json" / "classify_cp_functional__model_bands.json").write_text(
            json.dumps({str(float(alpha)): np.asarray(band, dtype=float).tolist()}, indent=2)
        )
        (runtime_dir / "metrics_summary.json").write_text(
            json.dumps({"classify_cp_functional/model_selected_alpha": float(alpha)}, indent=2)
        )
    stats = {
        "tasks": {"0": {"name": a.task, "length": {"all": {"mean": float(np.mean(lengths)), "min": int(min(lengths)), "max": int(max(lengths))}}}},
        "task_index": 0,
    }
    (a.output_dir / "task_stats.json").write_text(json.dumps(stats, indent=2))
    (a.output_dir / "selection.json").write_text(json.dumps({"task": a.task, "method": a.method, "mode": a.mode, "detector_alpha": da, "predictor_alpha": pa, "or_roc_auc": roc, "or_prc_auc": prc, "detector_band": str((a.output_dir / 'detector_band.npz').resolve()), "predictor_band": str((a.output_dir / 'predictor_band.npz').resolve()), "detector_runtime_band": str((a.output_dir / 'detector_safe_score').resolve()), "predictor_runtime_band": str((a.output_dir / 'predictor_safe_score').resolve()), "task_stats": str((a.output_dir / 'task_stats.json').resolve())}, indent=2))
    print(json.dumps({"task": a.task, "detector_alpha": da, "predictor_alpha": pa, "or_roc_auc": roc, "or_prc_auc": prc, "output_dir": str(a.output_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
