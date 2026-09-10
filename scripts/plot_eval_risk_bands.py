#!/usr/bin/env python3
"""Plot detector/predictor scores and fitted functional bands on eval episodes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TASKS = {
    "arrange_flower": ("pi05_real_robot_arrange_flower_evorl_detector", "56_real_robot_temporal_bce_detector_arrange_flower_seed42", "pi05_real_robot_arrange_flower_predictor", "60_real_robot_predictor_arrange_flower_seed42", "00009999"),
    "stack_plates": ("pi05_real_robot_stack_plates_evorl_detector", "56_real_robot_temporal_bce_detector_stack_plates_seed42", "pi05_real_robot_stack_plates_predictor", "60_real_robot_predictor_stack_plates_seed42", "00009999"),
    "pick_cubes": ("pi05_real_robot_pick_cubes_evorl_detector", "56_real_robot_temporal_bce_detector_pick_cubes_seed42", "pi05_real_robot_pick_cubes_predictor", "60_real_robot_predictor_pick_cubes_seed42", "00009999"),
    "fold_cloth": ("pi05_real_robot_temporal_bce_detector", "56_real_robot_temporal_bce_detector_fold_cloth_seed42", "pi05_concat_fold_cloth_obs1_interval1", "60_real_robot_predictor_fold_cloth_seed42", "00002000"),
}


def _array_path(root: Path, kind: str, config: str, run: str, step: str) -> Path:
    suffix = "obs1" if kind == "detector" else "obs1_interval1"
    return root / f"real_robot_{kind}" / "checkpoints" / config / run / suffix / "evals" / step / "safe_eval" / "collected_arrays.npz"


def _score_key(z: np.lib.npyio.NpzFile, split: str, preferred: str) -> str:
    keys = [k for k in z.files if k.startswith(split + "_scores__")]
    for k in keys:
        if preferred in k:
            return k
    if len(keys) != 1:
        raise KeyError(f"Cannot choose {split} score from {keys}")
    return keys[0]


def _episodes(path: Path, preferred: str):
    z = np.load(path, allow_pickle=True)
    split = "eval"
    scores = np.asarray(z[_score_key(z, split, preferred)], dtype=float)
    labels = np.asarray(z[split + "_labels"])
    eids = np.asarray(z[split + "_episode_ids"])
    dids = np.asarray(z[split + "_dataset_ids"])
    out = []
    for d, e in sorted(set(zip(dids.tolist(), eids.tolist())), key=lambda x: (str(x[0]), int(x[1]))):
        m = (dids == d) & (eids == e)
        order = np.argsort(np.asarray(z[split + "_frame_ids"])[m])
        out.append({"dataset_id": str(d), "episode_id": int(e), "scores": scores[m][order], "failure": bool(np.any(labels[m] == 0))})
    cal = set(zip(z["calib_dataset_ids"].tolist(), z["calib_episode_ids"].tolist()))
    ev = set(zip(z["eval_dataset_ids"].tolist(), z["eval_episode_ids"].tolist()))
    return out, {"calibration_episodes": len(cal), "eval_episodes": len(ev), "calibration_eval_overlap": len(cal & ev)}


def _band(path: Path, alpha: float) -> np.ndarray:
    p = path / "json" / "classify_cp_functional__model_bands.json"
    data = json.loads(p.read_text())
    key = min(data, key=lambda x: abs(float(x) - alpha))
    return np.asarray(data[key], dtype=float)


def _resize_band(band: np.ndarray, n: int) -> np.ndarray:
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(band)), band)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, default=Path("/data/users/liujingyuan/data/monitor_experiments"))
    ap.add_argument("--bands-root", type=Path, default=Path("/data/users/liujingyuan/data/monitor_experiments/real_robot_or_bands"))
    ap.add_argument("--output-dir", type=Path, default=Path("/data/users/liujingyuan/data/monitor_experiments/risk_band_plots"))
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--max-episodes", type=int, default=3)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for task in args.tasks:
        dc, dr, pc, pr, step = TASKS[task]
        dp = _array_path(args.results_root, "detector", dc, dr, step)
        pp = _array_path(args.results_root, "predictor", pc, pr, step)
        bdir = args.bands_root / task
        sel = json.loads((bdir / "selection.json").read_text())
        det, det_split = _episodes(dp, "safe_score")
        pred, pred_split = _episodes(pp, "predictor_score")
        det = det[: args.max_episodes]
        pred = pred[: args.max_episodes]
        db = _band(bdir / "detector_safe_score", float(sel["detector_alpha"]))
        pb = _band(bdir / "predictor_safe_score", float(sel["predictor_alpha"]))
        rows = []
        fig, axes = plt.subplots(len(det), 2, squeeze=False, figsize=(13, 3.2 * len(det)))
        for i, (de, pe) in enumerate(zip(det, pred)):
            for j, (ep, band, title, color) in enumerate(((de, db, "Detector", "tab:blue"), (pe, pb, "Predictor", "tab:green"))):
                ax = axes[i, j]; y = ep["scores"]; x = np.arange(len(y)); u = _resize_band(band, len(y))
                ax.plot(x, y, color=color, label="risk"); ax.plot(x, u, "--", color="tab:red", label="band")
                ax.set_title(f"{title} | episode {ep['episode_id']} | failure={ep['failure']}"); ax.set_xlabel("timestep"); ax.set_ylabel("risk"); ax.legend(); ax.grid(alpha=.25)
                rows.append({"episode_id": ep["episode_id"], "model": title.lower(), "n": len(y), "risk_min": float(y.min()), "risk_max": float(y.max()), "band_min": float(u.min()), "band_max": float(u.max()), "out_of_band_fraction": float(np.mean(y > u))})
        fig.suptitle(task); fig.tight_layout(); fig.savefig(args.output_dir / f"{task}_eval_risk_bands.png", dpi=150); plt.close(fig)
        summary[task] = {"detector_arrays": str(dp), "predictor_arrays": str(pp), "detector_alpha": sel["detector_alpha"], "predictor_alpha": sel["predictor_alpha"], "detector_split": det_split, "predictor_split": pred_split, "episodes": rows}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v["episodes"] for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
