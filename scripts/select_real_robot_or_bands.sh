#!/usr/bin/env bash
set -euo pipefail

# Select detector/predictor conformal bands for one of the real-robot tasks.
# Predictor arrays must first be produced by eval_safe_monitor.py on the same
# calibration/eval episode indices as the detector arrays.

TASK="${1:-}"
[[ -n "$TASK" ]] || { echo "usage: $0 {fold_cloth|hang_clothes|arrange_flower|stack_plates|pick_cubes}" >&2; exit 2; }

OPENPI_ROOT="${OPENPI_ROOT:-/data/users/liujingyuan/workspace/openpi_local}"
ROOT="${ROOT:-/data/users/liujingyuan/data/monitor_experiments}"
DET_ROOT="$ROOT/real_robot_detector/checkpoints"
PRED_ROOT="$ROOT/real_robot_predictor/checkpoints"
OUT_ROOT="${OUT_ROOT:-$ROOT/real_robot_or_bands/$TASK}"
SEED="${SEED:-42}"

case "$TASK" in
  fold_cloth)
    DET_CONFIG=pi05_real_robot_temporal_bce_detector
    DET_RUN="56_real_robot_temporal_bce_detector_fold_cloth_seed${SEED}"
    PRED_CONFIG=pi05_concat_fold_cloth_obs1_interval1
    PRED_RUN="60_real_robot_predictor_fold_cloth_seed${SEED}"
    ;;
  hang_clothes)
    DET_CONFIG=pi05_real_robot_temporal_bce_detector
    DET_RUN="56_real_robot_temporal_bce_detector_hang_clothes_seed${SEED}"
    PRED_CONFIG=pi05_concat_hang_clothes_obs1_interval1
    PRED_RUN="60_real_robot_predictor_hang_clothes_seed${SEED}"
    ;;
  arrange_flower|stack_plates|pick_cubes)
    DET_CONFIG="pi05_real_robot_${TASK}_evorl_detector"
    DET_RUN="56_real_robot_temporal_bce_detector_${TASK}_seed${SEED}"
    PRED_CONFIG="pi05_real_robot_${TASK}_predictor"
    PRED_RUN="60_real_robot_predictor_${TASK}_seed${SEED}"
    ;;
  *) echo "unsupported task: $TASK" >&2; exit 2 ;;
esac

DET_EVAL_ROOT="$DET_ROOT/$DET_CONFIG/$DET_RUN/obs1"
PRED_EVAL_ROOT="$PRED_ROOT/$PRED_CONFIG/$PRED_RUN/obs1_interval1"
DET_ARRAY="${DETECTOR_ARRAYS:-$(find "$DET_EVAL_ROOT" -path '*/evals/*/safe_eval/collected_arrays.npz' | sort | tail -1)}"
PRED_ARRAY="${PREDICTOR_ARRAYS:-$(find "$PRED_EVAL_ROOT" -path '*/evals/*/safe_eval/collected_arrays.npz' | sort | tail -1)}"

if [[ -z "$DET_ARRAY" || ! -f "$DET_ARRAY" ]]; then
  echo "Detector arrays not found under $DET_EVAL_ROOT" >&2
  echo "Run the detector evaluation first (56_real_robot_detector.sh with RUN_EVAL=1)." >&2
  exit 3
fi
if [[ -z "$PRED_ARRAY" || ! -f "$PRED_ARRAY" ]]; then
  echo "Predictor arrays not found under $PRED_EVAL_ROOT" >&2
  echo "The predictor training launcher does not create safe_eval arrays automatically." >&2
  echo "Evaluate its selected monitor checkpoint with:" >&2
  echo "  cd $OPENPI_ROOT" >&2
  echo "  python scripts/eval_safe_monitor.py --calib-repo-id <repo> --calib-index-path <calib.json> --eval-repo-id <repo> --eval-index-path <eval.json> --band-method none --no-render-video $PRED_CONFIG --exp-name $PRED_RUN/obs1_interval1/evals/<step> --checkpoint-base-dir $PRED_ROOT --assets-base-dir /data/users/liujingyuan/data/assets --monitor-weight-path $PRED_ROOT/$PRED_CONFIG/$PRED_RUN/obs1_interval1/monitor/<step>.msgpack" >&2
  exit 4
fi

python scripts/select_real_robot_or_bands.py \
  --task "$TASK" \
  --detector-arrays "$DET_ARRAY" \
  --predictor-arrays "$PRED_ARRAY" \
  --output-dir "$OUT_ROOT" \
  --method normalized_time \
  --mode functional \
  --alphas 0.02 0.05 0.10 0.15 0.20 0.25 0.30

echo
echo "Runtime paths:"
echo "  detector band: $OUT_ROOT/detector_band.npz"
echo "  predictor band: $OUT_ROOT/predictor_band.npz"
echo "  detector local runtime band: $OUT_ROOT/detector_safe_score/"
echo "  predictor runtime band: $OUT_ROOT/predictor_safe_score/"
echo "  task stats: $OUT_ROOT/task_stats.json"
echo "  selection: $OUT_ROOT/selection.json"
