#!/usr/bin/env bash
# Run the ECML 2026 evaluator locally on the competition level-0 scenario.
# Usage (from the project root):  bash submission/run_local_eval.sh [checkpoint.pth]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-eval_out}
[ $# -ge 1 ] && export CHECKPOINT="$1"
rm -rf "$OUT"
mkdir -p "$OUT/run" "$OUT/analysis"
PYTHONPATH=. flatland-trajectory-generate-from-policy \
  --data-dir "$OUT/run" \
  --policy submission.my_policy.MyPolicy \
  --obs-builder submission.my_observation_builder.MyObservationBuilder \
  --rewards flatland.envs.rewards.ECML2026Rewards \
  --env-path competition/level_0_scenario_1.pkl
flatland-trajectory-analysis --root-data-dir "$OUT/run" --output-dir "$OUT/analysis"
# The evaluator saves one env snapshot (.pkl) per step for replays; drop them
# unless KEEP_STATES=1.
[ "${KEEP_STATES:-0}" = "1" ] || rm -rf "$OUT/run/serialised_state"
echo; cat "$OUT/analysis/all_trains_arrived.csv"
