#!/usr/bin/env bash
# Benchmark the submission locally (no Docker) on the 27-scenario example curriculum
# plus the competition level-0 scenario, using the official ECML 2026 evaluator.
#
# Usage (from the project root, with flatland-rl==4.2.6 + torch installed):
#   bash submission/run_local_benchmark.sh [checkpoint.pth]
#
# Env vars: OUT (default benchmark_out), KEEP_STATES=1 to keep per-step env snapshots.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-benchmark_out}
[ $# -ge 1 ] && export CHECKPOINT="$1"

CUR_DIR=reinforcement_learning/curriculum/curriculum
if [ ! -d "$CUR_DIR/example_curriculum" ]; then
  unzip -q -o reinforcement_learning/curriculum/example_curriculum.zip -d "$CUR_DIR"
fi

[ "${ANALYSIS_ONLY:-0}" = "1" ] || rm -rf "$OUT"
mkdir -p "$OUT/run" "$OUT/analysis"

if [ -n "${SCENARIO_DIR:-}" ]; then
  ENV_PATHS=("$SCENARIO_DIR"/*.pkl)
else
  ENV_PATHS=("$CUR_DIR"/example_curriculum/*.pkl reinforcement_learning/sampling/level_0_scenario_1.pkl)
fi

for env_path in "${ENV_PATHS[@]}"; do
  env=$(basename "$env_path" .pkl)
  echo "=== $env"
  if [ "${ANALYSIS_ONLY:-0}" != "1" ]; then
    rm -rf "$OUT/run/$env" "$OUT/crashed_$env"
    mkdir -p "$OUT/run/$env"
    set +e
    PYTHONPATH=. flatland-trajectory-generate-from-policy \
      --data-dir "$OUT/run/$env" \
      --policy submission.my_policy.MyPolicy \
      --obs-builder submission.my_observation_builder.MyObservationBuilder \
      --rewards flatland.envs.rewards.ECML2026Rewards \
      --env-path "$env_path" \
      --ep-id "$env" > "$OUT/run/$env.log" 2>&1
    rc=$?
    set -e
    [ $rc -eq 0 ] || { echo "    CRASHED (exit $rc), see $OUT/run/$env.log"; touch "$OUT/crashed_$env"; }
  fi
  # the analysis CLI handles one trajectory per root dir and needs serialised_state
  if [ ! -e "$OUT/crashed_$env" ]; then
    rm -rf "$OUT/analysis/$env"
    flatland-trajectory-analysis --root-data-dir "$OUT/run/$env" --output-dir "$OUT/analysis/$env" \
      > "$OUT/analysis/$env.log" 2>&1 || echo "    analysis failed, see $OUT/analysis/$env.log"
  fi
  [ "${KEEP_STATES:-0}" = "1" ] || rm -rf "$OUT/run/$env/serialised_state"
done

python - "$OUT" <<'EOF'
import csv, glob, os, sys
out = sys.argv[1]
names = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{out}/run/*.log"))
print(f"\n{'scenario':<28}{'steps':>7}{'success_rate':>14}{'norm_reward':>13}")
srs, nrs = [], []
for n in names:
    f = f"{out}/analysis/{n}/all_trains_arrived.csv"
    if os.path.exists(f"{out}/crashed_{n}") or not os.path.exists(f):
        print(f"{n:<28}{'-':>7}{'CRASH/NA (counted 0)':>27}")
        srs.append(0.0); nrs.append(0.0)
        continue
    r = next(csv.DictReader(open(f)))
    sr, nr = float(r["success_rate"]), float(r["normalized_reward"])
    srs.append(sr); nrs.append(nr)
    print(f"{n:<28}{r['env_time']:>7}{sr:>14.3f}{nr:>13.3f}")
k = len(srs)
print(f"\n{k} scenarios | mean success_rate {sum(srs)/k:.3f} | mean normalized_reward {sum(nrs)/k:.3f}"
      f" | below 25% arrived: {sum(s < 0.25 for s in srs)}")
EOF
