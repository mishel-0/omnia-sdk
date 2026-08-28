#!/usr/bin/env bash
# omnia-sdk: one-command reproducible benchmark.
#   ./benchmarks/run_all.sh            -> runs, writes benchmarks/benchmark_results.json
#   ./benchmarks/run_all.sh --svs s.svs -> use your own slide
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== omnia-sdk run_all =="
python -m pip install -q -r requirements.txt
python -m pip install -q -e .
python -m omnia_sdk.benchmark "$@"
echo
echo "Result: $(python -c "import json;d=json.load(open('benchmarks/benchmark_results.json'));print(f\"{d['measurement']['data_speedup']}x on {d['machine']['gpu'] or 'CPU'}\")")"
