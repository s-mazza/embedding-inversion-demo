#!/bin/bash
# Canonical test runner for v2 training correctness.
#
# These are the tests that matter for the active training path (v2 from-scratch,
# jina-v3). Run after any change to model.py or train.py before committing.
#
# Usage:
#   bash run_tests.sh           # run canonical suite
#   bash run_tests.sh -v        # verbose (show individual test names)
#   bash run_tests.sh --quick   # skip slow identity-at-init tests

set -e
cd "$(dirname "$0")"

TESTS="test_training_correctness.py test_v2_audit2.py"
ARGS="$@"

# Strip --quick flag and pass rest to pytest
if echo "$ARGS" | grep -q -- "--quick"; then
    ARGS=$(echo "$ARGS" | sed 's/--quick//')
    TESTS="test_v2_audit2.py"
    echo "Quick mode: running audit tests only (test_v2_audit2.py)"
fi

echo "=== Running v2 training correctness tests ==="
python3 -m pytest $TESTS $ARGS
echo "=== Done ==="
