#!/usr/bin/env bash
# Run script with bash.

set -e # Stop script if any command fails.

PYTHON="$(which python)" # Python executable from active environment.

PYTHONPATH=. "$PYTHON" scripts/run_etl.py # Run ETL script with project root available for imports.