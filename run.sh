#!/usr/bin/env bash
# Run script with bash.

set -e # Stop script if any command fails.

if [ -d "/usr/local/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home" ]; then
  export JAVA_HOME="/usr/local/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
  export PATH="$JAVA_HOME/bin:$PATH"
fi

PYTHON="$(which python)" # Python executable from active environment.

PYTHONPATH=. "$PYTHON" scripts/run_etl.py # Run ETL script with project root available for imports.
