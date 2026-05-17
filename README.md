# Hotel No Show Prediction

## Setup

This project uses a pip virtual environment.

PySpark also requires Java. Install Java 17 before running the pipeline.

On macOS with Homebrew:

```bash
brew install openjdk@17
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
```

Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the ETL pipeline:

```bash
bash run.sh
```
