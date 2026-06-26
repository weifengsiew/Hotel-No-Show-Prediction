# Hotel No Show Prediction

<p align="center">
  <img src="notebooks/images/final_results_header.png" alt="Hotel no-show prediction report visual" width="900"/>
</p>

<p align="center">
  <strong>Hotel No Show Prediction</strong>
</p>

## 1. Project Objective

In this project, we leverage machine learning to accurately predict the probability that a hotel guest will show up for their booked stay ('0'), or if a no-show will occur ('1').

These probabilistic predictions are valuable as they can help inform hotel operations decisions, such as staffing, inventory management, reminder campaigns, deposit policies, and overbooking limits, so as to optimize business objectives such as profit.

## 2. Workflow Stages

To achieve the project objective, a machine learning workflow consisting of the following stages was developed, starting with the dataset noshow.db:

`noshow.db -> data ingestion -> data cleaning and validation -> feature engineering -> train/test split -> ML experiment -> pipeline selection and calibration -> holdout evaluation and results`

Note: a scikit-learn pipeline is composed of a preprocessor, which feeds engineered and preprocessed features to a machine learning model with specific hyperparameters to make a prediction.

<p align="center">
  <a href="http://127.0.0.1:4141/?types=nodes&expandAllPipelines=false&pid=__default__">Visualise the workflow stages in detail</a>
</p>

## 3. Python Frameworks

The workflow stages are supported by the following Python frameworks:

<p align="center">
  <img alt="Kedro" src="https://img.shields.io/badge/Kedro-Orchestration-243B53?style=for-the-badge">
  <img alt="pandas" src="https://img.shields.io/badge/pandas-Data%20Cleaning-150458?style=for-the-badge&logo=pandas&logoColor=white">
  <img alt="Great Expectations" src="https://img.shields.io/badge/Great%20Expectations-Validation-FF6319?style=for-the-badge">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-Feature%20Engineering-013243?style=for-the-badge&logo=numpy&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-Modeling-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white">
  <img alt="LightGBM" src="https://img.shields.io/badge/LightGBM-Modeling-02569B?style=for-the-badge">
  <img alt="XGBoost" src="https://img.shields.io/badge/XGBoost-Modeling-FF6600?style=for-the-badge">
  <img alt="MLflow" src="https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white">
  <img alt="joblib" src="https://img.shields.io/badge/joblib-Persistence-4B8BBE?style=for-the-badge">
  <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-Reporting-F37626?style=for-the-badge&logo=jupyter&logoColor=white">
  <img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-Visualisation-11557C?style=for-the-badge">
  <img alt="Seaborn" src="https://img.shields.io/badge/Seaborn-Visualisation-4B8BBE?style=for-the-badge">
</p>

|  | Python framework used |
| --- | --- |
| Pipeline scaffolding | Kedro |
| Pipeline visualisation | Kedro-Viz |
| Data ingestion | pandas |
| Data cleaning | pandas |
| Data validation | Great Expectations, pandas |
| Feature engineering | pandas, NumPy |
| Train/test split | scikit-learn |
| ML experiment | scikit-learn, LightGBM, XGBoost |
| Experiment logging | MLflow |
| Pipeline selection and calibration | scikit-learn |
| Holdout evaluation and results | scikit-learn, Matplotlib, Seaborn, Jupyter |
| Pipeline persistence | joblib |

## 4. Results Summary

### Final Pipeline

The best candidate pipeline was selected based on the highest mean cross-validated AUROC, then calibrated on the train set (80%) using isotonic regression.

| Components of final pipeline | Details |
| --- | --- |
| Preprocessor | `preprocessor_v1` |
| Model | `LGBMClassifier` |
| Hyperparameters | `learning_rate=0.05`, `min_child_samples=20`, `n_estimators=300`, `num_leaves=127`, `class_weight="balanced"` |
| Calibration | Isotonic regression |

### Evaluation Plots

The calibrated pipeline was evaluated on the test set (20%).

<p align="center">
  <img src="notebooks/images/roc_curve.png" alt="ROC curve" width="450"/>
  <img src="notebooks/images/precision_recall_curve.png" alt="Precision-recall curve" width="450"/>
</p>

<p align="center">
  <img src="notebooks/images/calibration_curve.png" alt="Calibration curve" width="450"/>
</p>

### Final Results

See [`notebooks/final_results.ipynb`](notebooks/final_results.ipynb) for the full results.

## 5. To Reproduce

### Environment Setup

Use Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS, install the OpenMP runtime required by LightGBM:

```bash
brew install libomp
```

Check the environment:

```bash
python --version
python -c "import kedro, pandas, pyarrow, sklearn, great_expectations, mlflow, lightgbm, xgboost; print('environment ok')"
```

### Run Workflow

Run the full workflow:

```bash
./run.sh
```

Run stages individually:

```bash
./scripts/stage1.sh
./scripts/stage2.sh
./scripts/stage3.sh
./scripts/stage4.sh
```

| Script | Runs |
| --- | --- |
| `scripts/stage1.sh` | `noshow.db` |
| `scripts/stage2.sh` | `data ingestion -> data cleaning and validation` |
| `scripts/stage3.sh` | `feature engineering` |
| `scripts/stage4.sh` | `train/test split -> ML experiment -> pipeline selection and calibration -> holdout evaluation and results` |
| `scripts/all_stages.sh` | Runs stages 1-4 |

## 6. Repository Structure

```text
.
|-- conf/
|   `-- base/
|       |-- catalog/              # Maps pipeline inputs/outputs to files
|       `-- parameters/           # Controls validation, model search, and calibration
|           |-- ml_experiment.yml
|           `-- model_hyperparams/ # Search grids for Random Forest, LightGBM, and XGBoost
|-- data/
|   |-- raw/                      # Downloaded noshow.db source data
|   |-- intermediate/             # Cleaned data before feature engineering
|   `-- processed/                # Feature-engineered table used for model training
|-- notebooks/
|   |-- final_results.ipynb       # Full results notebook
|   |-- final_results_helpers.py
|   `-- images/
|-- results/
|   |-- ml_experiments/           # best_model.joblib, test metrics, and final pipeline summary
|   |-- mlruns/                   # MLflow run params, metrics, tags, and artifacts
|   `-- validation/               # Raw and cleaned data validation results and failure tables
|-- scripts/
|   |-- all_stages.sh
|   |-- stage1.sh                 # noshow.db
|   |-- stage2.sh                 # data ingestion -> data cleaning and validation
|   |-- stage3.sh                 # feature engineering
|   `-- stage4.sh                 # train/test split -> ML experiment -> pipeline selection and calibration -> holdout evaluation and results
|-- src/
|   `-- hotel_no_show_prediction/
|       |-- pipelines/            # Kedro nodes and pipelines for each workflow stage
|       `-- sklearn_pipeline_components/ # Preprocessor/model registries and config loaders
|-- pyproject.toml
|-- requirements.txt
`-- run.sh                        # Full workflow entry point
```

## 7. Future Extensions

To add a new candidate pipeline:

1. Add a preprocessor builder in [`preprocessor_registry.py`](src/hotel_no_show_prediction/sklearn_pipeline_components/preprocessor_registry.py).
2. Register the estimator class in [`model_registry.py`](src/hotel_no_show_prediction/sklearn_pipeline_components/model_registry.py).
3. Add model defaults and the search grid under [`conf/base/parameters/model_hyperparams/`](conf/base/parameters/model_hyperparams/).
4. Add the candidate name, preprocessor type, and model key in [`ml_experiment.yml`](conf/base/parameters/ml_experiment.yml).

Rerun modeling after changing candidates or search grids:

```bash
./scripts/stage4.sh
```
