# Standard library imports
import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Related third-party imports
import joblib
import pandas as pd
import yaml

# Local application/library specific imports
from src.data_preparation import DataPreparation
from src.model_training import MODEL_REGISTRY, ModelTraining

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

REQUIRED_CONFIG_KEYS = [
    "file_path",
    "target_column",
    "id_column",
    "output_dir",
    "recover_columns",
    "dropna_columns",
    "age_corrections",
    "columns_to_drop",
    "binary_mappings",
    "numerical_features",
    "nominal_features",
    "passthrough_features",
    "val_test_size",
    "val_size",
    "random_state",
    "cv",
    "scoring",
    "models",
]


def load_config(config_path: str = "./src/config.yaml") -> Dict[str, Any]:
    """
    Load the configuration file and check that it is complete and valid.

    Args:
    -----
    config_path (str): Path to the YAML configuration file.

    Returns:
    --------
    Dict[str, Any]: The configuration dictionary.

    Raises:
    -------
    FileNotFoundError: If the configuration file does not exist.
    ValueError: If the file is not valid YAML, or any setting is missing or invalid.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    try:
        with open(path, "r") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise ValueError(f"Config file {config_path} is not valid YAML: {e}") from e

    if not isinstance(config, dict):
        raise ValueError(f"Config file {config_path} is empty or not a mapping.")
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"Config is missing required keys: {missing}")

    for key in ["val_test_size", "val_size"]:
        if not 0 < config[key] < 1:
            raise ValueError(f"'{key}' must be between 0 and 1, got {config[key]}.")
    if not isinstance(config["cv"], int) or config["cv"] < 2:
        raise ValueError(f"'cv' must be an integer of at least 2, got {config['cv']}.")
    if not config["models"]:
        raise ValueError("'models' must list at least one model.")
    for name, spec in config["models"].items():
        if spec.get("type") not in MODEL_REGISTRY:
            raise ValueError(
                f"Model '{name}' has unknown type '{spec.get('type')}'. "
                f"Choose from {list(MODEL_REGISTRY)}."
            )
        if spec.get("search", "none") not in ("none", "grid", "random"):
            raise ValueError(f"Model '{name}' search must be none, grid or random.")
        if spec.get("search", "none") != "none" and not spec.get("params"):
            raise ValueError(f"Model '{name}' needs 'params' to search over.")

    logging.info(f"Loaded config from {config_path}")
    return config


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file of student records.

    Args:
    -----
    file_path (str): Path to the CSV file.

    Returns:
    --------
    pd.DataFrame: The loaded data.

    Raises:
    -------
    FileNotFoundError: If the file does not exist.
    ValueError: If the file is empty.
    """
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"Data file is empty: {file_path}") from e
    logging.info(f"Loaded {file_path} with shape {df.shape}")
    return df


def train(config: Dict[str, Any]) -> Path:
    """
    Run the full training pipeline: load, check and clean the data, compare and tune models,
    select one with the one-standard-deviation rule, refit it on the training and validation data,
    evaluate it once on the test set and save the results.

    Args:
    -----
    config (Dict[str, Any]): The configuration dictionary.

    Returns:
    --------
    Path: The folder the run was saved to.
    """
    df = load_data(config["file_path"])
    data_prep = DataPreparation(config)
    data_prep.validate_schema(df)
    cleaned_df = data_prep.clean_data(df)

    model_training = ModelTraining(config, data_prep.preprocessor)
    X_train, X_val, X_test, y_train, y_val, y_test = model_training.split_data(
        cleaned_df
    )

    baseline_df = model_training.cross_validate_baselines(X_train, y_train)
    candidates = model_training.tune_models(X_train, y_train)
    comparison_df = model_training.compare_on_validation(candidates, X_val, y_val)

    selected_name = model_training.select_model(candidates)
    final_model, test_metrics, predictions = model_training.refit_and_evaluate(
        candidates[selected_name]["model"],
        selected_name,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )
    return model_training.save_artifacts(
        final_model,
        selected_name,
        candidates,
        baseline_df,
        comparison_df,
        test_metrics,
        y_test,
        predictions,
    )


def latest_model_path(output_dir: str) -> Path:
    """
    Find the model saved by the most recent training run.

    Args:
    -----
    output_dir (str): The folder that holds the timestamped training runs.

    Returns:
    --------
    Path: The path to the most recent final_model.joblib.
    """
    models = sorted(Path(output_dir).glob("*/final_model.joblib"))
    if not models:
        raise FileNotFoundError(
            f"No trained model found in {output_dir}. Run 'python main.py train' first."
        )
    return models[-1]


def predict(
    config: Dict[str, Any], input_path: str, output_path: str, model_path: Optional[str]
) -> None:
    """
    Predict O-level Mathematics scores for new students with a saved pipeline.
    The input must be in the same raw format as the training data, but final_test is not needed.

    Args:
    -----
    config (Dict[str, Any]): The configuration dictionary.
    input_path (str): Path to a CSV of new student records.
    output_path (str): Path to write the predictions to.
    model_path (Optional[str]): Path to a saved pipeline; the latest training run is used if not given.
    """
    model_file = Path(model_path) if model_path else latest_model_path(config["output_dir"])
    if not model_file.is_file():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    model = joblib.load(model_file)
    logging.info(f"Loaded model from {model_file}")

    df = load_data(input_path)
    data_prep = DataPreparation(config)
    data_prep.validate_schema(df, require_target=False)
    cleaned_df = data_prep.clean_data(df, inference=True)

    id_column = config["id_column"]
    features = cleaned_df.drop(columns=[config["target_column"]], errors="ignore")
    results = pd.DataFrame(
        {
            id_column: cleaned_df[id_column],
            "predicted_final_test": model.predict(features).round(1),
        }
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    logging.info(f"Saved {len(results)} predictions to {output_path}")


def parse_args() -> argparse.Namespace:
    """
    Parse the command-line arguments.

    Returns:
    --------
    argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Predict students' O-level Mathematics scores."
    )
    parser.add_argument(
        "--config", default="./src/config.yaml", help="Path to the YAML config file."
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("train", help="Train, select and evaluate a model (default).")

    predict_parser = subparsers.add_parser(
        "predict", help="Predict scores for new students with a saved model."
    )
    predict_parser.add_argument(
        "--input", required=True, help="CSV of new student records."
    )
    predict_parser.add_argument(
        "--output",
        default="./outputs/predictions.csv",
        help="Where to write the predictions.",
    )
    predict_parser.add_argument(
        "--model", help="Saved model to use (default: the latest training run)."
    )
    return parser.parse_args()


def main():
    """
    Run the pipeline from the command line: 'train' (the default) or 'predict'.
    """
    args = parse_args()
    try:
        config = load_config(args.config)
        if args.command == "predict":
            predict(config, args.input, args.output, args.model)
        else:
            train(config)
    except (FileNotFoundError, ValueError, KeyError) as e:
        logging.error(e)
        sys.exit(1)
    except Exception:
        logging.exception("The pipeline stopped because of an unexpected error.")
        sys.exit(1)


if __name__ == "__main__":
    main()
