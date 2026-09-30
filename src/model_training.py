# Standard library imports
import json
import logging
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Related third-party imports
import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline

# Model types that can be named in the config
MODEL_REGISTRY = {
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "RandomForestRegressor": RandomForestRegressor,
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
}

# Models that accept a random_state, so their results are reproducible
SEEDED_MODELS = {"RandomForestRegressor", "HistGradientBoostingRegressor"}


class ModelTraining:
    """
    A class used to train, compare, select and evaluate machine learning models on student records data.

    Attributes:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing parameters for model training and evaluation.
    preprocessor : sklearn.compose.ColumnTransformer
        A preprocessor pipeline for transforming numerical, nominal, and passthrough features.
    cv : sklearn.model_selection.KFold
        The folds shared by every model, so their cross-validation scores are comparable.
    """

    def __init__(self, config: Dict[str, Any], preprocessor: ColumnTransformer):
        """
        Initialize the ModelTraining class with configuration and preprocessor.

        Args:
        -----
        config (Dict[str, Any]): Configuration dictionary containing parameters for model training and evaluation.
        preprocessor (sklearn.compose.ColumnTransformer): A preprocessor pipeline for transforming numerical, nominal, and passthrough features.
        """
        self.config = config
        self.preprocessor = preprocessor
        self.cv = KFold(
            n_splits=config["cv"], shuffle=True, random_state=config["random_state"]
        )

    def split_data(
        self, df: pd.DataFrame
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
    ]:
        """
        Split the data into training, validation, and test sets.

        Args:
        -----
        df (pd.DataFrame): The input DataFrame containing the cleaned data.

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]: A tuple containing the training, validation, and test features and target variables.
        """
        logging.info("Starting data splitting.")
        X = df.drop(columns=self.config["target_column"])
        y = df[self.config["target_column"]]
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=self.config["val_test_size"],
            random_state=self.config["random_state"],
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=self.config["val_size"],
            random_state=self.config["random_state"],
        )
        logging.info(
            f"Data split into train {X_train.shape}, validation {X_val.shape}, test {X_test.shape}."
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def build_pipeline(self, model_name: str) -> Pipeline:
        """
        Build a pipeline for a model in the config, with a fresh copy of the preprocessor.

        Args:
        -----
        model_name (str): The model's name in the config.

        Returns:
        --------
        Pipeline: The untrained preprocessing and model pipeline.
        """
        spec = self.config["models"][model_name]
        params = dict(spec.get("init_params") or {})
        if spec["type"] in SEEDED_MODELS:
            params.setdefault("random_state", self.config["random_state"])
        model = MODEL_REGISTRY[spec["type"]](**params)
        return Pipeline(
            steps=[("preprocessor", clone(self.preprocessor)), ("regressor", model)]
        )

    def cross_validate_baselines(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> pd.DataFrame:
        """
        Score every model with its default settings using cross-validation on the training set.

        Args:
        -----
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target variable.

        Returns:
        --------
        pd.DataFrame: CV MAE, RMSE and R² (mean and standard deviation) for each model, best first.
        """
        logging.info("Cross-validating baseline models with default settings.")
        results = []
        for model_name in self.config["models"]:
            scores = cross_validate(
                self.build_pipeline(model_name),
                X_train,
                y_train,
                cv=self.cv,
                scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"],
                n_jobs=-1,
            )
            results.append(
                {
                    "model": model_name,
                    "CV MAE": -scores["test_neg_mean_absolute_error"].mean(),
                    "CV RMSE": -scores["test_neg_root_mean_squared_error"].mean(),
                    "CV R² (mean)": scores["test_r2"].mean(),
                    "CV R² (std)": scores["test_r2"].std(),
                }
            )
            logging.info(
                f"{model_name} baseline CV R²: {results[-1]['CV R² (mean)']:.4f} ± {results[-1]['CV R² (std)']:.4f}"
            )
        return pd.DataFrame(results).sort_values("CV R² (mean)", ascending=False)

    def tune_models(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Dict[str, Dict[str, Any]]:
        """
        Tune each model with the search set in the config, using cross-validation on the training set.
        Models with no search are scored with cross-validation at their default settings.

        Args:
        -----
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target variable.

        Returns:
        --------
        Dict[str, Dict[str, Any]]: For each model, the fitted best pipeline, its CV score (mean and
            standard deviation) and its best hyperparameters.
        """
        logging.info("Starting hyperparameter tuning.")
        scoring = self.config["scoring"]
        candidates = {}

        for model_name, spec in self.config["models"].items():
            pipeline = self.build_pipeline(model_name)
            search_type = spec.get("search", "none")

            if search_type == "none":
                scores = cross_validate(
                    pipeline, X_train, y_train, cv=self.cv, scoring=scoring, n_jobs=-1
                )["test_score"]
                candidates[model_name] = {
                    "model": pipeline.fit(X_train, y_train),
                    "cv_mean": scores.mean(),
                    "cv_std": scores.std(),
                    "best_params": {},
                }
            else:
                params = {
                    f"regressor__{name}": self._parse_param(value)
                    for name, value in spec["params"].items()
                }
                if search_type == "grid":
                    search = GridSearchCV(
                        pipeline, params, cv=self.cv, scoring=scoring, n_jobs=-1
                    )
                else:
                    search = RandomizedSearchCV(
                        pipeline,
                        params,
                        n_iter=spec.get("n_iter", 10),
                        cv=self.cv,
                        scoring=scoring,
                        random_state=self.config["random_state"],
                        n_jobs=-1,
                    )
                search.fit(X_train, y_train)
                candidates[model_name] = {
                    "model": search.best_estimator_,
                    "cv_mean": search.cv_results_["mean_test_score"][search.best_index_],
                    "cv_std": search.cv_results_["std_test_score"][search.best_index_],
                    "best_params": {
                        k.replace("regressor__", ""): v.item() if isinstance(v, np.generic) else v
                        for k, v in search.best_params_.items()
                    },
                }

            c = candidates[model_name]
            logging.info(
                f"{model_name} tuned CV {scoring}: {c['cv_mean']:.4f} ± {c['cv_std']:.4f} "
                f"with {c['best_params'] or 'default settings'}"
            )

        logging.info("Hyperparameter tuning completed.")
        return candidates

    def compare_on_validation(
        self,
        candidates: Dict[str, Dict[str, Any]],
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> pd.DataFrame:
        """
        Evaluate every tuned model on the validation set, alongside its CV score.

        Args:
        -----
        candidates (Dict[str, Dict[str, Any]]): The tuned models from tune_models.
        X_val (pd.DataFrame): The validation features.
        y_val (pd.Series): The validation target variable.

        Returns:
        --------
        pd.DataFrame: Validation metrics and CV scores for each model, best CV score first.
        """
        rows = []
        for model_name, c in candidates.items():
            metrics = self._evaluate_model(
                c["model"], X_val, y_val, f"{model_name} Validation Metrics:"
            )
            rows.append(
                {
                    "model": model_name,
                    **metrics,
                    "CV R² (mean)": c["cv_mean"],
                    "CV R² (std)": c["cv_std"],
                }
            )
        return pd.DataFrame(rows).sort_values("CV R² (mean)", ascending=False)

    def select_model(self, candidates: Dict[str, Dict[str, Any]]) -> str:
        """
        Select the final model: the highest mean CV score, but if a simpler model (listed earlier
        in the config) is within one standard deviation of the best, it is treated as tied and chosen instead.

        Args:
        -----
        candidates (Dict[str, Dict[str, Any]]): The tuned models from tune_models.

        Returns:
        --------
        str: The name of the selected model.

        Raises:
        -------
        ValueError: If there are no candidates, or none has a valid CV score.
        """
        valid = {n: c for n, c in candidates.items() if np.isfinite(c["cv_mean"])}
        if not valid:
            raise ValueError("No model produced a valid cross-validation score.")

        complexity_order: List[str] = list(self.config["models"])
        best_name = max(valid, key=lambda n: valid[n]["cv_mean"])
        best_mean, best_std = valid[best_name]["cv_mean"], valid[best_name]["cv_std"]
        tied = [n for n in valid if best_mean - valid[n]["cv_mean"] <= best_std]
        selected = min(tied, key=complexity_order.index)

        logging.info(f"Highest CV score: {best_name} ({best_mean:.4f} ± {best_std:.4f})")
        logging.info(f"Models tied within one standard deviation: {tied}")
        logging.info(f"Selected model: {selected}")
        return selected

    def refit_and_evaluate(
        self,
        model: Pipeline,
        model_name: str,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
    ) -> Tuple[Pipeline, Dict[str, float], pd.Series]:
        """
        Refit the selected model on the combined training and validation data, then evaluate it once on the test set.

        Args:
        -----
        model (Pipeline): The selected tuned pipeline.
        model_name (str): The name of the selected model.
        X_train, X_val, X_test (pd.DataFrame): The training, validation and test features.
        y_train, y_val, y_test (pd.Series): The training, validation and test target variables.

        Returns:
        --------
        Tuple[Pipeline, Dict[str, float], pd.Series]: The refitted pipeline, its test metrics and its test predictions.
        """
        logging.info("Refitting the selected model on the training and validation data.")
        final_model = clone(model)
        final_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
        test_metrics = self._evaluate_model(
            final_model, X_test, y_test, f"Final Test Metrics for {model_name}:"
        )
        predictions = pd.Series(
            final_model.predict(X_test), index=X_test.index, name="predicted"
        )
        return final_model, test_metrics, predictions

    def save_artifacts(
        self,
        final_model: Pipeline,
        model_name: str,
        candidates: Dict[str, Dict[str, Any]],
        baseline_df: pd.DataFrame,
        comparison_df: pd.DataFrame,
        test_metrics: Dict[str, float],
        y_test: pd.Series,
        predictions: pd.Series,
    ) -> Path:
        """
        Save the final pipeline, its metrics, the model comparisons, the test predictions and a
        snapshot of the config and library versions to a new timestamped folder.

        Returns:
        --------
        Path: The folder the run was saved to.
        """
        run_dir = Path(self.config["output_dir"]) / datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(final_model, run_dir / "final_model.joblib")
        val_row = comparison_df.set_index("model").loc[model_name]
        record = {
            "selected_model": model_name,
            "hyperparameters": candidates[model_name]["best_params"],
            "cv_r2_mean": round(float(candidates[model_name]["cv_mean"]), 4),
            "cv_r2_std": round(float(candidates[model_name]["cv_std"]), 4),
            "validation_metrics": {
                k: round(float(val_row[k]), 4) for k in ["MAE", "MSE", "RMSE", "R²"]
            },
            "test_metrics": {k: round(float(v), 4) for k, v in test_metrics.items()},
            "environment": {
                "python": platform.python_version(),
                "scikit-learn": sklearn.__version__,
                "pandas": pd.__version__,
                "numpy": np.__version__,
            },
        }
        with open(run_dir / "final_metrics.json", "w") as file:
            json.dump(record, file, indent=2, ensure_ascii=False)
        with open(run_dir / "config.yaml", "w") as file:
            yaml.safe_dump(self.config, file, sort_keys=False)

        baseline_df.round(4).to_csv(run_dir / "baseline_comparison.csv", index=False)
        comparison_df.round(4).to_csv(run_dir / "model_comparison.csv", index=False)
        pd.DataFrame({"actual": y_test, "predicted": predictions}).to_csv(
            run_dir / "test_predictions.csv"
        )
        logging.info(f"Saved the final model and results to {run_dir}")
        return run_dir

    @staticmethod
    def _parse_param(value: Any) -> Any:
        """
        Convert a hyperparameter from the config into a search space: a list is used as-is,
        and a {dist, low, high} dictionary becomes a scipy distribution.

        Args:
        -----
        value (Any): A list of values, or a dictionary describing a distribution.

        Returns:
        --------
        Any: A list or a scipy distribution for the search.
        """
        if not isinstance(value, dict):
            return value
        dist, low, high = value["dist"], value["low"], value["high"]
        if dist == "randint":
            return randint(low, high)
        if dist == "loguniform":
            return loguniform(low, high)
        if dist == "uniform":
            return uniform(low, high - low)
        raise ValueError(f"Unknown distribution '{dist}'. Use randint, uniform or loguniform.")

    @staticmethod
    def _evaluate_model(
        model: Pipeline, X: pd.DataFrame, y: pd.Series, header: str
    ) -> Dict[str, float]:
        """
        Evaluate a model on the given data and log the metrics.

        Args:
        -----
        model (Pipeline): The trained model pipeline.
        X (pd.DataFrame): The features to predict on.
        y (pd.Series): The true target variable.
        header (str): The line logged before the metrics.

        Returns:
        --------
        Dict[str, float]: A dictionary containing the evaluation metrics.
        """
        y_pred = model.predict(X)
        metrics = {
            "MAE": mean_absolute_error(y, y_pred),
            "MSE": mean_squared_error(y, y_pred),
            "RMSE": root_mean_squared_error(y, y_pred),
            "R²": r2_score(y, y_pred),
        }
        logging.info(header)
        for metric_name, metric_value in metrics.items():
            logging.info(f"{metric_name}: {metric_value:.4f}")
        return metrics
