# Standard library imports
import logging
from typing import Any, Dict, List

# Related third-party imports
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Raw columns that clean_data reads directly, apart from the target
REQUIRED_RAW_COLUMNS = [
    "student_id",
    "n_male",
    "n_female",
    "sleep_time",
    "wake_time",
    "age",
    "tuition",
    "CCA",
    "attendance_rate",
]

# Engineered features and the raw columns each one is built from
ENGINEERED_FEATURES = {
    "class_size": ["n_male", "n_female"],
    "sleep_hours": ["sleep_time", "wake_time"],
}


class DataPreparation:
    """
    A class used to clean and preprocess student records data.

    1. validate_schema: checks the raw data has the columns cleaning needs.
    2. clean_data: raw student records -> cleaned data with engineered features.
    3. preprocessor: ColumnTransformer to fit on the training set.

    Attributes:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing parameters for data cleaning and preprocessing.
    numerical_features : List[str]
        Numerical features used by the model, without any engineered feature switched off in the config.
    preprocessor : sklearn.compose.ColumnTransformer
        A preprocessor pipeline for transforming numerical, nominal, and passthrough features.
        Missing numerical values are filled with the median inside this pipeline, so the median
        is learned from the training set only.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DataPreparation class with a configuration dictionary.

        Args:
        -----
        config (Dict[str, Any]): Configuration dictionary containing parameters for data cleaning and preprocessing.
        """
        self.config = config
        enabled = config.get("engineered_features", {})
        self.disabled_features = [
            name for name in ENGINEERED_FEATURES if not enabled.get(name, True)
        ]
        self.numerical_features = [
            f for f in config["numerical_features"] if f not in self.disabled_features
        ]
        self.preprocessor = self._create_preprocessor()

    def validate_schema(self, df: pd.DataFrame, require_target: bool = True) -> None:
        """
        Checks that the raw DataFrame is not empty and has every column needed for cleaning.

        Args:
        -----
        df (pd.DataFrame): The raw DataFrame.
        require_target (bool): Whether the target column must be present (False for new students).

        Raises:
        -------
        ValueError: If the DataFrame is empty or any required column is missing.
        """
        if df.empty:
            raise ValueError("The input data is empty.")
        required = REQUIRED_RAW_COLUMNS + list(self.config["binary_mappings"])
        required += self.config["nominal_features"]
        required += [
            f for f in self.numerical_features if f not in ENGINEERED_FEATURES
        ]
        if require_target:
            required.append(self.config["target_column"])
        missing = sorted(set(required) - set(df.columns))
        if missing:
            raise ValueError(f"The input data is missing required columns: {missing}")
        logging.info(f"Schema check passed. Raw data shape: {df.shape}")

    def clean_data(self, df: pd.DataFrame, inference: bool = False) -> pd.DataFrame:
        """
        Cleans the input DataFrame by performing several preprocessing steps.

        Args:
        -----
        df (pd.DataFrame): The input DataFrame containing the raw data.
        inference (bool): True when cleaning new students for prediction. The target is not
            needed, rows are never dropped, and the ID column is kept to label the predictions.

        Returns:
        --------
        pd.DataFrame: The cleaned DataFrame.
        """
        logging.info(f"Starting data cleaning. Shape before cleaning: {df.shape}")
        df = df.copy()
        df["class_size"] = df["n_male"] + df["n_female"]
        df["sleep_hours"] = self._calculate_sleep_hours(
            df["sleep_time"], df["wake_time"]
        )
        df = self._clean_age(df, self.config["age_corrections"])
        df["tuition"] = df["tuition"].replace({"Y": "Yes", "N": "No"})
        df["CCA"] = df["CCA"].fillna("None").str.title().replace("None", "No CCA")

        if not inference:
            df = self._recover_from_duplicates(df, self.config["recover_columns"])
            before = len(df)
            df = df.drop_duplicates(subset=self.config["id_column"], keep="first")
            logging.info(f"Removed {before - len(df)} duplicate student records.")
            before = len(df)
            df = df.dropna(subset=self.config["dropna_columns"])
            logging.info(
                f"Dropped {before - len(df)} rows missing {self.config['dropna_columns']}."
            )
            df = df.reset_index(drop=True)

        columns_to_drop = list(self.config["columns_to_drop"]) + self.disabled_features
        if inference:
            columns_to_drop = [c for c in columns_to_drop if c != self.config["id_column"]]
        df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])

        for column, mapping in self.config["binary_mappings"].items():
            mapped = df[column].map(mapping)
            unknown = df.loc[mapped.isna() & df[column].notna(), column].unique()
            if len(unknown) > 0:
                raise ValueError(
                    f"Unexpected values in '{column}': {list(unknown)}. "
                    f"Expected one of {list(mapping)}."
                )
            df[column] = mapped

        logging.info(f"Data cleaning completed. Shape after cleaning: {df.shape}")
        return df

    def _create_preprocessor(self) -> ColumnTransformer:
        """
        Creates a preprocessor pipeline for transforming numerical, nominal, and passthrough features.
        Columns not listed in the config (such as the ID column) are dropped, so they can never reach the model.

        Returns:
        --------
        sklearn.compose.ColumnTransformer: A ColumnTransformer object for preprocessing the data.
        """
        numerical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        nominal_transformer = Pipeline(
            steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, self.numerical_features),
                ("nom", nominal_transformer, self.config["nominal_features"]),
                ("pass", "passthrough", self.config["passthrough_features"]),
            ],
            remainder="drop",
        )
        return preprocessor

    @staticmethod
    def _calculate_sleep_hours(sleep_time: pd.Series, wake_time: pd.Series) -> pd.Series:
        """
        Calculates the sleep duration in hours from sleep and wake times.
        Times that cannot be read give a missing value, which the preprocessor imputes.

        Args:
        -----
        sleep_time (pd.Series): Sleep times as strings in the format 'HH:MM'.
        wake_time (pd.Series): Wake times as strings in the format 'HH:MM'.

        Returns:
        --------
        pd.Series: The sleep duration in hours, accounting for sleep past midnight.
        """
        sleep = pd.to_datetime(sleep_time, format="%H:%M", errors="coerce")
        wake = pd.to_datetime(wake_time, format="%H:%M", errors="coerce")
        invalid = (sleep.isna() | wake.isna()).sum()
        if invalid:
            logging.warning(f"{invalid} rows have an invalid sleep or wake time.")
        sleep_duration = wake - sleep
        sleep_duration = sleep_duration.where(
            sleep_duration >= pd.Timedelta(0), sleep_duration + pd.Timedelta(days=1)
        )
        return sleep_duration.dt.total_seconds() / 3600

    @staticmethod
    def _clean_age(df: pd.DataFrame, corrections: Dict[int, int]) -> pd.DataFrame:
        """
        Fixes data entry errors in 'age' and imputes negative ages with the mode of valid ages.

        Args:
        -----
        df (pd.DataFrame): The DataFrame containing the 'age' column.
        corrections (Dict[int, int]): Mapping from each mistyped age to its corrected value.

        Returns:
        --------
        pd.DataFrame: The DataFrame with 'age' cleaned.
        """
        n_corrected = df["age"].isin(list(corrections)).sum()
        df["age"] = df["age"].replace(corrections)
        negative = df["age"] < 0
        if negative.any():
            valid_ages = df.loc[df["age"] > 0, "age"]
            if valid_ages.empty:
                raise ValueError("No valid ages to impute negative ages from.")
            df.loc[negative, "age"] = valid_ages.mode()[0]
        logging.info(
            f"Corrected {n_corrected} mistyped ages and imputed {negative.sum()} negative ages."
        )
        return df

    @staticmethod
    def _recover_from_duplicates(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Fills missing values in each column using the same student's duplicate row(s).

        Args:
        -----
        df (pd.DataFrame): The DataFrame containing a 'student_id' column.
        columns (List[str]): The names of the columns to be filled.

        Returns:
        --------
        pd.DataFrame: The DataFrame with missing values recovered where possible.
        """
        for column in columns:
            before = df[column].isna().sum()
            first_valid = df.groupby("student_id")[column].transform("first")
            df[column] = df[column].fillna(first_valid)
            logging.info(
                f"Recovered {before - df[column].isna().sum()} missing '{column}' values from duplicate rows."
            )
        return df
