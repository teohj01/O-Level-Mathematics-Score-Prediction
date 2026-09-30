# O-Level Mathematics Score Prediction

## Project Description
This project implements a machine learning pipeline to predict students' O-Level mathematics examination scores, to help the school identify weaker students before the examination. The pipeline includes data cleaning, preprocessing, model training, and evaluation. Multiple regression models are trained and compared to find the best performing one based on various metrics.

## Pre-requisites and Installation Instructions

### Requirements
- Anaconda or Miniconda
- Python 3.11
- Required packages listed in `requirements.txt`

### Installation
1. Clone the repository:
    ```
    git clone https://github.com/teohj01/O-Level-Mathematics-Score-Prediction.git
    cd O-Level-Mathematics-Score-Prediction
    ```

2. Create and activate a conda environment:
    ```
    conda create -n olevelenv python=3.11
    conda activate olevelenv
    ```

3. Install the required packages:
    ```
    pip install -r requirements.txt
    ```

## Pipeline Execution
To run the complete pipeline, execute:
```
python main.py
```

This will:
1. Load the configuration and check that every required setting is present and valid
2. Load the data and check that every required column is present
3. Clean the data and engineer features
4. Preprocess features
5. Split data into training, validation, and test sets
6. Train baseline models
7. Compare all models with 5-fold cross-validation
8. Perform hyperparameter tuning
9. Evaluate models and select the best performing model
10. Refit the selected model on the training and validation sets and evaluate it once on the test set
11. Save the final model and results to a timestamped folder in `outputs/`

## Logical Flow of the Pipeline

### 1. Configuration (`config.yaml`)
The pipeline starts with loading configuration parameters from `src/config.yaml`, which includes:
- Data file path
- Target column name and student ID column
- Output folder for the saved model and results
- Data cleaning
- Feature engineering
- Feature categorization (numerical, nominal, passthrough)
- Train/validation/test split ratios
- Cross-validation folds and scoring metric
- Models and their hyperparameter search ranges

### 2. Data Preparation (`DataPreparation` class)
The `DataPreparation` class handles:
- Engineering `class_size` from `n_male` and `n_female`
- Engineering `sleep_hours` from `sleep_time` and `wake_time`
- Correcting invalid values in `age`
- Standardizing `tuition` and `CCA` labels
- Recovering missing `final_test` and `attendance_rate` values from each student's duplicate row
- Removing duplicate students
- Dropping rows that are still missing `final_test`
- Dropping irrelevant features
- Converting `direct_admission`, `tuition` and `learning_style` to binary features
- Creating a preprocessor for feature transformation

### 3. Model Training (`ModelTraining` class)
The `ModelTraining` class manages:
- Splitting data into training, validation and test sets
- Comparing all models (Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting) with 5-fold cross-validation
- Hyperparameter tuning for all models except Linear Regression
- Evaluating models using multiple metrics (MAE, MSE, RMSE, R²)
- Selecting the best model based on cross-validated R² and choosing the simpler model when two are within one standard deviation
- Refitting the selected model on the training and validation sets and evaluating it on the test set
- Saving the final model, its metrics, the model comparison and the test predictions

### 4. Main Execution (`main.py`)
The main script orchestrates the entire pipeline by:
- Loading and validating configuration and data
- Initializing data preparation and model training
- Running the training and evaluation process
- Identifying and evaluating the best model on the test set
- Predicting scores for new students with a saved model
- Stopping with a clear error message if a file, setting or column is missing

## Key Findings and Feature Handling

### Exploratory Data Analysis

Before diving into the model training process, an exploratory data analysis (EDA) was conducted to understand the data better and identify any potential issues. Here are some key findings from the EDA:

**EDA Findings and Explanations**:

#### Dataset Overview
- The dataset contains 15,900 student records across 18 columns, covering student background, study habits, class composition, daily routine, and O-level mathematics score.
- 900 duplicate rows were identified and removed, reducing the dataset to 15,000 students.
- `CCA`, `final_test` and `attendance_rate` contain missing values, and `age` contains invalid single-digit and negative values.

#### Univariate Analysis
- About 57% of students take tuition, after 'Y' and 'N' were merged into 'Yes' and 'No'.
- 'Clubs', 'Sports', 'No CCA' and 'Arts' are almost evenly split, though CCA is compulsory under Ministry of Education (MOE), suggesting some students are exempted or their CCA was not recorded.
- Final test score is approximately symmetric, ranging from 32 to 100, with a mean of 67 close to the median of 68.
- Most students sleep between 21:00 and 23:00 and get 8 hours of sleep, while late sleepers get only 4 to 7 hours, as wake time is fixed by mode of transport.

#### Bivariate Analysis
- Median final test score is higher for DSA students, visual learners and students with tuition, and highest for students with no CCA.
- `gender`, `mode_of_transport`, `bag_color` and CCA type show almost identical score distributions, suggesting little relationship with final test score.
- Attendance rate and final test score are non-linearly related; students below 90% attendance all scored 50 or below, suggesting attendance acts as a minimum requirement rather than a steady predictor.

#### Correlation Analysis
- Spearman rank correlation was used instead of Pearson because several features take only a few ordered values and some relationships are non-linear, and it captures monotonic relationships.
- `class_size` and `number_of_siblings` show a moderate negative correlation with `final_test`.
- `sleep_hours` and `attendance_rate` show a weak positive correlation with `final_test`.
- `hours_per_week` shows a weak negative correlation.
- `age` shows no correlation with `final_test`.

### Data Cleaning
- Sleep hours are calculated from `sleep_time` and `wake_time` because the original values were clock times like '22:30', and converting them into a single duration makes them usable as a numerical feature.
- Class size is engineered by adding `n_male` and `n_female` because each only captures part of the class, and their sum correlates much more strongly with `final_test`.
- Invalid ages are corrected because ages of 5 and 6 were likely missing the leading '1', and negative ages are imputed with the mode (16) as their true values cannot be determined.
- `tuition` and `CCA` labels are standardised because the original values were inconsistent (e.g. 'Y' and 'Yes', 'SPORTS' and 'Sports'), and missing CCAs are combined with 'NONE' into 'No CCA' as both groups have almost the same scores and study habits.
- Missing `final_test` and `attendance_rate` values are filled from each student's duplicate row because both rows belong to the same student, so the values can be reliably recovered rather than dropped or imputed arbitrarily.
- Rows still missing `final_test` (about 3%) are dropped because it is the target variable, and imputing it would create made-up scores for the model to train and be evaluated on.
- Duplicate students are removed because keeping both records would double-weight those students and could place the same student in both the training and test sets.
- Irrelevant features are dropped because they are identifiers (`index`, `student_id`), show no relationship with `final_test` (`gender`, `age`, `mode_of_transport`, `bag_color`), or were replaced by engineered features (`n_male`, `n_female`, `sleep_time`, `wake_time`).
- `direct_admission`, `tuition` and `learning_style` are converted to 1/0 because they only have two categories each, and machine learning models require numeric input.
- `attendance_rate` and `sleep_hours` are both kept despite being moderately correlated because their VIF of around 4.7 is below the commonly used threshold of 5.

### Feature Processing
- **Numerical Features**: `number_of_siblings`, `hours_per_week`, `attendance_rate`, `class_size`, `sleep_hours`
- Imputed with the median using `SimpleImputer` because `attendance_rate` still has missing values, and fitting the imputer inside the pipeline calculates the median from the training set only, avoiding data leakage.
- Standardized using `StandardScaler` because the numerical features vary significantly in scale, and standardisation suits the scale-sensitive algorithms (linear models) planned for this analysis.
- **Nominal Features**: `CCA` - Encoded using `OneHotEncoder` because these are unordered categories, there is no inherent ranking among CCA types, so one-hot encoding avoids implying a false ordinal relationship.
- **Passthrough Features**: `direct_admission`, `learning_style`, `tuition` - Used as-is because they were already converted into binary values during data cleaning, so no additional encoding was needed.

## Model Choices and Evaluation

### Models Implemented and Justifications
1. **Linear Regression**: Basic model without regularization because it serves as a simple, interpretable baseline against which the performance of the other models can be measured.
2. **Ridge Regression**: Linear regression with L2 regularization because it shrinks coefficient magnitudes to reduce overfitting and stabilize estimates without eliminating any features entirely, which is useful given the correlation between `attendance_rate` and `sleep_hours`.
3. **Lasso Regression**: Linear regression with L1 regularization because it can shrink some coefficients exactly to zero, effectively performing feature selection, which helps identify whether any of the weaker predictors add little to the model.
4. **Random Forest**: An ensemble of decision trees because it can capture threshold effects and interactions that linear models cannot, such as students below about 90% attendance all scoring 50 or below.
5. **Gradient Boosting**: `HistGradientBoostingRegressor`, which builds trees one after another to correct earlier errors, because it often performs best on tabular data and checks whether extra complexity beyond Random Forest is worthwhile.

### Hyperparameter Tuning
- Grid search is performed for Ridge and Lasso models because each has one main hyperparameter, `alpha`, and grid search systematically evaluates every value from 0.0001 to 1000 on a log scale, so the best value is not stuck at the edge of the search range.
- `fit_intercept` is not tuned and is left at `True` because the intercept is the model's starting prediction for an average student, which should be close to the average score of about 67. Setting it to `False` would force this starting prediction to zero.
- Randomized search is performed for Random Forest and Gradient Boosting because they have several hyperparameters (such as tree depth, leaf size, number of trees and learning rate), and testing every combination would take too long. Randomized search tests 15 and 20 random combinations respectively, which usually finds a near-best combination in far less time.
- 5-fold cross-validation is used during tuning because it evaluates each hyperparameter combination across 5 different train/validation splits of the training data, reducing the risk of selecting a hyperparameter that only performs well on one specific split by chance.
- The final model is selected using cross-validated R² because it shows both typical performance and how stable it is across folds. If another model's score is within one standard deviation of the best, the two are treated as tied and the simpler model is chosen, as it is easier to explain and less likely to overfit.

### Evaluation Metrics
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual scores
- **MSE (Mean Squared Error)**: Average squared difference between predicted and actual scores
- **RMSE (Root Mean Squared Error)**: Square root of MSE, in the same unit as the target
- **R² (Coefficient of Determination)**: Proportion of variance in scores explained by the model

### Model Comparison and Results
The data is split into 80% training (11,647 students), 10% validation (1,456) and 10% test (1,456). All models are compared on the validation set, and the selected model is then evaluated once on the test set.

| Model | MAE | MSE | RMSE | R² |
|---|---|---|---|---|
| Gradient Boosting (Tuned) | 3.8072 | 29.7042 | 5.4502 | 0.8464 |
| Random Forest (Tuned) | 3.7307 | 29.9823 | 5.4756 | 0.8450 |
| Ridge Regression (Tuned alpha=10) | 7.2743 | 82.0592 | 9.0587 | 0.5757 |
| Lasso Regression (Tuned alpha=0.01) | 7.2754 | 82.0692 | 9.0592 | 0.5756 |
| Linear Regression | 7.2749 | 82.0724 | 9.0594 | 0.5756 |

- **Random Forest** is selected as the final model. Gradient Boosting has the highest cross-validated R² (0.853), but Random Forest (0.849) is within one standard deviation of it, so the two are effectively tied and the simpler model is chosen. On the validation set, Random Forest also has the lowest MAE.
- The non-linear models explain far more of the variation in scores (R² about 0.85) than the linear models (about 0.58), with roughly half the average error. This supports the EDA finding that some relationships, such as the attendance threshold, are not linear.
- Regularisation made no difference to the linear models, as there are far more students than features, so they are not overfitting to begin with.
- After refitting on the training and validation sets, Random Forest achieves **MAE 3.78, MSE 29.02, RMSE 5.39 and R² 0.8508** on the test set, very close to its validation results. This suggests the model generalises well to unseen students.
- In practical terms, the model's predictions are off by about 4 marks on average (MAE), which is accurate enough to flag students likely to score low for early support. However, as a Random Forest, it is harder to explain to teachers than a linear model, as there are no simple coefficients showing how each feature affects the score.
- The model explains about 85% of the variation in scores. Part of the remainder is likely due to factors not in the dataset, such as a student's ability in earlier examinations. `sleep_hours` should also be interpreted with caution, as about 92% of students sleep 8 hours, so its effect is estimated from a small group of students.
