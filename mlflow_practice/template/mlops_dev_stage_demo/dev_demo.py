from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import optuna
import mlflow
import warnings
import argparse
import logging
import pandas as pd
import numpy as np

pipeline = Pipeline(
    steps=[
        ("pca", PCA()),
        ("classifier", RandomForestClassifier())
    ]
)

# Define objective function for Optuna
def objective(trial):
    # Define parameters to tune
    n_components = trial.suggest_int("n_components", 2, 4)
    max_depth = trial.suggest_int("max_depth", 1, 10)

    # Set parameters for the current trial
    pipeline.set_params(pca__n_components=n_components, classifier__max_depth=max_depth)

    # Evaluate pipeline performance with cross-validation
    scores = cross_val_score(pipeline, X_train, y_train, cv=5)

    # Log parameters and metrics to MLflow with a unique run name
    run_name = f"p_{trial.number}"
    with mlflow.start_run(nested=True, run_name=run_name):
        mlflow.log_params({
            "n_components": n_components,
            "max_depth": max_depth
        })
        mlflow.log_metric("cv_score", scores.mean())

    # Return average cross-validation score as the objective value
    return scores.mean()

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    data = pd.read_csv("../Data/bronze/red-wine-quality.csv")

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)
    train.to_csv("./Data/silver/train.csv")
    test.to_csv("./Data/silver/test.csv")

    # The predicted column is "quality" which is a scalar from [3, 9]
    X_train = train.drop(["quality"], axis=1)
    X_test = test.drop(["quality"], axis=1)
    y_train = train[["quality"]]
    y_test = test[["quality"]]

    # Set MLflow experiment name
    mlflow.set_experiment('red-wine-quality_pipeline_optimization')

    # Perform hyperparameter optimization with Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    # Get the best parameters
    best_params = study.best_params

    # Log the best parameters and results to MLflow
    with mlflow.start_run():
        mlflow.log_params(best_params)
        mlflow.log_metric("Best cross-validation score", study.best_value)

        # Create pipeline with best parameters
        pipeline.set_params(
            pca__n_components=best_params["n_components"],
            classifier__max_depth=best_params["max_depth"]
        )

        # Train model with best parameters
        pipeline.fit(X_train, y_train)

        # Log the best model
        mlflow.sklearn.log_model(pipeline, "red-wine-quality_pipeline")
