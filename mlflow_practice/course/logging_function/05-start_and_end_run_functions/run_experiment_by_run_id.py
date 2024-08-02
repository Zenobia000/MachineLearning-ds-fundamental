import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn
from pathlib import Path


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# get arguments from command
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=False, default=0.7)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.7)
args = parser.parse_args()

# evaluation function
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from local
    data = pd.read_csv("../02-Tracking_component/data/red-wine-quality.csv")

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = args.alpha
    l1_ratio = args.l1_ratio

    mlflow.set_tracking_uri(uri="")

    print("The set tracking uri is ", mlflow.get_tracking_uri())

    # An instance of mlflow.entities.Experiment representing the new active experiment.
    exp = mlflow.set_experiment(experiment_name="experiment_2")

    print("Name: {}".format(exp.name))
    print("Experiment_id: {}".format(exp.experiment_id))
    print("Artifact Location: {}".format(exp.artifact_location))
    print("Tags: {}".format(exp.tags))
    print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
    print("Creation timestamp: {}".format(exp.creation_time))

    with mlflow.start_run(run_id="6520e59a04a940f9abaef6f68616b7eb"):
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(lr, "my_new_model_v3")

    # 如果不使用 with 進行資源管理，以下寫法也是可以。

    # mlflow.start_run(experiment_id=exp.experiment_id, run_name="E1"):
    # lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    # lr.fit(train_x, train_y)
    #
    # predicted_qualities = lr.predict(test_x)
    #
    # (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
    #
    # print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
    # print("  RMSE: %s" % rmse)
    # print("  MAE: %s" % mae)
    # print("  R2: %s" % r2)
    #
    # mlflow.log_param("alpha", alpha)
    # mlflow.log_param("l1_ratio", l1_ratio)
    # mlflow.log_metric("rmse", rmse)
    # mlflow.log_metric("r2", r2)
    # mlflow.log_metric("mae", mae)
    # mlflow.sklearn.log_model(lr, "my_new_model")
    #
    # mlflow.end_run()

