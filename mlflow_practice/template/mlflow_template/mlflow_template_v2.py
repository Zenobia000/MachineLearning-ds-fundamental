import warnings
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn
from pathlib import Path
from dotenv import load_dotenv
import os
import datetime
import time
import yaml

# 載入 .env 文件
load_dotenv()

# 載入配置文件
def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# evaluation function
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def load_and_preprocess_data(input_path, output_path):
    create_folder_if_not_exists(os.path.dirname(output_path))
    data = pd.read_csv(input_path)
    data.to_csv(output_path, index=False)
    return data

def split_data(data):
    train, test = train_test_split(data)
    return train, test

def save_splits(train, test, train_path, test_path):
    create_folder_if_not_exists(os.path.dirname(train_path))
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

def train_model(train_x, train_y, alpha, l1_ratio):
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    model.fit(train_x, train_y)
    return model

def log_metrics_and_params(mlflow, alpha, l1_ratio, rmse, mae, r2, model, run_name, data_path):
    params = {
        "alpha": alpha,
        "l1_ratio": l1_ratio
    }
    mlflow.log_params(params)
    metrics = {
        "rmse": rmse,
        "r2": r2,
        "mae": mae
    }
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, run_name)
    mlflow.log_artifacts(data_path)

def set_mlflow_tags(mlflow, exp, start_datetime, end_datetime, duration_time, tags_config):
    tags = tags_config.copy()
    tags.update({
        "experiment_id": exp.experiment_id,
        "experiment_name": exp.name,
        "run_id": mlflow.active_run().info.run_id,
        "run_name": mlflow.active_run().info.run_name,
        "start_time": start_datetime,
        "end_time": end_datetime,
        "duration_time": duration_time,
        "user": os.getenv("USER", "unknown")
    })
    mlflow.set_tags(tags)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    config = load_config()

    data = load_and_preprocess_data(config['data_path'], config['output_path'])
    train, test = split_data(data)
    save_splits(train, test, config['train_path'], config['test_path'])

    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = config['alpha']
    l1_ratio = config['l1_ratio']

    mlflow.set_tracking_uri(uri="")

    exp = mlflow.set_experiment(experiment_name=config['experiment_name'])

    start_time = time.time()
    start_datetime = datetime.datetime.now().isoformat()

    mlflow.start_run(run_name=config['run_name'])
    model = train_model(train_x, train_y, alpha, l1_ratio)

    predicted_qualities = model.predict(test_x)
    rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

    log_metrics_and_params(mlflow, alpha, l1_ratio, rmse, mae, r2, model, config['model_name'], "../Data/")

    end_time = time.time()
    end_datetime = datetime.datetime.now().isoformat()
    duration_time = end_time - start_time

    set_mlflow_tags(mlflow, exp, start_datetime, end_datetime, duration_time, config['tags'])

    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)

    mlflow.end_run()

    run = mlflow.last_active_run()
    print("Active run id is {}".format(run.info.run_id))
    print("Active run name is {}".format(run.info.run_name))
