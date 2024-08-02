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
from dotenv import load_dotenv
import os
import datetime
import time


# 載入 .env 文件
load_dotenv()

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

    # Read the wine-quality csv file from the URL
    data = pd.read_csv("../Data/bronze/red-wine-quality.csv")
    data.to_csv("../Data/silver/red-wine-quality.csv", index=False)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data, test_size=0.25, random_state=42)
    train.to_csv("../Data/silver/train.csv")
    test.to_csv("../Data/silver/test.csv")

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = args.alpha
    l1_ratio = args.l1_ratio

    mlflow.set_tracking_uri(uri="")

    print("The set tracking uri is ", mlflow.get_tracking_uri())
    exp = mlflow.set_experiment(experiment_name="model_1")

    print("Name: {}".format(exp.name))
    print("Experiment_id: {}".format(exp.experiment_id))
    print("Artifact Location: {}".format(exp.artifact_location))
    print("Tags: {}".format(exp.tags))
    print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
    print("Creation timestamp: {}".format(exp.creation_time))

    start_time = time.time()
    start_datetime = datetime.datetime.now().isoformat()

    mlflow.start_run(run_name="param_1")
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    # log parameters
    params = {
        "alpha": alpha,
        "l1_ratio": l1_ratio
    }
    mlflow.log_params(params)
    # log metrics
    metrics = {
        "rmse": rmse,
        "r2": r2,
        "mae": mae
    }
    mlflow.log_metrics(metrics)
    # log model
    mlflow.sklearn.log_model(lr, "model_1_param_1")
    mlflow.log_artifacts("../Data/")

    end_time = time.time()
    end_datetime = datetime.datetime.now().isoformat()
    duration_time = end_time - start_time

    tags = {
        "engineering": "ML platform",  # 工程標籤，用於標識涉及的工程團隊或平台
        "experiment_id": exp.experiment_id,  # 實驗ID，用於唯一標識一個實驗
        "experiment_name": exp.name,  # 實驗名稱，用於將運行分組到特定實驗中
        "run_id": mlflow.active_run().info.run_id,  # 當前運行的唯一標識符
        "run_name": mlflow.active_run().info.run_name,  # 運行名稱，用於標識特定運行
        "model_type": "ElasticNet",  # 模型類型標籤，用於標識所使用的模型類型
        "data_version": "v1.2",  # 數據版本標籤，用於標識所使用的數據集版本
        "run_type": "training",  # 運行類型標籤，用於標識運行的類型（如訓練、驗證等）
        "start_time": start_datetime,  # 開始時間標籤，用於記錄運行的開始時間
        "end_time": end_datetime,  # 結束時間標籤，用於記錄運行的結束時間
        "duration_time": duration_time,  # 持續時間標籤，用於記錄運行的持續時間
        "user": os.getenv("USER", "unknown"),  # 用戶標籤，用於標識執行運行的用戶
        "compute_resource": "CPU",  # 計算資源標籤，用於標識運行時使用的計算資源
        "source_version": "commit_abc123",  # 源碼版本標籤，用於標識所使用的源碼版本
        "release.status": "dev",  # 發佈狀態標籤，用於標識發佈候選版本的狀態（dev, test, prod）
        "notes": "Initial run with baseline parameters"  # 備註標籤，用於添加任何補充說明
    }

    mlflow.set_tags(tags)

    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)

    mlflow.end_run()

    run = mlflow.last_active_run()
    print("Active run id is {}".format(run.info.run_id))
    print("Active run name is {}".format(run.info.run_name))


