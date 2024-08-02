import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='app.log',
                    filemode='w')

logger = logging.getLogger(__name__)

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=False, default=0.5)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.5)
args = parser.parse_args()

logger.info(f'Arguments received: alpha={args.alpha}, l1_ratio={args.l1_ratio}')

# evaluation function
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Example data loading (replace with your dataset)
    try:
        data = pd.read_csv('./data/red-wine-quality.csv')
        logger.info('Data loaded successfully')
    except Exception as e:
        logger.error('Error loading data', exc_info=True)
        raise e


    X = data.drop(columns=['quality'])
    y = data['quality']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info('Data split into training and testing sets')

    alpha = args.alpha
    l1_ratio = args.l1_ratio

    # Initialize and train the ElasticNet model
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    model.fit(X_train, y_train)
    logger.info('Model trained successfully')

    # Make predictions
    predicted_qualities = model.predict(X_test)

    # Evaluate the model
    (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)

    # Print and log the evaluation metrics
    print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
    # Print and log the evaluation metrics
    print(f"Mean Squared Error: {rmse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R^2 Score: {r2}")

    logger.info(f"Mean Squared Error: {rmse}")
    logger.info(f"Mean Absolute Error: {mae}")
    logger.info(f"R^2 Score: {r2}")

