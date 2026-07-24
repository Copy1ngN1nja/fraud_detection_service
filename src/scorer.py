import pandas as pd
import logging
import lightgbm as lgb

# Настройка логгера
logger = logging.getLogger(__name__)

logger.info('Importing pretrained model...')

# Import model
BOOSTER = lgb.Booster(model_file='./models/tuned_lgbm_model.txt')

# Define optimal threshold
BEST_THRESHOLD = 0.4944105290781792
logger.info('Pretrained model imported successfully...')

# Make prediction
def make_pred(dt, path_to_file):
    predictions = BOOSTER.predict(dt)
    # Make submission dataframe
    submission = pd.DataFrame({
        'index':  pd.read_csv(path_to_file).index,
        'prediction': (predictions > BEST_THRESHOLD) * 1
    })
    logger.info('Prediction complete for file: %s', path_to_file)

    # Return proba for positive class
    return submission, predictions
