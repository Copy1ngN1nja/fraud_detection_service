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
def make_pred(dt, source="kafka_transaction_topic"):
    predictions = BOOSTER.predict(dt)
    y_pred = (predictions > BEST_THRESHOLD) * 1

    # Make submission dataframe (fraud_flag - имя, ожидаемое score_writer'ом
    # и колонкой scores.fraud_flag в Postgres)
    submission = pd.DataFrame({
        'score':  predictions,
        'fraud_flag': y_pred.astype(int)
    })

    logger.info('Prediction complete for data from: %s', source)

    # Return proba for positive class
    return submission, predictions, y_pred
