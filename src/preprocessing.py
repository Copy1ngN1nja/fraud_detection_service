# Import standard libraries
import math
import joblib
import pandas as pd
import numpy as np
import logging

# Import extra modules
from sklearn.impute import SimpleImputer 

logger = logging.getLogger(__name__)
RANDOM_STATE = 42
CAT_COLS = ['merch', 'cat_id', 'name_1', 'name_2', 'street', 'one_city', 'us_state', 'jobs']

def extract_time_features(input_df: pd.DataFrame) -> pd.DataFrame:
    logger.debug('Adding time features...')
    output_df = input_df.copy()

    output_df['transaction_time'] = pd.to_datetime(output_df['transaction_time'])
    output_df['year'] = output_df['transaction_time'].dt.year
    output_df['month'] = output_df['transaction_time'].dt.month
    output_df['day'] = output_df['transaction_time'].dt.day
    output_df['hour'] = output_df['transaction_time'].dt.hour
    output_df['minute'] = output_df['transaction_time'].dt.minute

    output_df = output_df.drop(columns=['transaction_time'])

    return output_df


def create_distance_features(inp_row: pd.Series) -> float:
    logger.debug('Calculating distance between client and merchant...')
    R = 6371 # радиус Земли в км
    distance = 0
    
    # рассчитываем разность между координатами покупателя и продавца
    dlat = math.radians(inp_row['lat'] - inp_row['merchant_lat'])
    dlon = math.radians(inp_row['lon'] - inp_row['merchant_lon'])

    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(inp_row['lat'])) \
        * math.cos(math.radians(inp_row['merchant_lat'])) * math.sin(dlon / 2) ** 2

    # рассчитываем расстояние по формуле гаверсинуса
    distance = 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return distance


def transform_num_features(input_df: pd.DataFrame) -> pd.DataFrame:
    logger.debug('Transforming numerical features...')
    output_df = input_df.copy()

    output_df['amount'] = np.log1p(output_df['amount'])
    output_df['population_city'] = np.log1p(output_df['population_city'])

    return output_df


def preprocess_data(test_df: pd.DataFrame, fitted_encoder) -> pd.DataFrame:
    logger.info('Preprocessing test data...')
    test_df['gender'] = test_df['gender'].map({"F": 1, 'M': 0})
    test_df = extract_time_features(test_df)
    
    num_cols = [col for col in test_df.columns if col not in CAT_COLS]
    test_df[num_cols] = test_df[num_cols].astype('float')

    test_df['distance'] = test_df.apply(create_distance_features, axis=1)

    cat_cols = test_df.select_dtypes(include=['object']).columns.tolist()
    test_df[cat_cols] = fitted_encoder.transform(test_df[cat_cols])

    test_df = transform_num_features(test_df)

    return test_df


# Main preprocessing function
def run_preproc(input_df):
    logger.info('Running preprocessing...')

    # Handle missing values
    imputer = SimpleImputer(strategy='most_frequent')
    input_df = pd.DataFrame(imputer.fit_transform(input_df), columns=input_df.columns)

    logger.info('Missing values handled. Proceeding to preprocessing...')

    # Load the fitted encoder (assuming it's saved as a joblib file)
    logger.info('Loading fitted encoder...')
    fitted_encoder = joblib.load('./models/catboost_encoder.joblib')
    logger.info('Fitted encoder loaded successfully.')

    # Preprocess the data
    logger.info('Preprocessing data...')
    preprocessed_df = preprocess_data(input_df, fitted_encoder=fitted_encoder)
    logger.info('Data preprocessing completed successfully.')

    return preprocessed_df