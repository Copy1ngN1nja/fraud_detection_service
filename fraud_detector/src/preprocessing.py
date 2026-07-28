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

# Загружается один раз при импорте модуля, а не на каждый вызов run_preproc
logger.info('Loading fitted encoder...')
FITTED_ENCODER = joblib.load('./models/catboost_encoder.joblib')
logger.info('Fitted encoder loaded successfully.')


def _build_fast_category_encoding(encoder):
    """Разворачивает CatBoostEncoder.mapping (sum/count по категориям) в плоские
    словари {категория: закодированное_значение}, один раз при старте.

    CatBoostEncoder.transform на вход без ``y`` считает для каждой строки
    ``(sum + mean*a) / (count + a)`` (или просто ``mean``, если категория
    встретилась в трейне только один раз), пересчитывая это по ВСЕМУ словарю
    категорий (до тысячи на колонку, см. CAT_COLS) при каждом вызове -
    так как нам всегда нужно только одно значение (для одной транзакции),
    выгоднее посчитать словарь один раз и потом просто делать O(1) lookup.
    """
    a = encoder.a
    mean = float(encoder._mean)
    cat_maps, nan_values = {}, {}

    for col, colmap in encoder.mapping.items():
        counts = colmap['count'].to_numpy()
        sums = colmap['sum'].to_numpy()
        encoded = np.where(counts > 1, (sums + mean * a) / (counts + a), mean)

        col_map, nan_value = {}, mean
        for category, value in zip(colmap.index, encoded):
            if pd.isna(category):
                nan_value = float(value)
            else:
                col_map[category] = float(value)

        cat_maps[col] = col_map
        nan_values[col] = nan_value

    return cat_maps, nan_values, mean


CAT_ENCODING_MAPS, CAT_ENCODING_NAN_VALUES, CAT_ENCODING_MEAN = _build_fast_category_encoding(FITTED_ENCODER)


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _encode_category(col: str, value):
    if _is_missing(value):
        return CAT_ENCODING_NAN_VALUES[col]
    return CAT_ENCODING_MAPS[col].get(value, CAT_ENCODING_MEAN)

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


def preprocess_data(test_df: pd.DataFrame, fitted_encoder=FITTED_ENCODER) -> pd.DataFrame:
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

    # Preprocess the data (uses the encoder loaded once at module import)
    logger.info('Preprocessing data...')
    preprocessed_df = preprocess_data(input_df)
    logger.info('Data preprocessing completed successfully.')

    return preprocessed_df


def run_preproc_row(raw_data: dict) -> pd.DataFrame:
    """Эквивалент run_preproc(pd.DataFrame([raw_data])) для одной транзакции
    из Kafka, но без пословных pandas-операций (SimpleImputer.fit_transform
    на выборке из одной строки, .apply(axis=1), .astype на весь DataFrame)
    и без пересчёта кодировки CatBoostEncoder по всему словарю категорий -
    вместо этого работаем с обычным dict и float, а категории кодируем через
    словари, посчитанные один раз при импорте модуля (см. выше).

    Импутация пропусков не переносится: SimpleImputer, обучаемый на одной
    строке, для непустых полей ничего не менял, а для пропущенных - либо
    не менял тоже, либо (если значение было NaN, а не None) вовсе ронял
    колонку, из-за несовпадения числа колонок с исходным DataFrame.
    """
    row = dict(raw_data)

    gender = row.get('gender')
    row['gender'] = 1.0 if gender == 'F' else (0.0 if gender == 'M' else float('nan'))

    timestamp = pd.Timestamp(row.pop('transaction_time'))
    row['year'] = float(timestamp.year)
    row['month'] = float(timestamp.month)
    row['day'] = float(timestamp.day)
    row['hour'] = float(timestamp.hour)
    row['minute'] = float(timestamp.minute)

    row['distance'] = create_distance_features(row)

    for col in CAT_COLS:
        row[col] = _encode_category(col, row.get(col))

    for key, value in row.items():
        if key not in CAT_COLS:
            row[key] = float(value)

    row['amount'] = math.log1p(row['amount'])
    row['population_city'] = math.log1p(row['population_city'])

    return pd.DataFrame([row])