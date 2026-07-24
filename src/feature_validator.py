import pandas as pd
import logging
import lightgbm as lgb
from scorer import BOOSTER

logger = logging.getLogger(__name__)

def make_feature_importance():
    logger.info('Calculating feature importance...')
    fi_df = pd.DataFrame({
        'feature': BOOSTER.feature_name(),
        'importance': BOOSTER.feature_importance(importance_type='gain')
    }).sort_values(by='importance', ascending=False)

    fi_df['importance_pct'] = fi_df['importance'] / fi_df['importance'].sum() * 100
    fi_df = fi_df.sort_values('importance_pct', ascending=False)
    fi_df = fi_df.head()
    fi_df = fi_df.drop(columns=['importance'], axis=1)

    logger.info('Feature importance calculated successfully.')
    return fi_df


def save_feature_importance(feature_importance, output_path):
    logger.info('Saving feature importance to: %s', output_path)
    feature_importance.to_json(output_path, orient='records')
    logger.info('Feature importance saved successfully.')


def get_feature_importance(output_path):
    feature_importance = make_feature_importance()
    save_feature_importance(feature_importance, output_path)