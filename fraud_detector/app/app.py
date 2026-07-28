import json
import os
import sys

import logging

from confluent_kafka import Consumer
from confluent_kafka import Producer
from prometheus_client import start_http_server, Summary, Counter, Histogram, Gauge

sys.path.append(os.path.abspath('./src'))
from feature_validator import validator as adversarial_validator
from preprocessing import run_preproc_row
from scorer import make_pred


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set kafka configuration file
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TRANSACTIONS_TOPIC = os.getenv("KAFKA_TRANSACTIONS_TOPIC", "transactions")
SCORING_TOPIC = os.getenv("KAFKA_SCORING_TOPIC", "scoring")

# Определяем метрики
PROCESSING_TIME = Summary('transaction_processing_seconds', 'Время обработки транзакции')
TRANSACTION_COUNT = Counter('transactions_total', 'Общее количество обработанных транзакций')

# Создаем более детальную гистограмму для распределения скоров
# Используем линейные бакеты с шагом 0.02 от 0 до 1 (50 бакетов)
FRAUD_SCORE = Histogram('fraud_score', 'Распределение скоров мошенничества',
                       buckets=[i/50.0 for i in range(51)])  # [0.0, 0.02, 0.04, ..., 0.98, 1.0]

FRAUD_RATIO = Gauge('fraud_ratio', 'Соотношение мошеннических транзакций к общему числу')

class ProcessingService:
    def __init__(self):
        self.consumer_config = {
            'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
            'group.id': 'ml-scorer',
            'auto.offset.reset': 'earliest',
        }
        self.producer_config = {
            'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
        }
        self.consumer = Consumer(self.consumer_config)
        self.consumer.subscribe([TRANSACTIONS_TOPIC])
        self.producer = Producer(self.producer_config)

        # Счетчики для метрик
        self.total_transactions = 0
        self.fraud_transactions = 0
        
        # Запуск HTTP-сервера для Prometheus
        start_http_server(8000)
        logger.info("Prometheus метрики доступны на порту 8000")

        
    @PROCESSING_TIME.time()
    def process_message(self, msg):
        try:
            # Десериализация JSON
            data = json.loads(msg.value().decode('utf-8'))

            # Извлекаем ID и данные
            transaction_id = data['transaction_id']
            raw_data = data['data']

            # Препроцессинг кодирует us_state/merch/cat_id (CatBoostEncoder),
            # поэтому сырые значения для дашбордов Grafana запоминаем заранее
            raw_us_state = raw_data.get('us_state')
            raw_merch = raw_data.get('merch')
            raw_cat_id = raw_data.get('cat_id')

            # Препроцессинг и предсказание
            processed_df = run_preproc_row(raw_data)
            submission, y_proba, y_pred = make_pred(processed_df, "kafka_stream")

            # Обновляем метрики
            TRANSACTION_COUNT.inc()
            FRAUD_SCORE.observe(y_proba[0])

            self.total_transactions += 1
            self.fraud_transactions += y_pred[0]

            FRAUD_RATIO.set(self.fraud_transactions / self.total_transactions)

            # Копим наблюдение для adversarial-валидации (каждые 1000 штук
            # триггерит переобучение и запись ROC-AUC/feature importance/PSI в БД)
            adversarial_validator.observe(processed_df, y_proba)

            # Добавляем ID и сырые категориальные признаки (для фильтров в Grafana)
            submission['transaction_id'] = transaction_id
            submission['us_state'] = raw_us_state
            submission['merch'] = raw_merch
            submission['cat_id'] = raw_cat_id

            # Отправка результата в топик scoring
            self.producer.produce(
                'scoring',
                value=submission.to_json(orient='records'),
            )
            self.producer.flush()
            return True
        except Exception as e:
            logger.exception(f"Error processing message: {e}")
            return False

    def process_messages(self):
        while True:
            msg = self.consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                logger.error(f"Kafka error: {msg.error()}")
                continue
            
            self.process_message(msg)


if __name__ == "__main__":
    logger.info('Starting Kafka ML scoring service...')
    service = ProcessingService()
    try:
        service.process_messages()
    except KeyboardInterrupt:
        logger.info('Service stopped by user')