import json
import os
import sys
import time
import uuid

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import psycopg2
import streamlit as st
from kafka import KafkaProducer
from sqlalchemy import create_engine

# Переиспользуем логику препроцессинга из fraud_detector/src/preprocessing.py
# (копия лежит в ./src, т.к. interface собирается отдельным Docker-контекстом)
sys.path.append(os.path.abspath('./src'))
from preprocessing import run_preproc


# Конфигурация Kafka
KAFKA_CONFIG = {
    "bootstrap_servers": os.getenv("KAFKA_BROKERS", "kafka:9092"),
    "topic": os.getenv("KAFKA_TOPIC", "transactions"),
}

HISTORICAL_TABLE = "historical_transactions"


def load_file(uploaded_file_):
    """Загрузка CSV файла в DataFrame."""
    try:
        return pd.read_csv(uploaded_file_)
    except Exception as e:
        st.error(f"Ошибка загрузки файла: {e!s}")
        return None


def send_to_kafka(df, topic, bootstrap_servers):
    """Отправка данных в Kafka с уникальным ID транзакции."""
    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            security_protocol="PLAINTEXT",
        )

        # Генерация уникальных ID для всех транзакций
        df['transaction_id'] = [str(uuid.uuid4()) for _ in range(len(df))]

        progress_bar = st.progress(0)
        total_rows = len(df)

        for idx, row in df.iterrows():
            # Отправляем данные вместе с ID
            producer.send(
                topic,
                value={
                    "transaction_id": row['transaction_id'],
                    "data": row.drop('transaction_id').to_dict(),
                },
            )
            progress_bar.progress((idx + 1) / total_rows)
            time.sleep(0.01)

        producer.flush()

        return True
    except Exception as e:
        st.error(f"Ошибка отправки данных: {e!s}")
        return False


# Функция подключения к БД
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        database=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )


def get_db_engine():
    url = (
        f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:"
        f"{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:"
        f"{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
    )
    return create_engine(url)


def upload_historical_data(raw_df: pd.DataFrame) -> int:
    """Препроцессит исторические транзакции по логике preprocessing.py и
    складывает их в Postgres. На этих данных feature_validator.py в сервисе
    fraud_detector обучает adversarial-модель (historical vs. current) и
    считает ROC-AUC/feature importance для дашборда в Grafana."""
    processed_df = run_preproc(raw_df)
    processed_df["uploaded_at"] = pd.Timestamp.now(tz="UTC")

    engine = get_db_engine()
    processed_df.to_sql(HISTORICAL_TABLE, engine, if_exists="append", index=False)
    return len(processed_df)


# Загрузка 10 последних фродовых транзакций
def load_fraud_transactions(limit=10):
    conn = get_db_connection()
    query = f"""
        SELECT * 
        FROM scores 
        WHERE fraud_flag = 1
        ORDER BY created_at DESC LIMIT {limit}
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df


# Загрузка 100 последних транзакций
def load_scores(limit=100):
    conn = get_db_connection()
    query = f"""
        SELECT score 
        FROM scores
        ORDER BY created_at DESC LIMIT {limit}
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df


# Инициализация состояния
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}

# Интерфейс
st.title("Отправка данных в Kafka")

# Блок загрузки файлов
uploaded_file = st.file_uploader(
    "Загрузите CSV файл с транзакциями",
    type=["csv"],
)

if uploaded_file and uploaded_file.name not in st.session_state.uploaded_files:
    # Добавляем файл в состояние
    st.session_state.uploaded_files[uploaded_file.name] = {
        "status": "Загружен",
        "df": load_file(uploaded_file),
    }
    st.success(f"Файл {uploaded_file.name} успешно загружен!")

# Список загруженных файлов
if st.session_state.uploaded_files:
    st.subheader("Список загруженных файлов")

    for file_name, file_data in st.session_state.uploaded_files.items():
        cols = st.columns([4, 2, 2])

        with cols[0]:
            st.markdown(f"**Файл:** `{file_name}`")
            st.markdown(f"**Статус:** `{file_data['status']}`")

        with cols[2]:
            if st.button(f"Отправить {file_name}", key=f"send_{file_name}"):
                if file_data["df"] is not None:
                    with st.spinner("Отправка..."):
                        success = send_to_kafka(
                            file_data["df"],
                            KAFKA_CONFIG["topic"],
                            KAFKA_CONFIG["bootstrap_servers"],
                        )
                        if success:
                            st.session_state.uploaded_files[file_name]["status"] = "Отправлен"
                            st.rerun()
                else:
                    st.error("Файл не содержит данных")

st.divider()
st.subheader("Исторические данные для adversarial-валидации")
st.caption(
    "Загрузите CSV с историческими транзакциями. Файл будет предобработан по той же "
    "логике, что и в fraud_detector/src/preprocessing.py, и сохранён в таблицу "
    f"`{HISTORICAL_TABLE}` в Postgres. На этих данных сервис fraud_detector (модуль "
    "feature_validator.py) каждые 100 новых транзакций обучает adversarial-модель "
    "(исторические vs. текущие данные) и считает ROC-AUC и топ-3 важных признака "
    "для дашборда в Grafana."
)

historical_file = st.file_uploader(
    "Загрузите исторический CSV файл",
    type=["csv"],
    key="historical_uploader",
)

if historical_file is not None and st.button("Загрузить исторические данные в БД"):
    hist_df = load_file(historical_file)
    if hist_df is not None:
        with st.spinner("Препроцессинг и запись в БД..."):
            try:
                n_rows = upload_historical_data(hist_df)
                st.success(f"Загружено {n_rows} исторических наблюдений в таблицу `{HISTORICAL_TABLE}`.")
            except Exception as e:
                st.error(f"Ошибка загрузки исторических данных: {e!s}")
    else:
        st.error("Файл не содержит данных")

if st.button("Посмотреть результаты"):
    st.subheader("Последние 10 фродовых транзакций:")
    fraud_df = load_fraud_transactions(limit=10)
    if not fraud_df.empty:
        st.dataframe(fraud_df[["transaction_id", "score", "fraud_flag", "created_at"]])
    else:
        st.write("Ура! Нет мошеннических транзакций!")

    st.subheader("Распределение вероятностей предсказанных моделью для последних транзакций:")
    score_df = load_scores(limit=100)
    if not score_df.empty:
        fig, ax = plt.subplots()

        sns.histplot(
            score_df['score'], 
            bins=100, 
            color='royalblue', 
            kde=True, 
            stat='density', 
            edgecolor='black',
            alpha=0.7, 
            ax=ax
        )
        
        ax.set_xlabel('Вероятность мошенничества')
        ax.set_ylabel('Плотность')
        ax.grid(ls=':', alpha=0.5)

        plt.suptitle('Распределение вероятностей мошенничества', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.write("Нет записей в базе для построения гистограммы")