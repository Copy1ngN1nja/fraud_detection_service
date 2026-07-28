FROM python:3.12-slim

WORKDIR /app

# Установка зависимостей для lightgbm
# Создание директории для логов
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app/logs \
    && touch /app/logs/service.log \
    && chmod -R 777 /app/logs # права на запись для всех пользователей

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Эксаонируем порт для Prometheus
EXPOSE 8000

CMD ["python", "./app/app.py"]