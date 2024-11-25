FROM python:3.10-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y ffmpeg libsndfile1 && apt-get clean

# Установка зависимостей Python
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем приложение
COPY app/ .

# Запуск приложения
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
