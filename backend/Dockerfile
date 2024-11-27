FROM python:3.10-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    git \
    mc \
    nano \
    ffmpeg \
    libsndfile1 \
    && apt-get clean

WORKDIR /app

# Копируем requirements 
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY app/ .

# Создаем директорию для моделей
RUN mkdir -p /models/whisper

# Проверяем, существует ли модель, и загружаем её только при необходимости
RUN python -c "import os; import whisper; \
if not os.path.exists('/models/whisper/large-v3.pt'): \
    whisper.load_model('large')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
