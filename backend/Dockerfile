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
model_path = '/models/whisper/large-v3.pt'; \
print(f'Checking for model at {model_path}...'); \
if not os.path.exists(model_path): \
    print('Model not found, downloading...'); \
    whisper.load_model('large', download_root='/models/whisper'); \
else: \
    print('Model already exists, skipping download.')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
