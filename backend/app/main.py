import os
import logging
import whisper
import mysql.connector
from functools import lru_cache
from fastapi import FastAPI, UploadFile, HTTPException
from mysql.connector import Error
from concurrent.futures import ThreadPoolExecutor
import threading
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@lru_cache(maxsize=1)
def load_whisper_model(model_name="large"):
    """Кэширование загрузки модели"""
    local_model_path = "/models/whisper"
    model_file = os.path.join(local_model_path, f"{model_name}-v3.pt")
    try:
        # Проверка наличия модели
        if not os.path.exists(model_file):
            logger.info(f"Model {model_name} not found locally, downloading...")
        else:
            logger.info(f"Model {model_name} found locally at {model_file}.")

        # Загрузка модели Whisper из указанного пути
        model = whisper.load_model(model_name, download_root=local_model_path)
        logger.info(f"Whisper model {model_name} loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Загрузка модели Whisper из локального хранилища
try:
    model = load_whisper_model()
except Exception as e:
    logger.error(f"Model loading error: {e}")
    raise HTTPException(status_code=500, detail=str(e))

# Конфигурация базы данных
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", "example"),
    "database": os.getenv("DB_NAME", "transcriptions"),
}

# Подключение к базе данных
def get_db_connection():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        logger.error(f"Ошибка подключения к базе данных: {e}")
        return None

# Фоновый обработчик задач
executor = ThreadPoolExecutor(max_workers=4)
lock = threading.Lock()

def process_task():
    while True:
        connection = get_db_connection()
        if not connection:
            time.sleep(5)  # Подождать, если нет подключения
            continue

        cursor = connection.cursor(dictionary=True)
        try:
            # Выбрать задачу в статусе "queued"
            with lock:
                cursor.execute("SELECT * FROM tasks WHERE status = 'queued' LIMIT 1 FOR UPDATE")
                task = cursor.fetchone()

            if task:
                logger.info(f"Processing task {task['id']}")
                # Обновить статус задачи на "processing"
                cursor.execute("UPDATE tasks SET status = 'processing' WHERE id = %s", (task['id'],))
                connection.commit()

                # Обработка файла
                result = model.transcribe(task['file_path'])
                transcription = result["text"]

                # Обновить результат в базе
                cursor.execute(
                    "UPDATE tasks SET status = %s, result = %s WHERE id = %s",
                    ("done", transcription, task["id"])
                )
                connection.commit()

                # Удалить временный файл
                os.remove(task["file_path"])
            else:
                time.sleep(2)  # Если задач нет, подождать
        except Exception as e:
            logger.error(f"Ошибка обработки задачи: {e}")
        finally:
            cursor.close()
            connection.close()

# Запуск фонового потока
executor.submit(process_task)

@app.post("/transcribe/")
async def transcribe(file: UploadFile):
    if not file.filename.endswith((".mp3", ".wav", ".m4a")):
        raise HTTPException(status_code=400, detail="Неверный формат файла")

    # Сохраняем файл временно
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Добавляем задачу в базу данных
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Ошибка подключения к базе данных")

    cursor = connection.cursor()
    try:
        cursor.execute("INSERT INTO tasks (file_path, status) VALUES (%s, %s)", (file_path, "queued"))
        task_id = cursor.lastrowid
        connection.commit()
    except Error as e:
        raise HTTPException(status_code=500, detail=f"Ошибка базы данных: {e}")
    finally:
        cursor.close()
        connection.close()

    return {"task_id": task_id, "status": "queued"}

@app.get("/task/{task_id}")
async def get_task_status(task_id: int):
    """Получение статуса задачи"""
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Ошибка подключения к базе данных")

    cursor = connection.cursor(dictionary=True)

    try:
        cursor.execute("SELECT * FROM tasks WHERE id = %s", (task_id,))
        task = cursor.fetchone()

        if not task:
            raise HTTPException(status_code=404, detail="Задача не найдена")

        return task

    finally:
        cursor.close()
        connection.close()
