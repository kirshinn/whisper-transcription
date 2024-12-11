import os
import bcrypt
import logging
import multiprocessing
import time
import whisper
import mysql.connector
from functools import lru_cache
from fastapi import FastAPI, Depends, UploadFile, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from mysql.connector import Error
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация приложения
app = FastAPI()

# Basic Auth конфигурация
security = HTTPBasic()

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    """
        Проверяет учетные данные пользователя, хранящиеся в базе данных.
        Использует bcrypt для проверки хэшированного пароля.
    """
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Ошибка подключения к базе данных")
    cursor = connection.cursor(dictionary=True)
    try:
        # Найти пользователя по имени
        cursor.execute("SELECT password FROM users WHERE username = %s", (credentials.username,))
        user = cursor.fetchone()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Неверные учетные данные",
                headers={"WWW-Authenticate": "Basic"},
            )

        # Сравнение хэшированного пароля
        hashed_password = user["password"].encode("utf-8")
        if not bcrypt.checkpw(credentials.password.encode("utf-8"), hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Неверные учетные данные",
                headers={"WWW-Authenticate": "Basic"},
            )
        
        return credentials # Возвращаем credentials, если авторизация успешна

    finally:
        cursor.close()
        connection.close()


def add_user(username: str, password: str):
    """
        Добавляет нового пользователя в базу данных с хэшированным паролем.
    """
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Ошибка подключения к базе данных")
    cursor = connection.cursor()

    try:
        # Проверка, существует ли пользователь
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Пользователь с таким именем уже существует",
            )

        # Хэширование пароля
        hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

        # Добавление пользователя
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (%s, %s)",
            (username, hashed_password),
        )
        connection.commit()
        return {"message": "Пользователь успешно добавлен"}
    except Error as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка базы данных: {e}",
        )
    finally:
        cursor.close()
        connection.close()


def verify_admin(credentials: HTTPBasicCredentials):
    """
        Проверяет, является ли текущий пользователь администратором.
    """
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Ошибка подключения к базе данных")
    cursor = connection.cursor(dictionary=True)

    try:
        # Проверка наличия пользователя
        cursor.execute("SELECT * FROM users WHERE username = %s", (credentials.username,))
        user = cursor.fetchone()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Неверные учетные данные",
            )

        # Проверка пароля
        if not bcrypt.checkpw(credentials.password.encode("utf-8"), user["password"].encode("utf-8")):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Неверные учетные данные",
            )

        # Проверка роли
        if user["role"] != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Доступ запрещен: требуется роль администратора",
            )
    finally:
        cursor.close()
        connection.close()


def clean_transcription(text: str) -> str:
    return text


def load_whisper_model(model_name="large"):
    """
        Загрузка модели.
    """
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

# Определение количества ядер CPU
CPU_COUNT = (multiprocessing.cpu_count() // 2)

# Конфигурация пула потоков с количеством воркеров
executor = ThreadPoolExecutor(max_workers=max(CPU_COUNT, 2)) # Ограничим задачи 2 потоками, если не определено значение CPU_COUNT

# Запуск задачи транскрибации аудио
def process_task():
    while True:
        try:
            # Загрузка модели Whisper из локального хранилища
            try:
                model = load_whisper_model()
            except Exception as e:
                logger.error(f"Model loading error: {e}")

            connection = get_db_connection()
            if not connection:
                time.sleep(5)
                continue

            cursor = connection.cursor(dictionary=True)
            
            try:
                cursor.execute(
                    "SELECT * FROM tasks WHERE status = 'queued' LIMIT 1 FOR UPDATE"
                )
                task = cursor.fetchone()

                if not task:
                    time.sleep(2)
                    continue

                # Проверяем, пустой ли файл
                if os.path.getsize(task["file_path"]) == 0:
                    logger.error(f"Пустой или некорректный файл: {task['file_path']}")
                    cursor.execute(
                        "UPDATE tasks SET status = 'error', result = 'Empty or invalid file' WHERE id = %s",
                        (task['id'],)
                    )
                    connection.commit()
                    continue  # Пропустить обработку этого задания

                # Логика обработки задачи
                logger.info(f"Processing task {task['id']}")
                cursor.execute(
                    "UPDATE tasks SET status = 'processing' WHERE id = %s", 
                    (task['id'],)
                )
                connection.commit()

                # Получаем значение temperature
                temperature = task.get('temperature', 0.2)

                # Получаем значение prompt
                prompt = task.get('prompts', None)
                
                logger.info(prompt)

                # Основная обработка
                result = model.transcribe(
                    task['file_path'], 
                    temperature=temperature, 
                    language="ru", 
                    initial_prompt=prompt
                )
                transcription = result["text"]

                # Очистка текста, обработка пунктуации и прочее
                if task['is_spelling']:
                    transcription = clean_transcription(transcription)

                # Обновление результата
                cursor.execute(
                    "UPDATE tasks SET status = %s, result = %s WHERE id = %s",
                    ("done", transcription, task["id"])
                )
                connection.commit()

                # Удаление временного файла
                os.remove(task["file_path"])

            except Exception as e:
                logger.error(f"Ошибка обработки задачи: {e}")
                # Обработка ошибок
                if task:
                    cursor.execute(
                        "UPDATE tasks SET status = 'error' WHERE id = %s", 
                        (task['id'],)
                    )
                    connection.commit()
            
            finally:
                cursor.close()
                connection.close()

        except Exception as global_error:
            logger.error(f"Глобальная ошибка: {global_error}")
            time.sleep(5)

# Динамический запуск потоков
def start_task_processors():
    for _ in range(max(CPU_COUNT, 2)):
        executor.submit(process_task)

# Вызов при старте приложения
start_task_processors()

# Обработчики

@app.get("/healthcheck")
async def root(credentials: HTTPBasicCredentials = Depends(authenticate)):
    """
        Пример защищенного маршрута.
    """
    return {"message": "Вы авторизованы и видите этот контент."}


@app.post("/transcribe")
async def transcribe(file: UploadFile, prompt: str = None, temperature: float = 0.2, is_spelling: bool = False, credentials: HTTPBasicCredentials = Depends(authenticate)):
    """
        Транскрибация аудио в текст.
    """
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
        cursor.execute("INSERT INTO tasks (file_path, status, temperature, prompts, is_spelling) VALUES (%s, %s, %s, %s, %s)", (file_path, "queued", temperature, prompt, int(is_spelling)))
        task_id = cursor.lastrowid
        connection.commit()
    except Error as e:
        raise HTTPException(status_code=500, detail=f"Ошибка базы данных: {e}")
    finally:
        cursor.close()
        connection.close()

    return {"task_id": task_id, "status": "queued"}


@app.get("/task/{task_id}")
async def get_task_status(task_id: int, credentials: HTTPBasicCredentials = Depends(authenticate)):
    """
        Получение статуса задачи.
    """
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


@app.post("/user/add")
async def add_new_user(username: str, password: str, credentials: HTTPBasicCredentials = Depends(authenticate)):
    verify_admin(credentials)
    return add_user(username=username, password=password)
