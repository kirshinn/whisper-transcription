from fastapi import FastAPI, UploadFile, HTTPException
import os
import whisper
import mysql.connector
from mysql.connector import Error

# Конфигурация базы данных
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", "example"),
    "database": os.getenv("DB_NAME", "transcriptions"),
}

app = FastAPI()

# Загрузка модели Whisper
model = whisper.load_model("base")

# Подключение к базе данных
def get_db_connection():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        print(f"Ошибка подключения к базе данных: {e}")
        return None

@app.post("/transcribe/")
async def transcribe(file: UploadFile):
    if not file.filename.endswith((".mp3", ".wav", ".m4a")):
        raise HTTPException(status_code=400, detail="Неверный формат файла")

    # Сохраняем файл временно
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Обновляем статус задачи в базе данных
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Ошибка подключения к базе данных")
    cursor = connection.cursor()

    try:
        cursor.execute("INSERT INTO tasks (file_path, status) VALUES (%s, %s)", (file_path, "processing"))
        task_id = cursor.lastrowid
        connection.commit()

        # Обрабатываем файл
        result = model.transcribe(file_path)
        transcription = result["text"]

        # Сохраняем результат в базе
        cursor.execute("UPDATE tasks SET status = %s, result = %s WHERE id = %s", ("done", transcription, task_id))
        connection.commit()
    except Error as e:
        raise HTTPException(status_code=500, detail=f"Ошибка базы данных: {e}")
    finally:
        cursor.close()
        connection.close()
        os.remove(file_path)

    return {"task_id": task_id, "transcription": transcription}
