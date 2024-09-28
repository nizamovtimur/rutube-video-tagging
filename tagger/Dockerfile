FROM python:3.10.11-slim-buster

# Устанавливаем рабочую директорию
WORKDIR /usr/src/app

# Отключаем запись файлов байт-кода Python и включаем вывод без буферизации
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Обновляем pip и устанавливаем зависимости
RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# Запускаем скрипт для загрузки модели
COPY ./model_load.py .
RUN python model_load.py

# Копируем весь проект в контейнер
COPY . /usr/src/app/

# Создаем папку для загрузки файлов
RUN mkdir -p /usr/src/app/uploads

# Выставляем права на папку (необязательно, если нужны специфические права)
RUN chmod 755 /usr/src/app/uploads
