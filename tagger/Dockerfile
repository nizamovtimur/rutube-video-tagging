FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

WORKDIR /usr/src/app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt

COPY ./model_load.py .
RUN python model_load.py

COPY . /usr/src/app/

RUN mkdir -p /usr/src/app/uploads

RUN chmod 755 /usr/src/app/uploads

CMD sh -s "python database.py && uvicorn main:app --host 0.0.0.0 --port 8000 --log-level warning"
