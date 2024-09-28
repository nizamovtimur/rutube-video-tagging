from os import environ

from dotenv import load_dotenv

load_dotenv("../.env")


class Config:
    SQLALCHEMY_DATABASE_URI = f"postgresql://{environ.get('POSTGRES_USER')}:{environ.get('POSTGRES_PASSWORD')}@{environ.get('POSTGRES_HOST')}/{environ.get('POSTGRES_DB')}"
    SECRET_KEY = ("SECRET_KEY") or "secret"
