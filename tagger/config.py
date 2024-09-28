import os
from dotenv import load_dotenv
from flask import Flask

load_dotenv("../.env")

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

app.config["SQLALCHEMY_DATABASE_URI"] = (
    f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}/{os.getenv('POSTGRES_DB')}"
)

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
