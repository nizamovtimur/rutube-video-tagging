import time
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm.autonotebook import tqdm
import numpy as np
import torch
from flask import request
import os
from config import app
from pgvector.sqlalchemy import Vector
from sqlalchemy import Text, create_engine, select, text, Engine
from sqlalchemy.orm import Mapped, declarative_base, mapped_column, Session

taxonomy = pd.read_csv("IAB_tags.csv")
device = "cuda" if torch.cuda.is_available() else "cpu"
encoder_model = SentenceTransformer(
    "saved_models/multilingual-e5-large-wikiutmn", device=device
)
Base = declarative_base()


class Tag(Base):
    __tablename__ = "tags"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(Text(), unique=True)
    embedding: Mapped[Vector] = mapped_column(Vector(1024))


try:
    engine = create_engine(app.config["SQLALCHEMY_DATABASE_URI"])
    with Session(engine) as session:
        session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        session.commit()
    Base.metadata.create_all(engine)
except Exception as e:
    print(e)
    time.sleep(2)


def title_tag(
    encoder_model: SentenceTransformer, taxonomy: pd.DataFrame, title_text: str
):
    session = Session(engine)

    def get_tags():
        tags = {}
        for i, row in tqdm(taxonomy.iterrows()):
            if isinstance(row["Уровень 1 (iab)"], str):
                tag_vector = encoder_model.encode(row["Уровень 1 (iab)"])
                tags[row["Уровень 1 (iab)"]] = tag_vector
            if isinstance(row["Уровень 2 (iab)"], str):
                tag_vector = encoder_model.encode(
                    row["Уровень 1 (iab)"] + ": " + row["Уровень 2 (iab)"],
                )
                tags[row["Уровень 1 (iab)"] + ": " + row["Уровень 2 (iab)"]] = (
                    tag_vector
                )
            if isinstance(row["Уровень 3 (iab)"], str):
                tag_vector = encoder_model.encode(
                    row["Уровень 1 (iab)"]
                    + ": "
                    + row["Уровень 2 (iab)"]
                    + ": "
                    + row["Уровень 3 (iab)"],
                )
                tags[
                    row["Уровень 1 (iab)"]
                    + ": "
                    + row["Уровень 2 (iab)"]
                    + ": "
                    + row["Уровень 3 (iab)"]
                ] = tag_vector
        return tags

    tags = get_tags()

    for tag, vector in tags.items():
        new_tag = Tag(title=tag, embedding=vector)
        session.add(new_tag)
    session.commit()

    def tag_ranger(engine: Engine, encoder_model: SentenceTransformer, title: str):
        with Session(engine) as session:
            embedding = encoder_model.encode(title)
            return session.scalars(
                select(Tag).order_by(Tag.embedding.cosine_distance(embedding)).limit(3)
            )

    query = tag_ranger(engine, encoder_model, title_text)

    return query


sentences = [
    "Роман Юнусов и популярный озвучер Карен Арутюнов попали в клуб богачей В новом выпуске шоу «Спортивный Интерес» Рома Юнусов и Карен Арутюнов почувствуют себя богатеями или даже мафиози. А всё потому, что им предстоит освоить вид спорта, куда без членства в клубе не попасть. Да, мы отправили героев на поле для гольфа. Солнце, трава, песок, клюшка - вот неполный список того, что мешало ребятам бить по мячу. Ну, а кто выполнил драйв и прошёл лунку быстрее оппонента, ты узнаешь, посмотрев выпуск до конца.",
    "Массовая культура, Спорт",
    "Транспорт, Спорт: Автогонки, События и достопримечательности: Спортивные события",
]

tags = []
for i in range(len(sentences)):
    tags.append(title_tag(encoder_model, taxonomy, sentences[i]))

print(tags)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSION"]
    )


@app.post("/upload")
def main():
    # print(request.files)
    if "file" not in request.files:
        return "No file part"
    file = request.files["file"]
    title = str(request.form.get("title"))
    description = request.form.get("description")
    tag = title_tag(title)
    if file.filename == "":
        return "There is no file"
    if not file or not allowed_file(file.filename):
        return "Invalid file type"
    filename = file.filename
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)
    return tag


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
