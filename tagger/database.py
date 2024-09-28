import numpy as np
import pandas as pd
from pgvector.sqlalchemy import Vector
from sentence_transformers import SentenceTransformer
from sqlalchemy import Text, create_engine, select, text, Engine
from sqlalchemy.orm import Mapped, declarative_base, mapped_column, Session

from config import app

Base = declarative_base()


class Tag(Base):
    __tablename__ = "tags"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(Text(), unique=True)
    embedding: Mapped[Vector] = mapped_column(Vector(1024))


def create_taxonomy(
    engine: Engine, encoder_model: SentenceTransformer, taxonomy_path: str
):
    taxonomy = pd.read_csv(taxonomy_path)
    with Session(engine) as session:
        session.query(Tag).delete()
        tags_concate = []
        for i, row in taxonomy.iterrows():
            if row["Уровень 3 (iab)"] is np.NaN:
                if row["Уровень 2 (iab)"] is np.NaN:
                    tags_concate.append(row["Уровень 1 (iab)"].strip())
                else:
                    tags_concate.append(f"{row["Уровень 1 (iab)"].strip()}: {row["Уровень 2 (iab)"].strip()}")
            else:
                tags_concate.append(f"{row["Уровень 1 (iab)"].strip()}: {row["Уровень 2 (iab)"].strip()}: {row["Уровень 3 (iab)"].strip()}")
        for tag_title in tags_concate:
            session.add(
                Tag(
                    title=tag_title,
                    embedding=encoder_model.encode(tag_title),
                )
            )
        session.commit()


def get_tags(
    engine: Engine,
    encoder_model: SentenceTransformer,
    title: str,
    description: str,
    video_path: str,
) -> str:
    with Session(engine) as session:
        tags = session.scalars(
            select(Tag.title)
            .order_by(Tag.embedding.cosine_distance(encoder_model.encode(title + " " + description)))
            .limit(1)
        ).all()
        return "['" + "', '".join(tags) + "']"


if __name__ == "__main__":
    try:
        engine = create_engine(app.config["SQLALCHEMY_DATABASE_URI"])
        with Session(engine) as session:
            session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            session.commit()
        Base.metadata.create_all(engine)
    except Exception as e:
        print(e)
