import pandas as pd
from pgvector.sqlalchemy import Vector
from sentence_transformers import SentenceTransformer
from sqlalchemy import Text, create_engine, select, text, Engine
from sqlalchemy.orm import Mapped, declarative_base, mapped_column, Session

from config import Config

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
        if len(session.query(Tag).all()) > 0:
            return
        tags_concate = []
        for i, row in taxonomy.iterrows():
            if not isinstance(row["Уровень 3 (iab)"], str):
                if not isinstance(row["Уровень 2 (iab)"], str):
                    if not isinstance(row["Уровень 1 (iab)"], str):
                        continue
                    tags_concate.append(row["Уровень 1 (iab)"].strip())
                else:
                    tags_concate.append(
                        f"{row['Уровень 1 (iab)'].strip()}: {row['Уровень 2 (iab)'].strip()}"
                    )
            else:
                tags_concate.append(
                    f"{row['Уровень 1 (iab)'].strip()}: {row['Уровень 2 (iab)'].strip()}: {row['Уровень 3 (iab)'].strip()}"
                )
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
    video_path: str | None = None,
):
    with Session(engine) as session:
        embedding = encoder_model.encode(title + " " + description)
        tags = session.scalars(
            select(Tag.title)
            .order_by(Tag.embedding.cosine_distance(embedding))
            .limit(3)
        ).all()
        return tags


if __name__ == "__main__":
    try:
        engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)
        with Session(engine) as session:
            session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            session.commit()
        Base.metadata.create_all(engine)
    except Exception as e:
        print(e)
