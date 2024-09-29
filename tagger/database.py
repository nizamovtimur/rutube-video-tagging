from typing import List
from faster_whisper import WhisperModel
from multi_rake import Rake
import pandas as pd
from pgvector.sqlalchemy import Vector
from sentence_transformers import SentenceTransformer
from sqlalchemy import Text, create_engine, select, text, Engine
from sqlalchemy.orm import Mapped, declarative_base, mapped_column, Session

from config import Config
from video2text import analyze_video

Base = declarative_base()


class Tag(Base):
    __tablename__ = "tags"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(Text(), unique=True)
    embedding: Mapped[Vector] = mapped_column(Vector(1024))


def create_taxonomy(
    engine: Engine, encoder_model: SentenceTransformer, taxonomy_path: str
):
    """Загружает таксономию в модель данных.

    Args:
        engine (Engine): подключение к БД;
        encoder_model (SentenceTransformer): модель трансформер;
        taxonomy_path (str): путь к файлу таксономии.
    """
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


def get_key_tokens(text: str) -> str:
    """Выделяет ключевые слова из полученной строки

    Args:
        text (str): исходная строка

    Returns:
        str: строка ключевых слов
    """
    rake_model = Rake(max_words=10)
    return "; ".join(rake_model.apply(text)[:10])


def transcribe_and_save(model: WhisperModel, video_path: str) -> str:
    """Расшифровывает аудиофайлы и сохраняет текст в файлы.

    Args:
        model (WhisperModel): модель, обрабатывающая аудиопоток;
        video_path (str): путь к видео.

    Returns:
        str: полный текст распознанного аудиоряда.
    """
    segments_n, _ = model.transcribe(
        audio=video_path,
        word_timestamps=False,
        condition_on_previous_text=False,
        vad_filter=True,
    )
    return "".join(segment.text for segment in segments_n)


def remove_lower_tags(tags: List[str]) -> List[str]:
    if len(tags) < 3:
        return tags
    final_tags = []
    final_tags.append(tags[0])
    if tags[1] not in tags[0] and tags[0] not in tags[1]:
        final_tags.append(tags[1])
    if (
        tags[2] not in tags[0]
        and tags[0] not in tags[2]
        and tags[2] not in tags[1]
        and tags[1] not in tags[2]
    ):
        final_tags.append(tags[2])
    return final_tags


def get_tags(
    engine: Engine,
    encoder_model: SentenceTransformer,
    # audio_model: WhisperModel,
    video_model,
    video_feature_extractor,
    video_tokenizer,
    video_device,
    title: str,
    description: str,
    video_path: str,
) -> List[str]:
    # audio_transcribition = transcribe_and_save(audio_model, video_path)
    # audio_tokens = get_key_tokens(audio_transcribition)
    frames_descriptions = analyze_video(
        video_path,
        video_model,
        video_feature_extractor,
        video_tokenizer,
        video_device,
        {"max_length": 20, "num_beams": 4, "num_return_sequences": 2},
    )
    video_description = "; ".join(frames_descriptions)
    with Session(engine) as session:
        embedding = encoder_model.encode(
            title
            + " "
            + description
            + "\n\n"
            # + audio_tokens
            # + "\n\n"
            + video_description
        )
        tags = session.scalars(
            select(Tag.title)
            .order_by(Tag.embedding.cosine_distance(embedding))
            .limit(3)
        ).all()
        return remove_lower_tags(list(tags))


if __name__ == "__main__":
    try:
        engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)
        with Session(engine) as session:
            session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            session.commit()
        Base.metadata.create_all(engine)
    except Exception as e:
        print(e)
