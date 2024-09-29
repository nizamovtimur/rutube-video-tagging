import pathlib
from fastapi import FastAPI, HTTPException, UploadFile
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine
import torch
from config import Config
from database import create_taxonomy, get_tags
from video2text import load_model_and_processors


app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"
encoder_model = SentenceTransformer(
    "saved_models/multilingual-e5-large-videotags", device=device
)
# TODO: move downloading to build stage
# audio_model = WhisperModel("small", compute_type="int8", device=device)
video_model, video_feature_extractor, video_tokenizer, video_device = (
    load_model_and_processors(device=device)
)
engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)
create_taxonomy(
    engine=engine, encoder_model=encoder_model, taxonomy_path="IAB_tags.csv"
)


@app.post("/predict_tokens")
async def predict_tokens(title: str, description: str, video: UploadFile):
    """Возвращает список тегов для заданного видео, описания и названия.

    Args:
        title (str): название видео;
        description (str): описание видео;
        video (UploadFile): видеофайл.

    Raises:
        HTTPException: возвращает код 400 при неверном формате видеофайла.

    Returns:
        list[str]: список тегов.
    """
    if video.content_type != "video/mp4":
        raise HTTPException(400, detail="Invalid video type")

    pathlib.Path("data").mkdir(parents=True, exist_ok=True)
    with open(f"data/{video.filename}", "wb") as local_file:
        local_file.write(video.file.read())

    return get_tags(
        engine=engine,
        encoder_model=encoder_model,
        # audio_model=audio_model,
        video_model=video_model,
        video_feature_extractor=video_feature_extractor,
        video_tokenizer=video_tokenizer,
        video_device=video_device,
        title=title,
        description=description,
        video_path=f"data/{video.filename}",
    )
