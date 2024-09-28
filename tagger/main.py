import pathlib
from fastapi import FastAPI, HTTPException, UploadFile
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine
import torch
from config import Config
from database import create_taxonomy, get_tags


app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"
encoder_model = SentenceTransformer(
    "saved_models/multilingual-e5-large-videotags", device=device
)
engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)
create_taxonomy(
    engine=engine, encoder_model=encoder_model, taxonomy_path="IAB_tags.csv"
)


@app.post("/predict_tokens")
async def predict_tokens(title: str, description, video: UploadFile):

    if video.content_type != "video/mp4":
        raise HTTPException(400, detail="Invalid video type")

    pathlib.Path("data").mkdir(parents=True, exist_ok=True)
    with open(f"data/{video.filename}", "wb") as local_file:
        local_file.write(video.file.read())

    return get_tags(
        engine=engine,
        encoder_model=encoder_model,
        title=title,
        description=description,
        video_path=f"data/{video.filename}",
    )
