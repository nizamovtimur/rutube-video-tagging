import os
from flask import request
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine
import torch

from config import app
from database import create_taxonomy, get_tags


device = "cuda" if torch.cuda.is_available() else "cpu"
encoder_model = SentenceTransformer(
    "saved_models/multilingual-e5-large-videotags", device=device
)
engine = create_engine(app.config["SQLALCHEMY_DATABASE_URI"])


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSION"]
    )


@app.post("/upload")
def main():
    # print(request.files)
    # if "file" not in request.files:
    #     return "No file part"
    # file = request.files["file"]
    title = str(request.form.get("title"))
    description = str(request.form.get("description"))
    tag = get_tags(
        engine=engine,
        encoder_model=encoder_model,
        title=title,
        description=description,
        video_path="NULL",
    )
    # if file.filename == "":
    #     return "There is no file"
    # if not file or not allowed_file(file.filename):
    #     return "Invalid file type"
    # filename = file.filename
    # file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    # file.save(file_path)
    return tag


if __name__ == "__main__":
    create_taxonomy(
        engine=engine, encoder_model=encoder_model, taxonomy_path="IAB_tags.csv"
    )
    app.run(host="0.0.0.0", port=8080, debug=True)
