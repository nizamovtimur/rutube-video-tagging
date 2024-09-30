from faster_whisper.utils import download_model
from sentence_transformers import SentenceTransformer
from transformers import pipeline

st_model = SentenceTransformer("nizamovtimur/multilingual-e5-large-videotags")
st_model.save("saved_models/multilingual-e5-large-videotags")

i2t_model = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
i2t_model.save_pretrained("saved_models/vit-gpt2-image-captioning")

download_model("small", output_dir="saved_models/faster-whisper-small")
