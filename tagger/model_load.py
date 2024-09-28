from sentence_transformers import SentenceTransformer

model_checkpoint = "nizamovtimur/multilingual-e5-large-videotags"
save_path = "saved_models/multilingual-e5-large-videotags"

model = SentenceTransformer(model_checkpoint)
model.save(save_path)
