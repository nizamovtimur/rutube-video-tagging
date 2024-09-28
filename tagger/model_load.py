from sentence_transformers import SentenceTransformer

model_checkpoint = "cointegrated/rubert-tiny2"
save_path = "model_load/cointegrated/rubert-tiny2"

model = SentenceTransformer(model_checkpoint)
model.save(save_path)
