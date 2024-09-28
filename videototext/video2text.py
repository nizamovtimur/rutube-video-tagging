import cv2
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import numpy as np

def calculate_frame_step(x, a=6.64, b=0.01, c=130):
    """Определить шаг между кадрами в зависимости от длины видео."""
    return int(a * np.log(b * (x + c)))

def load_model_and_processors():
    """Инициализировать модель."""
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, feature_extractor, tokenizer, device

def predict_step(image, model, feature_extractor, tokenizer, device, gen_kwargs):
    """Сгенерировать описание для данного изображения с помощью модели."""
    if image.mode != "RGB":
        image = image.convert(mode="RGB")

    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    return preds[0].strip()

def analyze_video(video_path, model, feature_extractor, tokenizer, device, gen_kwargs):
    """Анализировать видео и генерировать описания для выбранных кадров."""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    descriptions = []
    frame_step = 1

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
    frame_step = calculate_frame_step(video_length)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame)

        if frame_count % frame_step == 1:
            description = predict_step(pil_image, model, feature_extractor, tokenizer, device, gen_kwargs)
            descriptions.append(description)

        frame_count += 1

    cap.release()
    return descriptions

def main(video_file_path):
    """Загрузить модель и проанализировать заданный видеофайл для генерации описаний."""
    model, feature_extractor, tokenizer, device = load_model_and_processors()

    # Параметры генерации
    max_length = 30
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    # Возвращаеммый набор описаний кадров
    video_descriptions = analyze_video(video_file_path, model, feature_extractor, tokenizer, device, gen_kwargs)

    for desc in video_descriptions:
        print(desc)

if __name__ == "__main__":
    video_file_path = '/content/0a7a288165c6051ebd74010be4dc9aa8.mp4'
    main(video_file_path)
