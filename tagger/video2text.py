import cv2
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import numpy as np


def calculate_frame_step(x, a=6.64, b=0.01, c=130):
    """Определяет шаг между кадрами в зависимости от длины видео.

    Args:
        x (int): длина видео в секундах;
        alpha (float, optional): коэффициент масштабирования. По умолчанию 6.64;
        beta (float, optional): коэффициент, влияющий на логарифмическое изменение. По умолчанию 0.01;
        offset (float, optional): смещение, добавляемое к длине видео. По умолчанию 130.

    Returns:
        int: шаг между кадрами
    """
    return int(a * np.log(b * (x + c)))


def load_model_and_processors(device: str):
    """Инициализирует модель для анализа видеоряда.

    Returns:
        VisionEncoderDecoderModel|ViTImageProcessor|GPT2TokenizerFast|Device: модели для анализа видео.
    """
    model = VisionEncoderDecoderModel.from_pretrained(
        "saved_models/vit-gpt2-image-captioning"
    )
    feature_extractor = ViTImageProcessor.from_pretrained(
        "saved_models/vit-gpt2-image-captioning"
    )
    tokenizer = AutoTokenizer.from_pretrained("saved_models/vit-gpt2-image-captioning")
    device = torch.device(device)
    model.to(device)
    return model, feature_extractor, tokenizer, device


def predict_step(image, model, feature_extractor, tokenizer, device, gen_kwargs):
    """Генерирует описание для текущего изображения при помощи модели.

    Args:
        image (Image.Image): изображение для анализа;
        model (VisionEncoderDecoderModel): модель создания покадровых описаний;
        feature_extractor (ViTImageProcessor): модель анализа изображений;
        tokenizer (GPT2TokenizerFast): модель, генерации аннотаций;
        device (Device): устройство, производящее расчеты;
        gen_kwargs (dict[str, int]): параметры, передаваемые в метод генерации.

    Returns:
        str: Текстовое описание текущего изображения
    """
    if image.mode != "RGB":
        image = image.convert(mode="RGB")

    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    return preds[0].strip()


def analyze_video(video_path, model, feature_extractor, tokenizer, device, gen_kwargs):
    """Анализирует видео путём генерирования описания для выбранных кадров

    Args:
        video_path (str): путь к файлу видео
        model (VisionEncoderDecoderModel): модель создания покадровых описаний;
        feature_extractor (ViTImageProcessor): модель анализа изображений;
        tokenizer (GPT2TokenizerFast): модель генерации аннотаций;
        device (Device): устройство, производящее расчеты;
        gen_kwargs (dict[str, int]): параметры, передаваемые в метод генерации.

    Returns:
        list[str]: Список описаний проанализированных кадров
    """
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

        if frame_count % frame_step == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_image = cv2.resize(frame, (360, 360))
            pil_image = Image.fromarray(resized_image)
            description = predict_step(
                pil_image, model, feature_extractor, tokenizer, device, gen_kwargs
            )
            descriptions.append(description)

        frame_count += 1

    cap.release()
    return descriptions
