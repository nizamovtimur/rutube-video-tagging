import cv2
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def analyze_video(video_path: str, model, feature_extractor, tokenizer, device, gen_kwargs, max_workers=5):
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

    images_batch = []
    futures = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame)

            if frame_count % frame_step == 1:
                images_batch.append(pil_image)

            if len(images_batch) == max_workers:
                futures.append(executor.submit(predict_step, images_batch, model, feature_extractor, tokenizer, device, gen_kwargs))
                images_batch = []

            frame_count += 1

        # Обработка оставшихся изображений
        if images_batch:
            futures.append(executor.submit(predict_step, images_batch, model, feature_extractor, tokenizer, device, gen_kwargs))

        for future in as_completed(futures):
            descriptions.extend(future.result())

    cap.release()
    return descriptions
