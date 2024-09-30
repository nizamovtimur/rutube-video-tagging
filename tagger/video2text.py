import os
from cv2 import (
    resize,
    VideoCapture,
    CAP_PROP_POS_FRAMES,
    CAP_PROP_FRAME_COUNT,
    CAP_PROP_FPS,
    cvtColor,
    COLOR_BGR2RGB
)
from torch import device as Device
from torch import no_grad
from torch.cuda import is_available
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
)
from PIL import Image
from numpy import log
from concurrent.futures import ThreadPoolExecutor, as_completed

def calculate_frame_step(
        x: int,
        a: float=6.64,
        b: float=0.01,
        c: float=130
    ) -> int:
    """Определяет шаг между кадрами в зависимости от длины видео.

    Args:
        x (int): длина видео в секундах;
        alpha (float, optional): коэффициент масштабирования. По умолчанию 6.64;
        beta (float, optional): коэффициент, влияющий на логарифмическое изменение. По умолчанию 0.01;
        offset (float, optional): смещение, добавляемое к длине видео. По умолчанию 130.

    Returns:
        int: шаг между кадрами
    """
    return int(a * log(b * (x + c)))


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
    device = Device(device)
    model.to(device)
    return model, feature_extractor, tokenizer, device


def predict_step(images, model, feature_extractor, tokenizer, device, gen_kwargs):
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

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    with no_grad():
        output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return preds


def analyze_video(video_path: str, model, feature_extractor, tokenizer, device, gen_kwargs, max_workers=4):
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
    cap = VideoCapture(video_path)
    frame_count = 0
    descriptions = []
    frame_step = 1

    fps = cap.get(CAP_PROP_FPS)
    video_length = cap.get(CAP_PROP_FRAME_COUNT)
    frame_step = int(fps * calculate_frame_step(int(video_length / fps)))

    images_batch = []
    futures = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while frame_count < video_length:
            # Устанавливаем позицию кадра
            cap.set(CAP_PROP_POS_FRAMES, frame_count)

            ret, frame = cap.read()
            frame = cvtColor(frame, COLOR_BGR2RGB)
            resized_image = resize(frame, (360, 360))
            pil_image = Image.fromarray(resized_image)
            images_batch.append(pil_image)

            # Если достигли максимального размера батча, отправляем на обработку
            if len(images_batch) == max_workers:
                futures.append(executor.submit(predict_step, images_batch, model, feature_extractor, tokenizer, device, gen_kwargs))
                images_batch = []

            frame_count += frame_step + 1  # Пропускаем к следующему нужному кадру

        # Обработка оставшихся изображений
        if images_batch:
            futures.append(executor.submit(predict_step, images_batch, model, feature_extractor, tokenizer, device, gen_kwargs))

        for future in as_completed(futures):
            descriptions.extend(future.result())

    cap.release()
    return descriptions
