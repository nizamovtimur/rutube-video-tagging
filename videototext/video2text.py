import os
from cv2 import (
    VideoCapture,
    CAP_PROP_FRAME_COUNT,
    CAP_PROP_FPS,
    cvtColor,
    COLOR_BGR2RGB,
)
from torch import device as Device
from torch.cuda import is_available
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    GPT2TokenizerFast,
)
from PIL import Image
from numpy import log

MODEL_DIR = "./vit-gpt2-image-captioning"


def calculate_frame_step(
    x: int, a: float = 6.64, b: float = 0.01, c: float = 130
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


def load_model_and_processors() -> (
    VisionEncoderDecoderModel | ViTImageProcessor | GPT2TokenizerFast | Device
):
    """Инициализирует модель для анализа видеоряда.

    Returns:
        VisionEncoderDecoderModel|ViTImageProcessor|GPT2TokenizerFast|Device: модели для анализа видео.
    """
    if os.path.exists(MODEL_DIR):
        model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)
        feature_extractor = ViTImageProcessor.from_pretrained(MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    else:
        model = VisionEncoderDecoderModel.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )
        feature_extractor = ViTImageProcessor.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )

        model.save_pretrained(MODEL_DIR)
        feature_extractor.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)

    device = Device("cuda" if is_available() else "cpu")
    model.to(device)

    return model, feature_extractor, tokenizer, device


def predict_step(
    image: Image.Image,
    model: VisionEncoderDecoderModel,
    feature_extractor: ViTImageProcessor,
    tokenizer: GPT2TokenizerFast,
    device: Device,
    gen_kwargs: dict[str, int],
) -> str:
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


def analyze_video(
    video_path: str,
    model: VisionEncoderDecoderModel,
    feature_extractor: ViTImageProcessor,
    tokenizer: GPT2TokenizerFast,
    device: Device,
    gen_kwargs: dict[str, int],
) -> list[str]:
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

    video_length = int(cap.get(CAP_PROP_FRAME_COUNT) / cap.get(CAP_PROP_FPS))
    frame_step = calculate_frame_step(video_length)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cvtColor(frame, COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame)

        if frame_count % frame_step == 1:
            description = predict_step(
                pil_image, model, feature_extractor, tokenizer, device, gen_kwargs
            )
            descriptions.append(description)

        frame_count += 1

    cap.release()
    return descriptions


def analyze_video_file(video_file_path: str):
    """Управляет анализом видеоряда.

    Args:
        video_file_path (str): путь к файлу видео;

    Returns:
        list[str]: список описаний проанализированных кадров.
    """
    model, feature_extractor, tokenizer, device = load_model_and_processors()

    # Параметры генерации
    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    # Возвращаемый набор описаний кадров
    video_descriptions = analyze_video(
        video_file_path, model, feature_extractor, tokenizer, device, gen_kwargs
    )

    return video_descriptions
