import cv2
import torch
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    GPT2TokenizerFast,
)
from PIL import Image


def load_model_and_processors() -> (
    VisionEncoderDecoderModel | ViTImageProcessor | GPT2TokenizerFast | torch.device
):
    """Инициализация моделей

    Returns:
        VisionEncoderDecoderModel|ViTImageProcessor|GPT2TokenizerFast|torch.device: модели для анализа видео
    """
    model = VisionEncoderDecoderModel.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )
    feature_extractor = ViTImageProcessor.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, feature_extractor, tokenizer, device


def predict_step(
    image: Image.Image,
    model: VisionEncoderDecoderModel,
    feature_extractor: ViTImageProcessor,
    tokenizer: GPT2TokenizerFast,
    device: torch.device,
    gen_kwargs: dict[str, int],
) -> str:
    """Генерирует описание для изображения при помощи модели

    Args:
        image (Image.Image): озображение для анализа
        model (VisionEncoderDecoderModel): _description_
        feature_extractor (ViTImageProcessor): _description_
        tokenizer (GPT2TokenizerFast): _description_
        device (torch.device): _description_
        gen_kwargs (dict[str, int]): _description_

    Returns:
        str: _description_
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
    device: torch.device,
    gen_kwargs: dict[str, int],
) -> list[str]:
    """Анализ видео путём генерирования описания для выбранный кадров

    Args:
        video_path (str): путь к файлу видео
        model (VisionEncoderDecoderModel): _description_
        feature_extractor (ViTImageProcessor): _description_
        tokenizer (GPT2TokenizerFast): _description_
        device (torch.device): _description_
        gen_kwargs (dict[str, int]): _description_

    Returns:
        list[str]: _description_
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    descriptions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame)

        if frame_count % 10 == 1:
            description = predict_step(
                pil_image, model, feature_extractor, tokenizer, device, gen_kwargs
            )
            descriptions.append(description)

        frame_count += 1

    cap.release()
    return descriptions


def main(video_file_path: str):
    """Загрузка модели, получение пути к видео, анализ видео

    Args:
        video_file_path (str): путь к файлу видео
    """
    model, feature_extractor, tokenizer, device = load_model_and_processors()

    # Параметры генерации
    max_length = 24
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    video_descriptions = analyze_video(
        video_file_path, model, feature_extractor, tokenizer, device, gen_kwargs
    )

    for desc in video_descriptions:
        print(desc)


if __name__ == "__main__":
    video_file_path = "/content/0a7a288165c6051ebd74010be4dc9aa8.mp4"
    main(video_file_path)
