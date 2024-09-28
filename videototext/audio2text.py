import os
from faster_whisper import WhisperModel
from faster_whisper.utils import download_model

def transcribe(model: WhisperModel, video_path: str) -> str:
    """Расшифровывает аудиоданные

    Args:
        model (WhisperModel): модель анализа аудио
        video_path (str): путь к файлу видео

    Returns:
        str: Строка с транскрибацией аудиоряда всего видео
    """
    segments_n, _ = model.transcribe(
        audio=video_path,
        word_timestamps=False,
        condition_on_previous_text=False,
        vad_filter=True
    )

    # Возвращаемая строка транскрибации
    text = ''.join(segment.text for segment in segments_n)

    print(f"Расшифровка {video_path} завершена.")
    return text

def load_model(size_or_id: str) -> WhisperModel:
    """Загружает модель из локального каталога или скачивает, если она отсутствует

    Args:
        size_or_id (str): название модели

    Returns:
        WhisperModel: Инициализированная модель
    """
    model_path = f"./{size_or_id}"

    if not os.path.exists(model_path):
        print(f"Модель не найдена. Загружаем модель {size_or_id}...")
        download_model(size_or_id, output_dir=model_path, local_files_only=False)

    print(f"Загрузка модели из {model_path}.")
    model = WhisperModel(model_path, compute_type="int8", device="cuda")
    return model

def transcribe_video(video_path: str) -> str:
    """Функция, реализующая speech-to-text recognition для видео

    Args:
        video_path (str): путь к файлу видео

    Returns:
        str: Строка с транскрибацией аудиоряда всего видео
    """
    model_name = "small"
    model = load_model(model_name)

    audio_text = transcribe(model, video_path)
    return audio_text
