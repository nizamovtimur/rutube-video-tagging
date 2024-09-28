import os
from faster_whisper import WhisperModel

def transcribe_and_save(model: WhisperModel, video_path: str):
    """Расшифровывает аудиофайлы и сохраняет текст в файлы."""
    if filename.endswith(".mp4"):
        segments_n, _ = model.transcribe(audio=video_path, word_timestamps=False, condition_on_previous_text=False)

        text = ''.join(segment.text for segment in segments_n)
        output_filename = os.path.splitext(filename)[0] + ".txt"

        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Расшифровка {filename} завершена. Текст сохранен в {output_filename}")

def main(video_path: str):
    """Основная функция, объединяющая все шаги."""

    model = WhisperModel("medium", compute_type="int8", device="cuda")
    transcribe_and_save(model, video_path)

if __name__ == "__main__":
    filename = "/content/0a7a288165c6051ebd74010be4dc9aa8.mp4"
    main(filename)
