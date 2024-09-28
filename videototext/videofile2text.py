from audio2text import transcribe_video
from video2text import analyze_video_file

if __name__ == "__main__":
    video_path = "/content/0a7a288165c6051ebd74010be4dc9aa8.mp4"

    # Распознование аудиоряда
    transcribed_text = transcribe_video(video_path)
    print("transcriber:")
    print(transcribed_text)

    # Генерация описаний по видеоряду
    video_descriptions = analyze_video_file(video_path)
    print("\video_analyzer:")
    for desc in video_descriptions:
        print(desc)
