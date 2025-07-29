
import sounddevice as sd


def query_audio_devices():
    print("\n" + "-" * 50)
    print("Available audio input devices:")
    print(sd.query_devices())
    print("-" * 50)