import openwakeword
from openwakeword.model import Model
import pyaudio
import numpy as np
import queue
import sys
import os
import time

# --- Configuration ---
WAKEWORD_MODEL = os.environ.get("WAKEWORD_MODEL_NAME", "hey_jarvis")

# Audio settings
FORMAT = pyaudio.paInt16 # 16-bit integers
CHANNELS = 1
RATE = 16000
CHUNK_SIZE = 1280 # 80ms of audio

# A queue to hold audio data
audio_queue = queue.Queue()

def audio_callback(in_data, frame_count, time_info, status):
    """This callback is called by PyAudio for each new audio chunk."""
    try:
        audio_queue.put(in_data)
    except Exception as e:
        print(f"Error in audio_callback: {e}", file=sys.stderr)
    return (None, pyaudio.paContinue)

if __name__ == "__main__":
    pa = None
    stream = None
    try:
        # --- Instantiate PyAudio ---
        pa = pyaudio.PyAudio()
        print("--- PyAudio Details ---")
        print(f"Version: {pyaudio.get_portaudio_version_text()}")
        print(f"Host API Count: {pa.get_host_api_count()}")
        print(f"Default Input Device: {pa.get_default_input_device_info().get('name')}")
        print("-----------------------")

        # --- Instantiate OpenWakeWord ---
        print(f"Instantiating openwakeword model: {WAKEWORD_MODEL}...")
        oww_model = Model(wakeword_models=[WAKEWORD_MODEL])

        # --- Open Audio Stream ---
        # We are opening the default input device, which PyAudio will get from PulseAudio
        stream = pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
            stream_callback=audio_callback
        )
        stream.start_stream()
        print(f"\n--- Listening for '{WAKEWORD_MODEL}' using default input device ---")

        while stream.is_active():
            # Get a chunk of audio from the queue (put there by the callback)
            audio_data = audio_queue.get()
            
            # Convert byte data to a numpy array
            audio_chunk = np.frombuffer(audio_data, dtype=np.int16)

            # Feed the audio to openwakeword
            prediction = oww_model.predict(audio_chunk)

            if prediction[WAKEWORD_MODEL] > 0.5:
                print(f"Wake word detected! (Score: {prediction[WAKEWORD_MODEL]:.2f})")
                with audio_queue.mutex:
                    audio_queue.queue.clear()
            
            # A small sleep to prevent this loop from pegging the CPU
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if stream and stream.is_active():
            stream.stop_stream()
        if stream:
            stream.close()
        if pa:
            pa.terminate()
        print("Application finished.")
