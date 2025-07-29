import asyncio
import io
import threading

import numpy as np
import sounddevice as sd
from pydub import AudioSegment

from easy_ha_satellite.config import get_logger

from .schemas import OutputAudioConfig

logger = get_logger("audio_playback")


class AudioPlayback:
    def __init__(self, cfg: OutputAudioConfig, output_device: str | None = None):
        self._cfg = cfg
        self._device = output_device

        self._playback_thread: threading.Thread | None = None
        self._playback_lock = threading.Lock()
        self._playback_id = 0  # Used to signal the active playback task

    @classmethod
    def remix_audio(cls, audio_data: bytes, cfg: OutputAudioConfig) -> bytes:
        logger.debug("Remixing Audio")
        song = AudioSegment.from_file(io.BytesIO(audio_data))

        if song.frame_rate != cfg.sample_rate:
            song = song.set_frame_rate(cfg.sample_rate)
        if song.channels != cfg.channels:
            song = song.set_channels(cfg.channels)

        target_sample_width = np.dtype(cfg.dtype).itemsize
        if song.sample_width != target_sample_width:
            song = song.set_sample_width(target_sample_width)

        samples = np.array(song.get_array_of_samples())
        final_audio_data = samples.reshape((-1, song.channels)) if song.channels > 1 else samples
        audio_bytes = final_audio_data.tobytes()
        return audio_bytes

    async def play(self, audio_data: bytes, remix: bool = True) -> None:
        """
        Plays audio by starting a new playback thread. Immediately stops any
        previously playing audio.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._stop_and_play_sync, audio_data, remix)

    def _stop_and_play_sync(self, audio_data: bytes, remix: bool):
        """The new synchronous core of the play logic."""
        thread_to_join = None
        with self._playback_lock:
            # 1. Invalidate any existing playback thread by incrementing the ID.
            self._playback_id += 1

            # Grab a reference to the old thread to join it later, outside the lock.
            if self._playback_thread and self._playback_thread.is_alive():
                thread_to_join = self._playback_thread

            # Prepare the audio data.
            try:
                final_audio_data = audio_data
                if remix:
                    final_audio_data = self.remix_audio(final_audio_data, self._cfg)
            except Exception as e:
                logger.error(f"Failed to prepare audio for playback: {e}")
                return

            # Create and start the new playback thread.
            logger.debug("Starting new playback thread.")
            self._playback_thread = threading.Thread(
                target=self._playback_thread_target,
                args=(final_audio_data, self._playback_id),
            )
            self._playback_thread.start()

        # Now, outside the lock, wait for the old thread to finish.
        if thread_to_join:
            logger.debug("Waiting for previous playback thread to stop...")
            thread_to_join.join()

    def _playback_thread_target(self, audio_bytes: bytes, expected_id: int):
        """
        This function runs in a separate thread and handles the actual audio playback.
        It uses a blocking stream for simplicity and efficiency.
        """
        try:
            stream = sd.OutputStream(
                samplerate=self._cfg.sample_rate,
                channels=self._cfg.channels,
                device=self._device,
                dtype=self._cfg.dtype,
            )
            stream.start()
            logger.debug(f"Playback thread {expected_id} started stream on device {self._device}")

            chunk_size = (
                self._cfg.chunk_samples * self._cfg.channels * np.dtype(self._cfg.dtype).itemsize
            )
            total_bytes = len(audio_bytes)

            for i in range(0, total_bytes, chunk_size):
                # Check if we should still be playing ---
                with self._playback_lock:
                    if self._playback_id != expected_id:
                        logger.debug(f"Playback {expected_id} superseded. Stopping.")
                        break  # Exit the loop immediately

                chunk = audio_bytes[i : i + chunk_size]
                pcm_data = np.frombuffer(chunk, dtype=self._cfg.dtype)
                # Reshape for multi-channel audio before writing to the stream
                if self._cfg.channels > 1:
                    pcm_data = pcm_data.reshape(-1, self._cfg.channels)

                stream.write(pcm_data)

            stream.stop()
            stream.close()
            logger.debug(f"Playback thread {expected_id} finished cleanly.")

        except Exception:
            logger.exception(f"Error in playback thread {expected_id}.")

    async def stop(self) -> None:
        """Stops the current audio playback thread."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._stop_sync)

    def _stop_sync(self):
        """The new synchronous core of the stop logic."""
        with self._playback_lock:
            if self._playback_thread and self._playback_thread.is_alive():
                logger.debug("Stopping playback thread...")
                self._playback_id += 1  # Signal the thread to stop
                self._playback_thread.join()
                logger.debug("Playback thread stopped.")
            self._playback_thread = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.stop()

    @property
    def audio_config(self) -> OutputAudioConfig:
        return self._cfg
