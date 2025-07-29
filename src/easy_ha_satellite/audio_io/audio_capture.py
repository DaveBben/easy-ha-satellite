import asyncio
from contextlib import suppress
from queue import Empty, Full, Queue
from typing import Any

import numpy as np
import sounddevice as sd

from easy_ha_satellite.config import get_logger

from .audio_processing.utils import initialize_noise_reducer, reduce_noise_chunk
from .schemas import InputAudioConfig

logger = get_logger("audio_capture")


class AudioCapture:
    def __init__(
        self,
        cfg: InputAudioConfig,
        input_device: str | None = None,
        queue_maxsize: int = 256,
        reduce_noise: bool = True,
    ):
        self._cfg = cfg
        self._device = input_device
        self._stream = None
        self._q: Queue[bytes] = Queue(maxsize=queue_maxsize)
        self._closed = False

        # --- Noise Reduction State ---
        self._reduce_noise = reduce_noise
        self._reducer_state: dict[str, Any] | None = None
        if self._reduce_noise:
            if self._cfg.channels > 1:
                logger.warning("Noise reduction only supports mono audio. Disabling.")
                self._reduce_noise = False
            else:
                logger.info("Noise reduction enabled.")
                self._reducer_state = initialize_noise_reducer(
                    chunk_size=self._cfg.chunk_samples,
                    channels=self._cfg.channels,
                )

    def start(self, block: bool = True) -> None:
        """Acquire the mic and start the stream."""
        if self._stream and self._stream.active:
            logger.warning("Input stream already running.")
            return

        try:
            self._stream = sd.RawInputStream(
                samplerate=self._cfg.sample_rate,
                channels=self._cfg.channels,
                dtype=self._cfg.dtype,
                device=self._device,
                blocksize=self._cfg.chunk_samples,
                callback=self._audio_cb,
            )
            self._stream.start()
            logger.debug("Audio input stream started (device=%s)", self._device or "default")
        except Exception:
            logger.exception("Error starting input stream")
            self._stream = None
            raise

    def _audio_cb(self, indata: bytes, frames: int, time_info, status) -> None:
        """The audio callback. Applies noise reduction if enabled."""
        if self._closed:
            return
        if status:
            logger.warning("Input status: %s", status)

        processed_bytes = indata

        if self._reduce_noise and self._reducer_state:
            # Convert bytes to float32 numpy array for processing
            audio_chunk_np = np.frombuffer(indata, dtype=self._cfg.dtype).astype(np.float32)
            if np.issubdtype(self._cfg.dtype, np.integer):
                audio_chunk_np /= 32767.0

            # Apply noise reduction
            denoised_chunk = reduce_noise_chunk(audio_chunk_np, self._reducer_state)

            # Convert back to original dtype and then to bytes
            if np.issubdtype(self._cfg.dtype, np.integer):
                denoised_chunk = (denoised_chunk * 32767).astype(self._cfg.dtype)
            processed_bytes = denoised_chunk.tobytes()

        try:
            self._q.put_nowait(processed_bytes)
        except Full:
            logger.warning("Input queue full; dropping audio chunk.")

    def get_chunk(self, timeout: float | None = 0.1, block: bool = True) -> bytes | None:
        try:
            if block:
                return self._q.get(timeout=timeout)
            return self._q.get_nowait()
        except Empty:
            return None

    def clear_buffer(self) -> None:
        self._drain_queue()

    def stop(self) -> None:
        self._closed = True
        with suppress(Full):
            self._q.put_nowait(None)
        if self._stream:
            try:
                self._stream.stop()
                self._stream.abort()
            except Exception:
                logger.exception("Error stopping input stream")
            finally:
                self._stream = None
        self._drain_queue()
        logger.debug("Audio input stream stopped.")

    def __enter__(self) -> "AudioCapture":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def _drain_queue(self) -> None:
        while True:
            try:
                self._q.get_nowait()
            except Empty:
                break


class AsyncCaptureSession:
    """
    Async wrapper to AudioCapture synchronously.
    """

    def __init__(self, cfg, device: str | None = None, reduce_noise: bool = True):
        self._cfg = cfg
        self._device = device
        self._reduce_noise = reduce_noise
        self._cap: AudioCapture | None = None

    async def __aenter__(self) -> AudioCapture:
        self._cap = AudioCapture(self._cfg, self._device, reduce_noise=self._reduce_noise)
        await asyncio.to_thread(self._cap.start)
        return self._cap

    async def __aexit__(self, exc_type, exc, tb) -> None:
        assert self._cap is not None
        await asyncio.to_thread(self._cap.stop)
