import asyncio
from contextlib import suppress
from queue import Empty, Full, Queue

import sounddevice as sd

from easy_ha_satellite.config import get_logger

from .schemas import InputAudioConfig

logger = get_logger("audio_capture")


class AudioCapture:
    def __init__(
        self,
        cfg: InputAudioConfig,
        input_device: str | None = None,
        queue_maxsize: int = 256,
    ):
        self._cfg = cfg
        self._device = input_device
        self._stream = None
        self._q: Queue[bytes] = Queue(maxsize=queue_maxsize)
        self._closed = False


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
        if self._closed:
            return
        if status:
            logger.warning("Input status: %s", status)

        processed_bytes = indata

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

    def __init__(self, cfg, device: str | None = None):
        self._cfg = cfg
        self._device = device
        self._cap: AudioCapture | None = None

    async def __aenter__(self) -> AudioCapture:
        self._cap = AudioCapture(self._cfg, self._device)
        await asyncio.to_thread(self._cap.start)
        return self._cap

    async def __aexit__(self, exc_type, exc, tb) -> None:
        assert self._cap is not None
        await asyncio.to_thread(self._cap.stop)
