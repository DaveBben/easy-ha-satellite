import asyncio
import queue
from multiprocessing.synchronize import Semaphore
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
        mic_lock: Semaphore | None = None,
    ):
        self._cfg = cfg
        self._device = input_device
        self._stream = None
        self._q: Queue[bytes] = Queue(maxsize=queue_maxsize)
        self._closed = False
        self._lock = mic_lock
        self._have_lock = False

    def start(self, block: bool = True, lock_timeout: float | None = None) -> None:
        """Acquire the mic lock (if provided) and start the stream."""
        if self._stream and self._stream.active:
            logger.warning("Input stream already running.")
            return

        if self._lock:
            ok = self._lock.acquire(block, lock_timeout)
            if not ok:
                raise RuntimeError("Could not acquire microphone lock")
            self._have_lock = True

        def _audio_cb(indata, frames, time_info, status):
            if status:
                logger.warning("Input status: %s", status)
            try:
                self._q.put_nowait(bytes(indata))
            except Full:
                logger.warning("Input queue full; dropping audio chunk.")

        try:
            self._stream = sd.RawInputStream(
                samplerate=self._cfg.sample_rate,
                channels=self._cfg.channels,
                dtype=self._cfg.dtype,
                device=self._device,
                blocksize=self._cfg.chunk_samples,
                callback=_audio_cb,
            )
            self._stream.start()
            logger.debug("Audio input stream started (device=%s)", self._device or "default")
        except Exception:
            logger.exception("Error starting input stream")
            self._cleanup_lock()
            self._stream = None
            raise

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
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                logger.exception("Error stopping input stream")
            finally:
                self._stream = None
        self._closed = True
        self._drain_queue()
        self._cleanup_lock()
        logger.debug("Audio input stream stopped.")

    def _put(self, item: bytes) -> None:
        try:
            self._q.put_nowait(item)
        except queue.Full:
            logger.warning("Input capture queue full; dropping audio chunk.")

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

    def _cleanup_lock(self) -> None:
        if self._have_lock:
            self._lock.release()
            self._have_lock = False


class AsyncCaptureSession:
    """
    Async wrapper to take the mic lock and AudioCapture synchronously.
    """

    def __init__(self, cfg, sem: Semaphore, device: str | None = None):
        self._cfg = cfg
        self._sem = sem
        self._device = device
        self._cap: AudioCapture | None = None

    async def __aenter__(self) -> AudioCapture:
        # Acquire lock, start capture off main loop
        await asyncio.to_thread(self._sem.acquire)
        self._cap = AudioCapture(self._cfg, self._device)
        self._cap.start()
        return self._cap

    async def __aexit__(self, exc_type, exc, tb) -> None:
        assert self._cap is not None
        await asyncio.to_thread(self._cap.stop)
        self._sem.release()
