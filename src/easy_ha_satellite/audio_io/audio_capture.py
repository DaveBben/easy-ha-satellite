import asyncio
from collections import deque
from contextlib import suppress
from queue import Empty, Full, Queue

import numpy as np
import sounddevice as sd
from webrtc_noise_gain import AudioProcessor

from easy_ha_satellite.config import get_logger

from .schemas import InputAudioConfig

logger = get_logger("audio_capture")


class SimpleAudioProcessor:
    """
    Lightweight audio processor with automatic gain control and basic noise reduction.
    Designed for low CPU usage on devices like Raspberry Pi.
    """

    def __init__(
        self,
        target_dbfs: float = -20.0,
        attack_time: float = 0.1,
        release_time: float = 1.0,
        sample_rate: int = 16000,
        noise_gate_ratio: float = 2.0,
    ):
        self.target_dbfs = target_dbfs
        # Convert dBFS to linear scale for int16 audio
        self.target_rms = 10 ** (target_dbfs / 20) * 32768

        # Time constants for smooth gain adjustment
        # Based on 10ms chunks (160 samples at 16kHz)
        chunks_per_second = sample_rate / 160
        self.attack = 1.0 - np.exp(-1.0 / (attack_time * chunks_per_second))
        self.release = 1.0 - np.exp(-1.0 / (release_time * chunks_per_second))

        self.gain = 1.0
        self.min_gain = 0.1
        self.max_gain = 20.0  # Allow higher gain for quiet inputs

        # Noise reduction parameters
        self.noise_floor = None
        self.noise_buffer = deque(maxlen=50)  # 500ms of RMS history
        self.noise_gate_ratio = noise_gate_ratio

        # Prevent amplifying silence
        self.gate_threshold = 0.01 * 32768  # -40 dBFS

        # Debug counters
        self.chunk_count = 0
        self.debug_interval = 500  # Log every 5 seconds at 16kHz/10ms

        logger.info(
            f"Initialized SimpleAudioProcessor: target={target_dbfs}dBFS, "
            f"attack={attack_time}s, release={release_time}s"
        )

    def Process10ms(self, audio_bytes: bytes) -> object:
        """
        Process a 10ms audio chunk. Returns an object with 'audio' attribute
        to match the webrtc_noise_gain interface.
        """
        # Convert to float32 for processing
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)

        # Calculate RMS for this chunk
        rms = np.sqrt(np.mean(audio**2))

        # Update noise floor estimate
        self._update_noise_floor(rms)

        # Apply noise gate
        if self.noise_floor and rms < self.noise_floor * self.noise_gate_ratio:
            # This is likely background noise, attenuate it
            audio *= 0.3

        # Apply AGC only if signal is above gate threshold
        if rms > self.gate_threshold:
            audio = self._apply_agc(audio, rms)
        else:
            # Gradually reduce gain during silence
            self.gain *= 0.99
            self.gain = max(self.gain, self.min_gain)

        # Soft clipping to prevent harsh distortion
        audio = np.tanh(audio / 32768) * 32768

        # Convert back to int16
        audio_out = np.clip(audio, -32768, 32767).astype(np.int16)

        # Debug logging every few seconds
        self.chunk_count += 1
        if self.chunk_count % self.debug_interval == 0:
            out_rms = np.sqrt(np.mean(audio_out.astype(np.float32) ** 2))
            noise_floor_str = f"{self.noise_floor:.0f}" if self.noise_floor else "None"
            logger.debug(
                f"AGC: input_rms={rms:.0f} ({20 * np.log10(rms / 32768):.1f}dBFS), "
                f"output_rms={out_rms:.0f} ({20 * np.log10(out_rms / 32768):.1f}dBFS), "
                f"gain={self.gain:.2f}, noise_floor={noise_floor_str}"
            )

        # Return object with audio attribute for compatibility
        class Result:
            pass

        result = Result()
        result.audio = audio_out.tobytes()
        return result

    def _update_noise_floor(self, rms: float):
        """Update noise floor estimate from quiet periods."""
        self.noise_buffer.append(rms)

        if len(self.noise_buffer) >= 10:  # Need at least 100ms of data
            # Use 20th percentile as noise floor estimate
            self.noise_floor = float(np.percentile(list(self.noise_buffer), 20))

    def _apply_agc(self, audio: np.ndarray, rms: float) -> np.ndarray:
        """Apply automatic gain control with attack/release smoothing."""
        if rms > 0:
            # Calculate desired gain to reach target RMS
            desired_gain = self.target_rms / rms
            desired_gain = np.clip(desired_gain, self.min_gain, self.max_gain)

            # Apply attack/release envelope
            if desired_gain < self.gain:
                # Attack: respond quickly to loud sounds
                self.gain += (desired_gain - self.gain) * self.attack
            else:
                # Release: respond slowly when getting quieter
                self.gain += (desired_gain - self.gain) * self.release

            # Apply gain
            audio = audio * self.gain

        return audio


class AudioCapture:
    def __init__(
        self,
        cfg: InputAudioConfig,
        input_device: str | None = None,
        queue_maxsize: int = 256,
        webrtc_noise_gain: bool = True,
        auto_gain_dbfs: int = 31,
        noise_supression_level: int = 2,
        use_simple_processor: bool = True,
    ):
        self._cfg = cfg
        self._device = input_device
        self._stream = None
        self._q: Queue[bytes] = Queue(maxsize=queue_maxsize)
        self._closed = False

        self._audio_proc = None
        if use_simple_processor:
            # Use the lightweight numpy-based processor
            # WebRTC uses values like 31 to mean target level, convert to reasonable dBFS
            # Higher values = louder output, WebRTC 31 ≈ -12dBFS, 25 ≈ -18dBFS
            target_dbfs = -50.0 + (auto_gain_dbfs * 1.2)  # Convert to dBFS range
            target_dbfs = max(-30.0, min(-6.0, target_dbfs))  # Clamp to reasonable range

            self._audio_proc = SimpleAudioProcessor(
                target_dbfs=target_dbfs,
                sample_rate=cfg.sample_rate,
                attack_time=0.05,  # Faster attack for wake word detection
                release_time=0.5,
                noise_gate_ratio=noise_supression_level,
            )
            logger.info(f"Using SimpleAudioProcessor with target {target_dbfs:.1f}dBFS")
        elif webrtc_noise_gain:
            # Use webrtc processor
            self._audio_proc = AudioProcessor(auto_gain_dbfs, noise_supression_level)
            logger.info("Using webrtc_noise_gain AudioProcessor")

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
        audio = bytes(indata)
        if self._audio_proc:
            audio = self._audio_proc.Process10ms(bytes(indata)).audio
        try:
            self._q.put_nowait(audio)
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

    def __init__(self, cfg, device: str | None = None, use_simple_processor: bool = False):
        self._cfg = cfg
        self._device = device
        self._use_simple_processor = use_simple_processor
        self._cap: AudioCapture | None = None

    async def __aenter__(self) -> AudioCapture:
        self._cap = AudioCapture(
            self._cfg, self._device, use_simple_processor=self._use_simple_processor
        )
        await asyncio.to_thread(self._cap.start)
        return self._cap

    async def __aexit__(self, exc_type, exc, tb) -> None:
        assert self._cap is not None
        await asyncio.to_thread(self._cap.stop)
