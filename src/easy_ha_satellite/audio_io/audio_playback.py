import asyncio
import io
from contextlib import suppress
from queue import Empty, Full, Queue

import numpy as np
import sounddevice as sd
from pydub import AudioSegment

from easy_ha_satellite.config import get_logger

from .audio_processing.utils import prepare_audio
from .schemas import OutputAudioConfig

logger = get_logger("audio_playback")


class AudioPlayback:
    def __init__(
        self, cfg: OutputAudioConfig, output_device: str | None = None, queue_maxsize: int = 0
    ):
        self._cfg = cfg
        self._device = output_device
        self._stream = None
        self._buf_q: Queue[bytes] = Queue(maxsize=queue_maxsize)
        self._feeder_task: asyncio.Task | None = None
        self._is_running = False

    async def play(self, audio_data: np.ndarray | bytes, source_sr: int | None = None) -> None:
        """
        Plays audio, ensuring it's converted to the correct output format first.

        This is the primary method for playback. It handles two cases:
        1. audio_data is bytes: Decodes from any format (MP3, WAV, etc.) and prepares it.
        2. audio_data is ndarray: Prepares the raw PCM data. `source_sr` is required.

        Args:
            audio_data: The audio data to play.
            source_sr: The sample rate of the audio. Required if audio_data is a
                       NumPy array, but ignored if it's bytes.
        """
        await self.stop()

        final_audio_data: np.ndarray
        try:
            if isinstance(audio_data, bytes):
                logger.debug("Received raw bytes. Decoding and preparing audio...")
                song = AudioSegment.from_file(io.BytesIO(audio_data))

                if song.frame_rate != self._cfg.sample_rate:
                    song = song.set_frame_rate(self._cfg.sample_rate)
                if song.channels != self._cfg.channels:
                    song = song.set_channels(self._cfg.channels)

                target_sample_width = np.dtype(self._cfg.dtype).itemsize
                if song.sample_width != target_sample_width:
                    song = song.set_sample_width(target_sample_width)

                samples = np.array(song.get_array_of_samples())
                final_audio_data = (
                    samples.reshape((-1, song.channels)) if song.channels > 1 else samples
                )

            elif isinstance(audio_data, np.ndarray):
                logger.debug("Received NumPy array. Preparing audio...")
                if source_sr is None:
                    raise ValueError(
                        "source_sr must be provided when passing a NumPy array to play()"
                    )

                final_audio_data = prepare_audio(
                    audio_data=audio_data,
                    source_sr=source_sr,
                    target_sr=self._cfg.sample_rate,
                    target_channels=self._cfg.channels,
                    target_dtype=self._cfg.dtype,
                )
            else:
                raise TypeError(f"Unsupported audio_data type: {type(audio_data)}")

        except Exception as e:
            logger.error(f"Failed to prepare audio for playback: {e}")
            return

        logger.debug("Starting new audio stream feeder.")
        self._feeder_task = asyncio.create_task(self._prime_and_feed(final_audio_data))

    async def _prime_and_feed(self, audio_data: np.ndarray):
        """
        Primes the buffer with the first chunk, starts the stream, then feeds the rest.
        This prevents an initial click/pop from a buffer underrun.
        """
        try:
            audio_bytes = audio_data.tobytes()
            chunk_size_bytes = self._cfg.bytes_per_chunk

            # 1. Prime the buffer with the first chunk.
            first_chunk = audio_bytes[:chunk_size_bytes]
            if not first_chunk:
                logger.warning("Audio data is empty, nothing to play.")
                return

            self._clear_buffer()
            await self.put_chunk(first_chunk)

            # 2. Now that the buffer is primed, start the stream.
            await self.start()

            # 3. Feed the rest of the audio data.
            for i in range(chunk_size_bytes, len(audio_bytes), chunk_size_bytes):
                await self.put_chunk(audio_bytes[i : i + chunk_size_bytes])

            # 4. Signal the end of the stream.
            await self.put_chunk(None)
            logger.info("Audio stream finished feeding.")

        except asyncio.CancelledError:
            logger.info("Audio feeder task was cancelled.")
        except Exception:
            logger.exception("Error in audio feeder task.")

    async def _feeder(self, audio_data: np.ndarray):
        """
        This task chunks the audio data and puts it on the queue piece by piece.
        This allows the task to be cancelled cleanly between chunks.
        """
        try:
            # Use the bytes_per_chunk property from the config object
            chunk_size_bytes = self._cfg.bytes_per_chunk
            audio_bytes = audio_data.tobytes()
            num_bytes = len(audio_bytes)

            logger.debug(f"Feeding {num_bytes} bytes in chunks of {chunk_size_bytes}")

            # Loop and queue the data in chunks
            for i in range(0, num_bytes, chunk_size_bytes):
                chunk = audio_bytes[i : i + chunk_size_bytes]
                await self.put_chunk(chunk)

            # Signal the end of the stream
            await self.put_chunk(None)
            logger.debug("Audio stream finished.")

        except asyncio.CancelledError:
            logger.debug("Audio feeder task was cancelled.")
        except Exception:
            logger.exception("Error in audio feeder task.")

    def _audio_cb(self, outdata: np.ndarray, frames: int, time, status: sd.CallbackFlags):
        """The sounddevice callback function. Runs in a separate thread."""
        if status:
            logger.warning("Output status: %s", status)

        try:
            chunk = self._buf_q.get_nowait()
            if chunk is None:  # End of stream signal
                outdata.fill(0)
                raise sd.CallbackStop

            # Reshape the 1D buffer into a 2D array of (samples, channels)
            pcm = np.frombuffer(chunk, dtype=self._cfg.dtype).reshape(-1, self._cfg.channels)
            frames_to_copy = min(len(pcm), len(outdata))
            outdata[:frames_to_copy] = pcm[:frames_to_copy]

            if frames_to_copy < len(outdata):
                outdata[frames_to_copy:] = 0

        except Empty:
            outdata.fill(0)  # Underrun
            logger.warning("Audio buffer underrun")
            return

    async def start(self) -> None:
        """Starts the audio output stream."""
        if self._is_running:
            return

        logger.info("Starting Audio Output stream (device=%s)", self._device or "default")
        try:
            # Clear queue before starting
            self._clear_buffer()
            self._stream = sd.OutputStream(
                samplerate=self._cfg.sample_rate,
                channels=self._cfg.channels,
                device=self._device,
                dtype=self._cfg.dtype,
                blocksize=self._cfg.chunk_samples,
                callback=self._audio_cb,
            )
            self._stream.start()
            self._is_running = True
        except Exception:
            logger.exception("Error starting the output stream")
            self._stream = None
            raise

    async def stop(self) -> None:
        """Stops the feeder task and closes the current audio stream."""
        if not self._is_running and not self._feeder_task:
            return

        logger.debug("Stopping audio playback.")
        if self._feeder_task and not self._feeder_task.done():
            self._feeder_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._feeder_task
            self._feeder_task = None

        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                logger.exception("Error stopping output stream")
            finally:
                self._stream = None

        self._clear_buffer()
        self._is_running = False

    async def put_chunk(self, data: bytes | None) -> None:
        """Puts a chunk of data on the queue, waiting if it's full."""
        while self._is_running:
            try:
                self._buf_q.put_nowait(data)
                return
            except Full:
                # Cooperatively wait for a slot to open up
                await asyncio.sleep(0.005)  # Sleep for a short time

    def _clear_buffer(self) -> None:
        """Removes all items from the queue."""
        with suppress(Empty):
            while True:
                self._buf_q.get_nowait()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.stop()

    @property
    def audio_config(self) -> OutputAudioConfig:
        return self._cfg
