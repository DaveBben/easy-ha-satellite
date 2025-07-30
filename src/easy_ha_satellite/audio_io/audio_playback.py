import asyncio
import io
import queue
import threading
from dataclasses import dataclass
from enum import Enum

import numpy as np
import sounddevice as sd
from pydub import AudioSegment

from easy_ha_satellite.config import get_logger

from .schemas import OutputAudioConfig

logger = get_logger("audio_playback")


class PlaybackCommand(Enum):
    PLAY = "play"
    STOP = "stop"
    SHUTDOWN = "shutdown"


@dataclass
class PlaybackItem:
    command: PlaybackCommand
    audio_data: bytes | None = None
    future: asyncio.Future | None = None
    loop: asyncio.AbstractEventLoop | None = None


class AudioPlayback:
    def __init__(self, cfg: OutputAudioConfig, output_device: str | None = None):
        self._cfg = cfg
        self._device = output_device

        self._playback_queue: queue.Queue[PlaybackItem] = queue.Queue()
        self._playback_thread: threading.Thread | None = None
        self._thread_started = False
        self._shutdown_event = threading.Event()

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

    def _start_playback_thread(self):
        """Start the persistent playback thread if not already running."""
        if not self._thread_started:
            self._thread_started = True
            self._playback_thread = threading.Thread(
                target=self._playback_worker,
                daemon=True,
            )
            self._playback_thread.start()
            logger.debug("Started persistent playback thread")

    async def play(self, audio_data: bytes, remix: bool = True) -> None:
        """
        Queues audio for playback. Immediately stops any currently playing audio.
        """
        logger.debug(f"AudioPlayback.play() called with {len(audio_data)} bytes, remix={remix}")

        logger.debug("Starting playback thread...")
        self._start_playback_thread()

        # Prepare audio data
        try:
            logger.debug("Preparing audio data...")
            final_audio_data = audio_data
            if remix:
                logger.debug("Remixing audio...")
                final_audio_data = await asyncio.to_thread(
                    self.remix_audio, final_audio_data, self._cfg
                )
            logger.debug(f"Audio preparation complete, final size: {len(final_audio_data)} bytes")
        except Exception as e:
            logger.error(f"Failed to prepare audio for playback: {e}")
            return

        # Stop any current playback
        logger.debug("Queuing STOP command...")
        self._playback_queue.put(PlaybackItem(PlaybackCommand.STOP))

        # Queue new audio
        logger.debug(
            f"Creating future and queuing PLAY command with {len(final_audio_data)} bytes..."
        )
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        # Store the loop reference with the item so the worker thread can use it
        item = PlaybackItem(PlaybackCommand.PLAY, final_audio_data, future)
        item.loop = loop  # Add loop reference
        self._playback_queue.put(item)
        logger.debug(f"PLAY command queued, queue size: {self._playback_queue.qsize()}")

        # Wait for playback to complete
        logger.debug("Waiting for playback to complete...")
        await future
        logger.debug("Playback completed!")

    def _playback_worker(self):
        """
        Persistent worker thread that processes audio playback commands from the queue.
        """
        logger.debug("Playback worker thread started")
        stream = None
        current_future = None

        try:
            # Create the output stream once
            logger.debug(
                f"Creating OutputStream: rate={self._cfg.sample_rate}, channels={self._cfg.channels}, device={self._device}, dtype={self._cfg.dtype}"
            )
            stream = sd.OutputStream(
                samplerate=self._cfg.sample_rate,
                channels=self._cfg.channels,
                device=self._device,
                dtype=self._cfg.dtype,
            )
            logger.debug("OutputStream created, starting stream...")
            stream.start()
            logger.debug(f"Persistent stream started successfully on device {self._device}")

            while not self._shutdown_event.is_set():
                try:
                    # Wait for commands with timeout
                    item = self._playback_queue.get(timeout=0.1)

                    if item.command == PlaybackCommand.SHUTDOWN:
                        logger.debug("Received shutdown command")
                        break

                    elif item.command == PlaybackCommand.STOP:
                        # Just mark current playback as complete
                        if current_future and not current_future.done():
                            if hasattr(item, "loop") and item.loop:
                                item.loop.call_soon_threadsafe(current_future.set_result, None)
                            else:
                                current_future.set_result(None)
                        current_future = None
                        logger.debug("Stopped current playback")

                    elif item.command == PlaybackCommand.PLAY:
                        logger.debug(
                            f"Received PLAY command with {len(item.audio_data) if item.audio_data else 0} bytes"
                        )
                        # Mark previous playback as complete
                        if current_future and not current_future.done():
                            if hasattr(item, "loop") and item.loop:
                                item.loop.call_soon_threadsafe(current_future.set_result, None)
                            else:
                                current_future.set_result(None)

                        current_future = item.future
                        current_loop = item.loop
                        logger.debug("Starting audio playback...")

                        # Play the audio
                        self._play_audio_on_stream(stream, item.audio_data)
                        logger.debug("Audio playback finished")

                        # Mark playback as complete
                        if current_future and not current_future.done():
                            logger.debug("Setting future result")
                            if current_loop:
                                current_loop.call_soon_threadsafe(current_future.set_result, None)
                            else:
                                current_future.set_result(None)
                        current_future = None

                except queue.Empty:
                    continue
                except Exception:
                    logger.exception("Error in playback worker")
                    if current_future and not current_future.done():
                        # Try to use loop if available, otherwise set directly
                        try:
                            if "current_loop" in locals() and current_loop:
                                current_loop.call_soon_threadsafe(
                                    current_future.set_exception, Exception("Playback failed")
                                )
                            else:
                                current_future.set_exception(Exception("Playback failed"))
                        except Exception:
                            # If we can't set the exception, at least log it
                            logger.exception("Failed to set exception on future")
                    current_future = None

        finally:
            if stream:
                stream.stop()
                stream.close()
            logger.debug("Playback worker thread stopped")

    def _play_audio_on_stream(self, stream: sd.OutputStream, audio_bytes: bytes):
        """Play audio data on the existing stream."""
        chunk_size = (
            self._cfg.chunk_samples * self._cfg.channels * np.dtype(self._cfg.dtype).itemsize
        )
        total_bytes = len(audio_bytes)

        for i in range(0, total_bytes, chunk_size):
            # Check if we should stop
            if not self._playback_queue.empty():
                # Peek at the queue to see if there's a stop command
                try:
                    next_item = self._playback_queue.queue[0]
                    if next_item.command in (PlaybackCommand.STOP, PlaybackCommand.SHUTDOWN):
                        logger.debug("Interrupting playback for new command")
                        break
                except IndexError:
                    pass

            chunk = audio_bytes[i : i + chunk_size]
            pcm_data = np.frombuffer(chunk, dtype=self._cfg.dtype)

            # Reshape for multi-channel audio
            if self._cfg.channels > 1:
                pcm_data = pcm_data.reshape(-1, self._cfg.channels)

            stream.write(pcm_data)

    async def stop(self) -> None:
        """Stops any current audio playback."""
        if self._thread_started:
            self._playback_queue.put(PlaybackItem(PlaybackCommand.STOP))

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.shutdown()

    async def shutdown(self):
        """Shutdown the playback thread and clean up resources."""
        if self._thread_started:
            logger.debug("Shutting down playback thread")
            self._shutdown_event.set()
            self._playback_queue.put(PlaybackItem(PlaybackCommand.SHUTDOWN))

            if self._playback_thread:
                await asyncio.to_thread(self._playback_thread.join, timeout=2.0)

            self._thread_started = False
            logger.debug("Playback thread shutdown complete")

    def play_immediate(self, audio_data: bytes, remix: bool = True) -> None:
        """
        Play audio immediately with minimal latency. Bypasses the queue system.
        Intended for urgent alerts and notifications that need instant playback.
        
        This is a synchronous function that blocks until playback is complete.
        Use for short audio clips (< 2 seconds) to avoid blocking the calling thread.
        """
        logger.debug(f"play_immediate() called with {len(audio_data)} bytes")
        
        try:
            # Prepare audio data
            final_audio_data = audio_data
            if remix:
                final_audio_data = self.remix_audio(final_audio_data, self._cfg)
            
            # Create and start stream immediately
            with sd.OutputStream(
                samplerate=self._cfg.sample_rate,
                channels=self._cfg.channels,
                device=self._device,
                dtype=self._cfg.dtype,
            ) as stream:
                logger.debug("Immediate playback stream created and started")
                
                # Convert audio data to numpy array
                pcm_data = np.frombuffer(final_audio_data, dtype=self._cfg.dtype)
                
                # Reshape for multi-channel audio
                if self._cfg.channels > 1:
                    pcm_data = pcm_data.reshape(-1, self._cfg.channels)
                
                # Play all audio data at once
                stream.write(pcm_data)
                logger.debug("Immediate playback completed")
                
        except Exception as e:
            logger.error(f"Failed to play audio immediately: {e}")
            raise

    @property
    def audio_config(self) -> OutputAudioConfig:
        return self._cfg
