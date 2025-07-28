from typing import Any

import numpy as np
import pyroomacoustics as pra
import sounddevice as sd
from scipy.signal import resample
from pyroomacoustics.denoise.spectral_subtraction import SpectralSub


def query_audio_devices():
    print("\n" + "-" * 50)
    print("Available audio input devices:")
    print(sd.query_devices())
    print("-" * 50)


def initialize_noise_reducer(
    chunk_size: int,
    channels: int = 1,
    nfft: int = 512,
    db_reduc: float = 10.0,
    lookback: int = 12,
) -> dict[str, Any]:
    """
    Initializes a state dictionary for streaming noise reduction.

    Args:
        chunk_size (int): The number of samples per chunk (the hop size).
        channels (int): The number of channels (only mono is supported).
        nfft (int): The FFT size.
        db_reduc (float): Maximum suppression in dB per frequency bin.
        lookback (int): Number of frames to look back for noise floor estimation.

    Returns:
        A state dictionary to be passed to `reduce_noise_chunk`.
    """
    if channels > 1:
        raise ValueError("Only mono audio is supported for noise reduction.")

    # The hop size for STFT must match the chunk size of the input stream
    hop = chunk_size

    # Create the analysis and synthesis windows
    analysis_window = pra.hann(nfft)
    synthesis_window = pra.transform.stft.compute_synthesis_window(analysis_window, hop)

    state = {
        "stft": pra.transform.STFT(
            nfft,
            hop=hop,
            analysis_window=analysis_window,
            synthesis_window=synthesis_window,
            streaming=True,
        ),
        "scnr": SpectralSub(nfft=nfft, db_reduc=db_reduc, lookback=lookback, beta=3, alpha=1.2),
    }
    return state


def reduce_noise_chunk(audio_chunk: np.ndarray, state: dict[str, Any]) -> np.ndarray:
    """
    Processes a single chunk of audio for noise reduction using a state dictionary.
    This function is designed for continuous, streaming data.

    Args:
        audio_chunk (np.ndarray): A 1D NumPy array of audio data.
        state (Dict[str, Any]): The state dictionary from `initialize_noise_reducer`.

    Returns:
        np.ndarray: The denoised audio chunk.
    """
    stft = state["stft"]
    scnr = state["scnr"]

    # Analyze the incoming chunk (updates stft.X)
    stft.analysis(audio_chunk)

    # Compute the gain filter based on the new chunk and internal state
    gain_filter = scnr.compute_gain_filter(stft.X)

    # Apply the filter and synthesize the denoised chunk
    denoised_chunk = stft.synthesis(gain_filter * stft.X)

    return denoised_chunk.astype(audio_chunk.dtype)


def prepare_audio(
    audio_data: np.ndarray,
    source_sr: int,
    target_sr: int,
    target_channels: int,
    target_dtype: np.dtype,
) -> np.ndarray:
    """
    Prepares an audio signal to match a target format by handling channel
    conversion, resampling, and data type conversion.

    Args:
        audio_data (np.ndarray): The input audio signal as a NumPy array.
                                 Can be mono (1D) or stereo (2D).
        source_sr (int): The sample rate of the input audio data.
        target_sr (int): The desired sample rate for the output.
        target_channels (int): The desired number of channels (1 for mono, 2 for stereo).
        target_dtype (np.dtype): The desired NumPy data type (e.g., np.float32, np.int16).

    Returns:
        np.ndarray: The processed audio data in the target format.
    """
    # --- 1. Ensure input is a NumPy array ---
    if not isinstance(audio_data, np.ndarray):
        raise TypeError("audio_data must be a NumPy array.")

    # Make a copy to avoid modifying the original array
    processed_data = audio_data.copy()

    # --- 2. Channel Conversion ---
    # Ensure the array is at least 2D for consistent channel handling
    if processed_data.ndim == 1:
        processed_data = processed_data[:, np.newaxis]  # Convert 1D mono to 2D

    source_channels = processed_data.shape[1]

    if source_channels != target_channels:
        if target_channels == 1:
            # Downmix to mono by averaging channels
            processed_data = np.mean(processed_data, axis=1, keepdims=True)
        elif target_channels == 2 and source_channels == 1:
            # Upmix mono to stereo by duplicating the channel
            processed_data = np.concatenate([processed_data, processed_data], axis=1)
        else:
            # For other conversions (e.g., 5.1 to stereo), a more complex
            # downmixing matrix would be needed. This is a placeholder.
            raise ValueError(
                f"Channel conversion from {source_channels} to {target_channels} is not supported."
            )

    # --- 3. Resampling ---
    if source_sr != target_sr:
        duration = processed_data.shape[0] / source_sr
        num_samples_out = int(duration * target_sr)
        # Use scipy.signal.resample for high-quality resampling
        processed_data = resample(processed_data, num_samples_out, axis=0)

    # --- 4. Data Type Conversion ---
    source_dtype = processed_data.dtype

    if source_dtype != target_dtype:
        # Check for float-to-int or int-to-float conversions to apply scaling
        is_source_float = np.issubdtype(source_dtype, np.floating)
        is_target_int = np.issubdtype(target_dtype, np.integer)

        if is_source_float and is_target_int:
            # Convert float [-1.0, 1.0] to integer
            # This is the most common conversion case
            max_val = np.iinfo(target_dtype).max
            processed_data = (processed_data * max_val).astype(target_dtype)
        elif not is_source_float and not is_target_int:
            # Float-to-float or int-to-int, just change type
            processed_data = processed_data.astype(target_dtype)
        else:
            # For int-to-float, we need to normalize
            # This is less common but important for correctness
            source_max_val = np.iinfo(source_dtype).max
            processed_data = processed_data.astype(target_dtype) / source_max_val

    # If the final output should be mono and is 2D, flatten it to 1D
    if target_channels == 1 and processed_data.ndim == 2:
        processed_data = processed_data.flatten()

    return processed_data
