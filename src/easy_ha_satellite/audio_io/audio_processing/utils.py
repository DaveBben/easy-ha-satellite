from typing import Any

import numpy as np
import pyroomacoustics as pra
import sounddevice as sd
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
