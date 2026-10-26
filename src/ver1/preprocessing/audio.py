"""Bounded audio decoding and the shared sample-accurate window generator."""

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr


@dataclass
class Window:
    start: int
    audio: np.ndarray  # Real samples only, before gain and padding.
    left: int
    right: int


def decoded_blocks(path, config):
    """Native decoding with stateful resampling; FFmpeg handles other codecs."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    rate, size = config["sample_rate"], config["decode_block_samples"]
    try:
        source = sf.SoundFile(path)
    except (RuntimeError, sf.LibsndfileError):
        source = None
    if source is not None:
        with source:
            resampler = soxr.ResampleStream(source.samplerate, rate, 1, dtype="float32", quality="HQ")
            try:
                for block in source.blocks(blocksize=size, dtype="float32", always_2d=True):
                    block = block.mean(axis=1)
                    result = resampler.resample_chunk(block)
                    if len(result):
                        yield result
                tail = resampler.resample_chunk(np.empty(0, np.float32), last=True)
                if len(tail):
                    yield tail
            except (RuntimeError, sf.LibsndfileError) as error:
                raise ValueError(f"Audio decode failed for {path}: {error}") from error
        return
    if not shutil.which("ffmpeg"):
        raise ValueError(f"Cannot decode {path}; install FFmpeg for this format")
    # Match arithmetic channel averaging, rather than FFmpeg's weighted downmix.
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                            "-show_entries", "stream=channels", "-of", "csv=p=0", str(path)],
                           capture_output=True, text=True, check=False)
    try:
        channels = int(probe.stdout.strip())
        if probe.returncode or channels < 1:
            raise ValueError
    except ValueError as error:
        raise ValueError(f"Cannot inspect audio in {path}: {probe.stderr.strip()}") from error
    downmix = "pan=mono|c0=" + "+".join(f"{1/channels}*c{i}" for i in range(channels))
    command = ["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-map", "0:a:0",
               "-af", downmix, "-ar", str(rate), "-f", "f32le", "pipe:1"]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        try:
            while block := process.stdout.read(size * 4):
                if len(block) % 4:
                    raise ValueError(f"Incomplete decoded sample in {path}")
                yield np.frombuffer(block, dtype="<f4").copy()
            if process.wait():
                errors.seek(0)
                raise ValueError(f"Audio decode failed for {path}: {errors.read().decode(errors='replace')}")
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
            process.wait()


def windows(blocks, config):
    """Stop at the first window covering EOF, including exact-boundary EOF."""
    rate = config["sample_rate"]
    length = round(rate * config["window_seconds"])
    stride = round(rate * config["stride_seconds"])
    start = round(rate * config["first_start_seconds"])
    buffer = np.empty(0, np.float32)
    offset = total = 0
    last_end = -1
    for block in blocks:
        if not np.isfinite(block).all():
            raise ValueError("Decoded audio contains nonfinite samples")
        buffer = np.concatenate((buffer, block))
        total += len(block)
        while start + length <= total:
            left = max(0, -start)
            yield Window(start, buffer[max(0, start) - offset:start + length - offset].copy(), left, 0)
            last_end = start + length
            start += stride
            keep = min(total, max(0, start))
            buffer = buffer[keep - offset:]
            offset = keep
    if total == 0:
        raise ValueError("Audio contains no samples")
    if last_end < total:
        yield Window(start, buffer[max(0, start) - offset:].copy(), max(0, -start), start + length - total)
