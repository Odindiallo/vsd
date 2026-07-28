#!/usr/bin/env python3
"""
Build a fused audio+visual timeline from a video URL or local video file.

The output directory contains sensitive working data: the source URL in
MANIFEST.json, local/output paths, downloaded video, extracted audio, PNG
frames, OCR text, and the transcript. Use --cleanup to remove generated
video/audio/frames after timeline.json is written; timeline.json,
transcript.json (when available), and MANIFEST.json remain and still contain
extracted text and source metadata.

Pipeline:
  1. Acquire   : yt-dlp downloads low-res video and audio for URL inputs.
  2. Keyframes : ffmpeg scene detection with exact showinfo pts_time values.
  3. Dedupe    : perceptual hashes remove near-duplicate frames.
  4. Cap       : --max-frames evenly downsamples while retaining endpoints.
  5. OCR       : grayscale -> upscale -> Otsu -> tesseract (psm 11 default).
  6. Transcribe: faster-whisper emits timestamped transcript segments.
  7. Fuse      : frames and narration become timeline.json.

The script degrades gracefully and loudly when optional capabilities are
missing. Every subprocess is invoked with a list, never through a shell.

Usage:
  python extract_video.py <URL|path> [--out DIRECTORY] [--force] [--cleanup]
      [--max-filesize 500M] [--max-duration 7200] [--max-frames 150]
      [--scene-threshold 0.3] [--dedupe-threshold 6] [--ocr-psm 11]
      [--whisper-model small] [--process-timeout 1800]

When --out is omitted, a fresh timestamped directory is created. An explicitly
selected non-empty directory is refused unless --force is present.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_MAX_FILESIZE = "500M"
DEFAULT_MAX_DURATION = 7200.0
DEFAULT_PROCESS_TIMEOUT = 1800.0
COMMON_VIDEO_HOSTS = (
    "youtube.com",
    "youtu.be",
    "youtube-nocookie.com",
    "vimeo.com",
    "dailymotion.com",
    "twitch.tv",
    "tiktok.com",
    "facebook.com",
    "instagram.com",
    "x.com",
    "twitter.com",
)
SIZE_RE = re.compile(r"^(?P<number>[0-9]+(?:\.[0-9]+)?)(?P<unit>[KMGTPE]?)(?:i?B)?$", re.I)
PTS_RE = re.compile(
    r"pts_time:\s*([-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?)"
)


# ----------------------------- dependency checks ---------------------------- #

def have_cmd(name: str) -> bool:
    return shutil.which(name) is not None


def try_import(mod: str):
    try:
        return __import__(mod)
    except Exception:
        return None


PYTHON_INSTALL_HINT = "python -m pip install -r requirements.txt"
INSTALL_HINTS = {
    "yt-dlp": PYTHON_INSTALL_HINT,
    "ffmpeg": "install ffmpeg (brew install ffmpeg / apt install ffmpeg)",
    "faster_whisper": PYTHON_INSTALL_HINT,
    "pytesseract": PYTHON_INSTALL_HINT,
    "tesseract": "install tesseract-ocr (brew install tesseract / apt install tesseract-ocr)",
    "PIL": PYTHON_INSTALL_HINT,
    "imagehash": PYTHON_INSTALL_HINT,
}

# Collected during a run so the final summary can shout about degradation.
DEGRADED: list[str] = []


def degrade(msg: str):
    DEGRADED.append(msg)
    print(f"[!] DEGRADED: {msg}")


# ------------------------------ path safeguards ------------------------------ #

class UnsafeOutputError(ValueError):
    """Raised when an output write/removal would be unsafe."""


def _absolute(path: Path) -> Path:
    """Return an absolute lexical path without following symlinks."""
    return Path(os.path.abspath(os.fspath(path)))


def _assert_no_symlink_components(path: Path):
    """Reject any existing symlink in path or its parent chain."""
    absolute = _absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise UnsafeOutputError(f"refusing symlink path component: {current}")


def _assert_tree_has_no_symlinks(root: Path):
    if not root.exists():
        return
    for current, dirs, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in dirs + files:
            candidate = current_path / name
            if candidate.is_symlink():
                raise UnsafeOutputError(f"refusing symlink in output directory: {candidate}")


def _contained_path(path: Path, root: Path, *, allow_root: bool = False) -> Path:
    """Resolve and validate a path without permitting escape from root."""
    root_abs = _absolute(root)
    path_abs = _absolute(path)
    _assert_no_symlink_components(root_abs)
    _assert_no_symlink_components(path_abs)
    root_resolved = root_abs.resolve(strict=True)
    candidate = path_abs.resolve(strict=False)
    if candidate == root_resolved:
        if allow_root:
            return candidate
        raise UnsafeOutputError(f"refusing operation on output root itself: {candidate}")
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise UnsafeOutputError(
            f"refusing path outside output directory: {candidate}"
        ) from exc
    return candidate


def prepare_output_dir(out_arg: str | None, force: bool) -> Path:
    """Create a safe output directory and enforce explicit-output semantics."""
    if out_arg is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        candidate = Path.cwd() / f"vsd_work-{stamp}"
        explicit = False
    else:
        candidate = Path(out_arg).expanduser()
        explicit = True

    candidate = _absolute(candidate)
    _assert_no_symlink_components(candidate)
    if candidate.exists() and not candidate.is_dir():
        raise UnsafeOutputError(f"output path is not a directory: {candidate}")
    if explicit and candidate.exists() and any(candidate.iterdir()) and not force:
        raise UnsafeOutputError(
            f"output directory is not empty: {candidate}; pass --force to reuse it"
        )
    if not explicit and candidate.exists():
        raise UnsafeOutputError(f"generated output directory already exists: {candidate}")

    candidate.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(candidate)
    _assert_tree_has_no_symlinks(candidate)
    return candidate.resolve(strict=True)


def safe_make_dir(path: Path, root: Path) -> Path:
    candidate = _contained_path(path, root)
    candidate.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(candidate)
    if not candidate.is_dir():
        raise UnsafeOutputError(f"expected a directory: {candidate}")
    return candidate


def safe_write_text(path: Path, text: str, root: Path):
    """Write text under root while refusing symlink targets."""
    candidate = _contained_path(path, root)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(candidate.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(candidate, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)


def safe_write_json(path: Path, value, root: Path):
    safe_write_text(path, json.dumps(value, indent=2), root)


def safe_remove(path: Path, root: Path):
    """Remove one file only after containment and symlink validation."""
    candidate = _contained_path(path, root)
    if candidate.is_symlink():
        raise UnsafeOutputError(f"refusing to remove symlink: {candidate}")
    if not candidate.exists():
        return
    if not candidate.is_file():
        raise UnsafeOutputError(f"refusing non-file removal: {candidate}")
    os.remove(candidate)


def safe_remove_tree(path: Path, root: Path):
    """Remove a generated directory without following symlinks."""
    candidate = _contained_path(path, root)
    if not candidate.exists():
        return
    if candidate.is_symlink() or not candidate.is_dir():
        raise UnsafeOutputError(f"refusing unsafe directory removal: {candidate}")
    _assert_tree_has_no_symlinks(candidate)
    for current, dirs, files in os.walk(candidate, topdown=False, followlinks=False):
        current_path = Path(current)
        for name in files:
            safe_remove(current_path / name, root)
        for name in dirs:
            directory = _contained_path(current_path / name, root)
            os.rmdir(directory)
    os.rmdir(candidate)


def cleanup_selection(out: Path) -> list[Path]:
    """Return generated intermediate media paths selected by --cleanup."""
    selected: set[Path] = set()
    for pattern in ("video.*", "audio.*", "audio_limited.*"):
        selected.update(out.glob(pattern))
    frames = out / "frames"
    if frames.exists() or frames.is_symlink():
        selected.add(frames)
    return sorted(selected, key=lambda path: path.name)


def reset_selection(out: Path) -> list[Path]:
    """Return all generated paths reset when --force reuses an output."""
    selected = set(cleanup_selection(out))
    for name in ("timeline.json", "transcript.json", "MANIFEST.json"):
        path = out / name
        if path.exists() or path.is_symlink():
            selected.add(path)
    return sorted(selected, key=lambda path: path.name)


def _remove_selection(out: Path, selected: list[Path]):
    _assert_tree_has_no_symlinks(out)
    for path in selected:
        if path.is_dir():
            safe_remove_tree(path, out)
        else:
            safe_remove(path, out)


def remove_generated_intermediates(out: Path):
    """Remove only generated media namespaces, preserving retained artifacts."""
    _remove_selection(out, cleanup_selection(out))


def reset_generated_outputs(out: Path):
    """Reset generated artifacts before an explicitly forced run."""
    _remove_selection(out, reset_selection(out))


# ------------------------------- acquisition -------------------------------- #

def is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def is_common_video_host(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    return any(host == known or host.endswith(f".{known}") for known in COMMON_VIDEO_HOSTS)


def warn_for_uncommon_host(url: str) -> bool:
    """Warn, but never reject, a valid uncommon video host."""
    if is_common_video_host(url):
        return False
    host = urlparse(url).hostname or "unknown host"
    print(
        f"[!] WARNING: {host} is not recognized as a common video host. "
        "Continuing with the configured size, duration, and timeout limits."
    )
    return True


def parse_filesize(value: str) -> int:
    """Convert a yt-dlp-style size such as 500M to bytes."""
    match = SIZE_RE.fullmatch(value.strip())
    if not match:
        raise ValueError("expected a size such as 500M, 2G, or 750MiB")
    number = float(match.group("number"))
    unit = match.group("unit").upper()
    power = "KMGTPE".find(unit) + 1 if unit else 0
    return int(number * (1024 ** power))


def filesize_arg(value: str) -> str:
    try:
        parse_filesize(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return value


def _format_seconds(value: float) -> str:
    return f"{value:g}"


def download(url: str, out: Path, max_filesize: str, max_duration: float,
             timeout: float):
    """Download capped low-res video and audio streams via yt-dlp."""
    if not have_cmd("yt-dlp"):
        raise RuntimeError(
            "yt-dlp is required to process a URL but was not found.\n"
            f"    Install: {INSTALL_HINTS['yt-dlp']}\n"
            "    In a network-restricted environment, download the video "
            "elsewhere and pass the local file path instead."
        )
    warn_for_uncommon_host(url)
    video_tmpl = str(_contained_path(out / "video.%(ext)s", out))
    audio_tmpl = str(_contained_path(out / "audio.%(ext)s", out))
    duration_filter = f"duration <=? {_format_seconds(max_duration)}"
    common = [
        "--no-playlist",
        "--max-filesize", max_filesize,
        "--match-filter", duration_filter,
    ]
    try:
        subprocess.run(
            ["yt-dlp", *common, "-f",
             "worstvideo[height>=360]/worstvideo/worst/best",
             "-o", video_tmpl, url],
            check=True,
            timeout=timeout,
        )
        subprocess.run(
            ["yt-dlp", *common, "-f", "bestaudio/best", "-x",
             "--audio-format", "mp3", "-o", audio_tmpl, url],
            check=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"yt-dlp exceeded the {timeout:g}-second process timeout"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "yt-dlp failed. The video may exceed --max-filesize or "
            "--max-duration, require login, be unavailable, or the environment "
            f"may lack network access. Underlying error: {exc}"
        ) from exc

    _assert_tree_has_no_symlinks(out)
    video = next((path for path in out.glob("video.*") if path.is_file()), None)
    audio = next((path for path in out.glob("audio.*") if path.is_file()), None)
    return video, audio


def extract_audio_from_file(video: Path, out: Path, max_duration: float,
                            timeout: float):
    if not have_cmd("ffmpeg"):
        degrade("ffmpeg missing - local audio extraction skipped.")
        return None
    audio = _contained_path(out / "audio.mp3", out)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video), "-t",
             _format_seconds(max_duration), "-vn", "-acodec", "libmp3lame",
             str(audio)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        degrade(f"audio extraction failed or timed out - no transcript ({exc}).")
        return None
    return audio


def trim_downloaded_audio(audio: Path | None, out: Path, max_duration: float,
                          timeout: float):
    """Enforce the duration cap before downloaded audio reaches Whisper."""
    if audio is None:
        return None
    if not have_cmd("ffmpeg"):
        degrade("ffmpeg missing - downloaded audio cannot be duration-limited; "
                "transcription skipped.")
        return None
    limited = _contained_path(out / "audio_limited.mp3", out)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio), "-t",
             _format_seconds(max_duration), "-vn", "-acodec", "libmp3lame",
             str(limited)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        degrade(f"audio duration limiting failed - no transcript ({exc}).")
        return None
    return limited


# -------------------------------- keyframes --------------------------------- #

def _run_select(video: Path, out_pattern: str, vf: str, max_duration: float,
                timeout: float, label: str) -> str:
    """Run ffmpeg select+showinfo; stderr contains exact pts_time lines."""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", str(video), "-t",
             _format_seconds(max_duration), "-vf", vf, "-vsync", "vfr",
             out_pattern],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        degrade(f"frame extraction exceeded the {timeout:g}-second timeout.")
        stderr = exc.stderr or ""
        return stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
    if proc.returncode != 0:
        detail = " ".join((proc.stderr or "").split())[:240]
        suffix = f" stderr: {detail}" if detail else ""
        degrade(
            f"{label} frame extraction failed - ffmpeg exited "
            f"{proc.returncode}.{suffix}"
        )
    return proc.stderr or ""


def _parse_pts(stderr: str) -> list[float]:
    """Parse exact showinfo pts_time values in emitted-frame order."""
    return [float(value) for value in PTS_RE.findall(stderr)]


def extract_keyframes(video: Path, frames_dir: Path, scene_thr: float,
                      fps: float, max_duration: float, timeout: float,
                      output_root: Path):
    """Extract scene-cut frames; use fps fallback only when none are emitted."""
    frames_dir = safe_make_dir(frames_dir, output_root)
    if not have_cmd("ffmpeg"):
        degrade("ffmpeg missing - no visual track (frames/OCR skipped).")
        return []

    stderr = _run_select(
        video,
        str(_contained_path(frames_dir / "scene_%04d.png", output_root)),
        f"select='gt(scene,{scene_thr})',showinfo",
        max_duration,
        timeout,
        "scene",
    )
    _assert_tree_has_no_symlinks(frames_dir)
    frames = sorted(frames_dir.glob("scene_*.png"))
    times = _parse_pts(stderr)

    # Existing fallback: only sample by fps when scene detection emits no frames.
    if not frames:
        stderr = _run_select(
            video,
            str(_contained_path(frames_dir / "fps_%04d.png", output_root)),
            f"fps={fps},showinfo",
            max_duration,
            timeout,
            "fps fallback",
        )
        _assert_tree_has_no_symlinks(frames_dir)
        frames = sorted(frames_dir.glob("fps_*.png"))
        times = _parse_pts(stderr)

    # Use approximate spacing only for the existing count-mismatch fallback.
    stamped = []
    if len(times) == len(frames) and frames:
        stamped = [
            {"path": str(frame), "time": round(timestamp, 2)}
            for frame, timestamp in zip(frames, times)
        ]
    else:
        duration = probe_duration(video, timeout) or float(len(frames))
        duration = min(duration, max_duration)
        count = len(frames)
        for index, frame in enumerate(frames):
            timestamp = (duration * index / count) if count else float(index)
            stamped.append({"path": str(frame), "time": round(timestamp, 2)})
        if frames:
            degrade("frame/timestamp count mismatch - used approximate "
                    "even-spacing for frame times.")
    return stamped


def probe_duration(video: Path, timeout: float):
    if not have_cmd("ffprobe"):
        return None
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def dedupe_frames(frames, threshold: int, output_root: Path):
    """Drop perceptually near-identical consecutive frames."""
    imagehash = try_import("imagehash")
    pillow = try_import("PIL")
    if not imagehash or not pillow:
        degrade("imagehash/pillow missing - keeping all frames (no dedupe).")
        return frames
    from PIL import Image
    kept, previous = [], None
    for frame in frames:
        try:
            image_hash = imagehash.phash(Image.open(frame["path"]))
        except Exception:
            kept.append(frame)
            continue
        if previous is None or (image_hash - previous) > threshold:
            kept.append(frame)
            previous = image_hash
        else:
            safe_remove(Path(frame["path"]), output_root)
    return kept


def cap_frames(frames, max_frames: int, output_root: Path):
    """Evenly downsample to at most max_frames, retaining first and last."""
    if max_frames <= 0 or len(frames) <= max_frames:
        return frames
    if max_frames == 1:
        keep = [frames[0]]
    elif max_frames == 2:
        keep = [frames[0], frames[-1]]
    else:
        keep = [frames[0]]
        middle = max_frames - 2
        step = (len(frames) - 2) / (middle + 1)
        for index in range(1, middle + 1):
            keep.append(frames[1 + int(index * step)])
        keep.append(frames[-1])
    keep_paths = {frame["path"] for frame in keep}
    for frame in frames:
        if frame["path"] not in keep_paths:
            safe_remove(Path(frame["path"]), output_root)
    print(f"[i] capped {len(frames)} -> {len(keep)} frames (--max-frames)")
    return keep


# ----------------------------------- OCR ------------------------------------ #

def _otsu_threshold(gray) -> int:
    """Compute a Pure-Python Otsu threshold from a grayscale histogram."""
    histogram = gray.histogram()[:256]
    total = sum(histogram)
    if total == 0:
        return 128
    sum_all = sum(index * histogram[index] for index in range(256))
    background_weight = 0
    background_sum = 0.0
    best_between = -1.0
    threshold = 128
    for index in range(256):
        background_weight += histogram[index]
        if background_weight == 0:
            continue
        foreground_weight = total - background_weight
        if foreground_weight == 0:
            break
        background_sum += index * histogram[index]
        background_mean = background_sum / background_weight
        foreground_mean = (sum_all - background_sum) / foreground_weight
        between = (
            background_weight
            * foreground_weight
            * (background_mean - foreground_mean) ** 2
        )
        if between >= best_between:
            best_between = between
            threshold = index
    return threshold


def _prep(img):
    """Preserve OCR preprocessing: grayscale -> 2x -> contrast -> Otsu."""
    from PIL import Image, ImageOps
    gray = ImageOps.grayscale(img)
    gray = gray.resize((gray.width * 2, gray.height * 2), Image.LANCZOS)
    gray = ImageOps.autocontrast(gray)
    threshold = _otsu_threshold(gray)
    return gray.point(lambda pixel: 255 if pixel > threshold else 0)


def ocr_frames(frames, psm: int):
    if not frames:
        return frames, "no frames"
    pytesseract = try_import("pytesseract")
    pillow = try_import("PIL")
    if not pytesseract or not pillow or not have_cmd("tesseract"):
        degrade("OCR unavailable - on_screen_text will be EMPTY. "
                f"({INSTALL_HINTS['tesseract']})")
        for frame in frames:
            frame["on_screen_text"] = ""
        return frames, "unavailable"
    from PIL import Image
    config = f"--oem 3 --psm {psm}"
    failures = 0
    first_error = None
    for frame in frames:
        text = ""
        try:
            image = Image.open(frame["path"])
            text = pytesseract.image_to_string(_prep(image), config=config)
            if not text.strip():
                from PIL import ImageOps
                text = pytesseract.image_to_string(
                    ImageOps.grayscale(image), config=config
                )
        except Exception as exc:
            failures += 1
            if first_error is None:
                first_error = exc
            text = ""
        frame["on_screen_text"] = " ".join(text.split())
    if failures:
        degrade(
            f"OCR failed for {failures} frame(s) - affected on_screen_text "
            f"entries are EMPTY. ({first_error})"
        )
        return frames, f"degraded ({failures} frame failures)"
    return frames, "done"


# ------------------------------- transcription ------------------------------ #

def transcribe(audio: Path | None, out: Path, model_name: str):
    faster_whisper = try_import("faster_whisper")
    if not faster_whisper or audio is None:
        degrade("faster-whisper or duration-limited audio missing - NO "
                "transcript (narration will be empty). "
                f"({INSTALL_HINTS['faster_whisper']})")
        return []
    from faster_whisper import WhisperModel
    try:
        model = WhisperModel(model_name, device="auto", compute_type="int8")
        segments, _ = model.transcribe(str(audio), vad_filter=True)
        result = [
            {
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip(),
            }
            for segment in segments
        ]
    except Exception as exc:
        degrade(f"transcription failed - NO transcript (narration will be "
                f"empty). ({exc})")
        return []
    safe_write_json(out / "transcript.json", result, out)
    return result


# ---------------------------------- fusion ---------------------------------- #

def fuse(frames, transcript):
    """Attach the transcript segment covering each frame timestamp."""
    def narration_at(timestamp):
        for index, segment in enumerate(transcript):
            if segment["start"] <= timestamp <= segment["end"]:
                return index, segment["text"], True
        if transcript:
            index, segment = min(
                enumerate(transcript),
                key=lambda item: abs(item[1]["start"] - timestamp),
            )
            return index, segment["text"], False
        return None, "", False

    timeline = []
    matched_segments = set()
    for frame_index, frame in enumerate(frames):
        segment_index, narration, covered = narration_at(frame["time"])
        if covered:
            matched_segments.add(segment_index)
        timeline.append({
            "time": frame["time"],
            "narration": narration,
            "on_screen_text": frame.get("on_screen_text", ""),
            "frame": os.path.basename(frame["path"]),
            "_sort": (frame["time"], 0, frame_index),
        })
    for segment_index, segment in enumerate(transcript):
        if segment_index in matched_segments:
            continue
        timeline.append({
            "time": segment["start"],
            "narration": segment["text"],
            "on_screen_text": "",
            "frame": None,
            "_sort": (segment["start"], 1, segment_index),
        })
    timeline.sort(key=lambda event: event["_sort"])
    for event in timeline:
        del event["_sort"]
    return timeline


def manifest_settings(args) -> dict:
    """Return stable settings recorded in MANIFEST.json."""
    return {
        "scene_threshold": args.scene_threshold,
        "fps_fallback": args.fps,
        "max_frames": args.max_frames,
        "dedupe_threshold": args.dedupe_threshold,
        "ocr_psm": args.ocr_psm,
        "whisper_model": args.whisper_model,
        "max_filesize": args.max_filesize,
        "max_duration": args.max_duration,
        "cleanup": args.cleanup,
        "force": args.force,
        "process_timeout": args.process_timeout,
    }


# ----------------------------------- main ----------------------------------- #

def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Fused audio+visual timeline builder")
    parser.add_argument("source", help="video URL or local file path")
    parser.add_argument(
        "--out",
        default=None,
        help="output directory (default: fresh timestamped ./vsd_work-*)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="permit reuse of an explicitly selected non-empty output directory",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="remove generated video/audio/frames after timeline creation",
    )
    parser.add_argument(
        "--max-filesize",
        type=filesize_arg,
        default=DEFAULT_MAX_FILESIZE,
        help="maximum downloaded/local input size (default: 500M)",
    )
    parser.add_argument(
        "--max-duration",
        type=positive_float,
        default=DEFAULT_MAX_DURATION,
        help="maximum seconds processed; known longer URLs are skipped (default: 7200)",
    )
    parser.add_argument(
        "--process-timeout",
        type=positive_float,
        default=DEFAULT_PROCESS_TIMEOUT,
        help="timeout in seconds for each downloader/ffmpeg/ffprobe process (default: 1800)",
    )
    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=0.3,
        help="ffmpeg scene-cut sensitivity: 0.25-0.3 fast-cut, 0.45-0.5 hard cuts",
    )
    parser.add_argument(
        "--fps",
        type=positive_float,
        default=1.0,
        help="fallback sampling rate if scene detection finds nothing",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=150,
        help="cap on keyframes after dedupe (0 = no cap)",
    )
    parser.add_argument(
        "--dedupe-threshold",
        type=int,
        default=6,
        help="perceptual-hash distance; higher merges more similar frames",
    )
    parser.add_argument(
        "--ocr-psm",
        type=int,
        default=11,
        help="tesseract mode: 11=sparse text, 6=uniform block, 3=auto",
    )
    parser.add_argument(
        "--whisper-model",
        default="small",
        help="faster-whisper model size (tiny/base/small/medium/large)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    DEGRADED.clear()
    args = parse_args(argv)
    try:
        out = prepare_output_dir(args.out, args.force)
        if args.force:
            reset_generated_outputs(out)
        frames_dir = safe_make_dir(out / "frames", out)
    except (OSError, UnsafeOutputError) as exc:
        sys.exit(f"[x] Unsafe output directory: {exc}")

    steps = {}
    if is_url(args.source):
        try:
            video, downloaded_audio = download(
                args.source,
                out,
                args.max_filesize,
                args.max_duration,
                args.process_timeout,
            )
        except RuntimeError as exc:
            sys.exit(f"[x] {exc}")
        audio = trim_downloaded_audio(
            downloaded_audio, out, args.max_duration, args.process_timeout
        )
        steps["acquire"] = "downloaded via yt-dlp"
    else:
        # Local inputs may be symlinks because they are read-only sources.
        # Symlink refusal applies to every output write and removal instead.
        video = Path(args.source).expanduser().resolve(strict=False)
        if not video.is_file():
            sys.exit(f"[x] File not found: {video}")
        try:
            video.relative_to(out)
        except ValueError:
            pass
        else:
            sys.exit("[x] Local input must be outside the output directory")
        if video.stat().st_size > parse_filesize(args.max_filesize):
            sys.exit(
                f"[x] Local input exceeds --max-filesize {args.max_filesize}: {video}"
            )
        audio = extract_audio_from_file(
            video, out, args.max_duration, args.process_timeout
        )
        steps["acquire"] = "local file"

    frames = (
        extract_keyframes(
            video,
            frames_dir,
            args.scene_threshold,
            args.fps,
            args.max_duration,
            args.process_timeout,
            out,
        )
        if video
        else []
    )
    frames = dedupe_frames(frames, args.dedupe_threshold, out)
    frames = cap_frames(frames, args.max_frames, out)
    steps["keyframes"] = f"{len(frames)} kept"

    frames, ocr_status = ocr_frames(frames, args.ocr_psm)
    steps["ocr"] = ocr_status

    transcript = transcribe(audio, out, args.whisper_model)
    steps["transcribe"] = f"{len(transcript)} segments"

    timeline = fuse(frames, transcript)
    safe_write_json(out / "timeline.json", timeline, out)
    steps["fuse"] = f"{len(timeline)} records"

    if args.cleanup:
        remove_generated_intermediates(out)
        steps["cleanup"] = "removed generated video/audio/frames"

    manifest = {
        "source": args.source,
        "steps": steps,
        "degraded": DEGRADED,
        "settings": manifest_settings(args),
        "outputs": {
            "timeline": str(out / "timeline.json"),
            "transcript": str(out / "transcript.json"),
            "frames_dir": None if args.cleanup else str(frames_dir),
        },
    }
    safe_write_json(out / "MANIFEST.json", manifest, out)

    print("\n=== done ===")
    for name, value in steps.items():
        print(f"  {name:11s}: {value}")
    if DEGRADED:
        print("\n  ⚠️  RESULT IS DEGRADED - some steps did not fully run:")
        for detail in DEGRADED:
            print(f"      - {detail}")
        print("  The doc built from this will be weaker; install the missing "
              "tools above for a full multimodal pass.")
    print(f"\nKey artifact -> {out / 'timeline.json'}")
    if args.cleanup:
        print("Intermediate video, audio, and frames were removed (--cleanup).")
    else:
        print("Feed timeline.json and selected frames from frames/ back to the "
              "/vsd skill to write the architecture doc.")


if __name__ == "__main__":
    main()
