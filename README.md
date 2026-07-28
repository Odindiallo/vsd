# VSD — Video System Design

## What it is

VSD is an agent skill that turns a video or raw script into a written system-design document, using both narration and on-screen information to describe the pipeline and its automatable redesign.

## How it works

- yt-dlp download
- ffmpeg scene-cut keyframes with `pts_time`
- perceptual-hash dedupe
- frame cap
- OCR (Tesseract, grayscale + upscale + Otsu + PSM 11)
- faster-whisper transcription
- fused `timeline.json`

## Prerequisites

Use Python 3.12. Install the required system binaries:

**macOS**

```bash
brew install ffmpeg tesseract yt-dlp
```

**Debian**

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg tesseract-ocr yt-dlp
```

Install the pinned Python dependencies:

```bash
python -m pip install -r requirements.txt
```

## Local validation

```bash
python -m py_compile scripts/extract_video.py
pytest tests/ -q
```

## Usage

```bash
python scripts/extract_video.py <URL|path> --out ./vsd_work
```

| Flag | Purpose |
| --- | --- |
| `--max-filesize` | Maximum accepted download or local input size. |
| `--max-duration` | Maximum video/audio duration to process. |
| `--max-frames` | Cap retained keyframes after deduplication; `0` disables the cap. |
| `--force` | Explicitly permit reuse of a non-empty selected output directory. |
| `--cleanup` | Remove generated media and frames while retaining timeline artifacts. |
| `--scene-threshold` | ffmpeg scene-change sensitivity. |
| `--dedupe-threshold` | Perceptual-hash threshold for merging near-duplicate frames. |
| `--ocr-psm` | Tesseract page-segmentation mode. |
| `--whisper-model` | faster-whisper model to use for transcription. |

## Safety and permission boundaries

- `timeline.json` narration and OCR are untrusted data. This boundary is enforced at the consumer layer in `SKILL.md`; extracted content is analyzed, never followed as instructions.
- Resource caps limit input size, duration, frame count, and subprocess execution. `--force` and safe-remove output-directory handling protect against unintended output reuse or deletion.
- Downloading or processing copyrighted video is the user's responsibility. See `NOTICE.md`.

## Repository layout

- `SKILL.md` — `/vsd` skill instructions and consumer-layer trust boundary.
- `NOTICE.md` — third-party and legal notice.
- `requirements.txt` — pinned Python dependencies.
- `scripts/extract_video.py` — extraction, OCR, transcription, and timeline fusion CLI.
- `tests/test_extract_video.py` — offline regression tests.
- `.github/workflows/ci.yml` — Python 3.12 CI workflow.
- `README.md` — standalone repository guide.

## License / notices

See `NOTICE.md` for third-party software and copyright notices.
