# Smart Behavioral Video Compression

Sentio Mind POC Assignment - Project 2

This repository contains a production-style `solution.py` for intelligent CCTV compression.
It preserves behaviorally important frames while aggressively removing redundant footage.

## Goal

Reduce raw CCTV upload size while preserving frames that may contain meaningful human activity.

Required outputs:

1. `compressed_output.mp4` - H.264 MP4 at 12 fps
2. `compression_report.html` - offline report with metrics and storyboard thumbnails
3. `segments_kept.json` - machine-readable metadata for kept/discarded frame logic

## Assignment Algorithm (Implemented Order)

For each processed frame:

1. **pHash similarity check**
If the similarity with the last kept frame is `> 0.95`, discard it as a near-duplicate.

2. **Optical flow motion score**
If the motion score is `< 0.05`, mark it as a low-motion discard candidate.

3. **Haar face detection override**
If a low-motion frame contains a face, keep it to ensure behavioral events are captured.

4. **Context frame rule**
If a frame is still discardable (no motion, no face), force keeping one context frame at least every 3 seconds.

5. **Re-encode kept frames**
Encode all kept frames sequentially into an H.264 MP4 at 12 fps.

## Performance Strategy

The pipeline is optimized for varying conditions using two distinct modes:

- **Fast mode (default):** Processes every Nth frame (`--frame-step`, default `6`) and uses a heavily downscaled compute resolution (`--compute-scale`, default `0.15`). It aggressively skips expensive operations like running Haar cascades unless motion drops below `0.08`, making it exceptionally fast for mostly static inputs.
- **Strict mode:** A full-frame pass (`--strict`) effectively enforcing `frame-step=1`. It guarantees a comprehensive analysis of every single frame.

This split design allows you to seamlessly slide between extreme speed and forensic strictness as the testing context requires.

## Files

- `solution.py` - full pipeline implementation handling video I/O, scoring, discarding, and reporting.
- `demo.mp4` / `video_sample_1.mov` - local sample input videos available in the directory.
- `Sentio_Assignement_Video_compression.pdf` - assignment prompt (if provided in your context).

## Environment Setup

### 1. Python Environment (3.9+)

Install necessary Python dependencies:

```bash
pip install opencv-python numpy
```

### 2. FFmpeg Installation (Required for optimal H.264 compression)

- **Windows:** Install FFmpeg and ensure it's successfully added to your system `PATH`.
- **Ubuntu:** Run `sudo apt install ffmpeg`

## Usage

### Default (Fast Mode)

```bash
python solution.py --input demo.mp4
```

### Strict Full-Frame Mode

```bash
python solution.py --input demo.mp4 --strict
```

### Custom Tuning

```bash
python solution.py --input demo.mp4 --frame-step 4 --compute-scale 0.2
```

## Output Schema Breakdown

`segments_kept.json` stores analytical and serialization details such as:

- Overall metrics (size/duration/fps summaries, reduction percentage).
- Processing metadata (`frame_step`, overall processed count, timing/elapsed data).
- The dynamic kept segment list, which aggregates individual frame events into grouped blocks. For each segment:
  - `segment_id`, `start_frame`, `end_frame`, `start_sec`, `end_sec`
  - `frames_kept`, `reason_kept`
  - `face_detected`, `motion_score_avg`
- Discard trackers quantifying exactly why frames were dropped:
  - `near_duplicate_phash`
  - `low_motion_no_face`
  - `total_discarded`

*Note: If testing metrics require a slightly different shape, you can manipulate the final JSON dictionary block in `extract_intelligent_frames` without unspooling the structural analysis.*

## HTML Report Details

The generated `compression_report.html` is completely standalone and zero-dependency (offline). It visually summarizes:

- Original vs compressed filesize impacts.
- Aggregate frame discard percentages and segment counts.
- Dynamic **Storyboard Cards** utilizing inline Base64 thumbnails to instantly prove *why* the algorithm retained specific segment jumps (e.g., face detection kicks vs motion thresholding).

## Engineering Notes & Technical Choices

- **Optical Flow Strategy:** Employs OpenCV's Farneback flow algorithm exclusively on heavily resized grayscale layers to cut floating point overhead while preserving general layout motion magnitudes.
- **Face Detection:** Haarcascade (`haarcascade_frontalface_default.xml`) offers the most reliable, dependency-free bounding-box approach natively available inside headless `opencv-python`.
- **Duplicate Suppression:** Instead of expensive pixel diffing or MSE, `pHash` (a 64-bit structural hash representation of the image) uses simple bitwise Hamming distance to spot sequential duplication regardless of mild compression noise.
- **Codec Pass-through:** Orchestrates raw video frames straight into an `ffmpeg` subprocess using the `libx264` codec, constrained dynamically by CRF `28` and the `veryfast` preset for aggressive, visual-first compression.
- **Fallback Recovery:** Auto-detects missing FFmpeg environments and reverts to OpenCV's native `VideoWriter` as an unoptimal temporary `.tmp.avi` output string instead of crashing.

## Quick Validation Checklist

- Output `.mp4` video plays correctly in standard players like VLC without stuttering or dropping keyframes.
- The `compression_report.html` opens cleanly in modern browsers, even offline, with fully functioning CSS gradients and images.
- The output `json` correctly aggregates adjacent frame hits into cohesive readable segments.
- Running on standard CCTV test cases should exceed 70% raw size reduction smoothly.
