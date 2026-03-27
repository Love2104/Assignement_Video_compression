"""
Sentio Mind - Project 2 - Smart Behavioral Video Compression.

Required public functions preserved:
- compute_phash
- phash_similarity
- compute_motion_score
- has_face
- should_keep_frame
- frame_to_b64_thumb
- write_frames_to_video
- generate_compression_report
- extract_intelligent_frames
"""

import argparse
import base64
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np


VIDEO_IN_DEFAULT = Path("video_sample_1.mov")
VIDEO_OUT = Path("compressed_output.mp4")
REPORT_HTML_OUT = Path("compression_report.html")
SEGMENTS_JSON_OUT = Path("segments_kept.json")

PHASH_THRESHOLD = 0.95
MOTION_DISCARD_THRESH = 0.05
CONTEXT_EVERY_SEC = 3.0
OUTPUT_FPS = 12
OUTPUT_CRF = 28

# Fast defaults (can be overridden from CLI)
DEFAULT_FRAME_STEP = 6
DEFAULT_COMPUTE_SCALE = 0.15
DEFAULT_FACE_CHECK_MOTION_THRESH = 0.08

# Report configuration
SEGMENT_GAP_SEC = 1.2
REPORT_THUMB_WIDTH = 240
MAX_REPORT_SEGMENTS = 20


def compute_phash(gray_small: np.ndarray) -> int:
    """Compute a 64-bit average hash and return as integer."""
    reduced = cv2.resize(gray_small, (8, 8), interpolation=cv2.INTER_AREA)
    avg = float(reduced.mean())
    bits = (reduced > avg).astype(np.uint8).reshape(-1)

    h = 0
    for bit in bits:
        h = (h << 1) | int(bit)
    return h


def phash_similarity(h1: int, h2: int) -> float:
    """Return normalized similarity in [0,1] for two 64-bit hashes."""
    if h1 is None or h2 is None:
        return 0.0
    distance = (h1 ^ h2).bit_count()
    return 1.0 - (distance / 64.0)


def compute_motion_score(prev_small: np.ndarray, curr_small: np.ndarray) -> float:
    """Dense optical flow motion score using Farneback on small grayscale frames."""
    if prev_small is None:
        return 0.0

    flow = cv2.calcOpticalFlowFarneback(
        prev_small,
        curr_small,
        None,
        pyr_scale=0.5,
        levels=1,
        winsize=7,
        iterations=1,
        poly_n=5,
        poly_sigma=1.1,
        flags=0,
    )
    mag = np.abs(flow[..., 0]) + np.abs(flow[..., 1])
    return float(np.mean(mag))


def has_face(gray_face: np.ndarray, cascade) -> bool:
    """Return True if Haar cascade detects at least one face."""
    if cascade is None or cascade.empty():
        return False

    faces = cascade.detectMultiScale(
        gray_face,
        scaleFactor=1.2,
        minNeighbors=3,
        minSize=(12, 12),
    )
    return len(faces) > 0


def should_keep_frame(
    curr_hash: int,
    prev_kept_hash: int,
    motion: float,
    gray_face: np.ndarray,
    last_kept_time_sec: float,
    current_time_sec: float,
    cascade,
) -> tuple:
    """
    Required algorithm order:
    Step 1: pHash duplicate drop (>0.95 similarity).
    Step 2: motion < 0.05 becomes discard candidate.
    Step 3: face detection overrides low-motion discard.
    Step 4: keep one context frame every 3 seconds.
    Step 5: encoding is done later.

    Returns (keep, reason, face_found).
    """
    similarity = phash_similarity(curr_hash, prev_kept_hash) if prev_kept_hash is not None else 0.0

    # Step 1
    if similarity > PHASH_THRESHOLD:
        return False, "discarded_duplicate", False

    # Step 2
    is_static = motion < MOTION_DISCARD_THRESH
    if not is_static:
        return True, "motion_above_threshold", False

    # Step 3
    face_found = has_face(gray_face, cascade)
    if face_found:
        return True, "face_detected", True

    # Step 4
    if (current_time_sec - last_kept_time_sec) >= CONTEXT_EVERY_SEC:
        return True, "context_frame", False

    return False, "discarded_static", False


def frame_to_b64_thumb(frame: np.ndarray, width: int = REPORT_THUMB_WIDTH) -> str:
    """Encode a resized thumbnail as base64 JPEG for offline HTML report."""
    h, w = frame.shape[:2]
    target_h = max(1, int(h * width / max(w, 1)))
    thumb = cv2.resize(frame, (width, target_h), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 72])
    if not ok:
        return ""
    return base64.b64encode(encoded).decode("ascii")


def write_frames_to_video(kept_frames: list, output_path: Path, fps: float, frame_size: tuple):
    """Compatibility helper. Writes in-memory frames to H.264 via ffmpeg."""
    if not kept_frames:
        return

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{frame_size[0]}x{frame_size[1]}",
        "-pix_fmt",
        "bgr24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(OUTPUT_CRF),
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for frame in kept_frames:
            proc.stdin.write(frame.tobytes())
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait()


def _new_segment(segment_id: int, frame_idx: int, ts: float, reason: str, motion: float, face_found: bool, thumb_b64: str):
    return {
        "segment_id": segment_id,
        "start_frame": frame_idx,
        "end_frame": frame_idx,
        "start_sec": round(ts, 2),
        "end_sec": round(ts, 2),
        "frames_kept": 1,
        "reason_kept": reason,
        "face_detected": bool(face_found),
        "motion_score_avg": round(float(motion), 3),
        "thumbnail_b64": thumb_b64,
    }


def _append_segment(final_segments: list, report_segments: list, segment: dict):
    if segment is None:
        return

    machine = {
        "segment_id": segment["segment_id"],
        "start_frame": segment["start_frame"],
        "end_frame": segment["end_frame"],
        "start_sec": segment["start_sec"],
        "end_sec": segment["end_sec"],
        "frames_kept": segment["frames_kept"],
        "reason_kept": segment["reason_kept"],
        "face_detected": segment["face_detected"],
        "motion_score_avg": segment["motion_score_avg"],
    }
    final_segments.append(machine)

    report_item = dict(machine)
    report_item["thumbnail_b64"] = segment["thumbnail_b64"]
    report_segments.append(report_item)


def _transcode_h264(temp_path: Path, output_path: Path):
    ffmpeg = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if not ffmpeg:
        temp_path.replace(output_path)
        return

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(temp_path),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(OUTPUT_CRF),
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if res.returncode == 0:
        temp_path.unlink(missing_ok=True)
    else:
        temp_path.replace(output_path)


def generate_compression_report(segments: list, stats: dict, output_path: Path):
    """Write a standalone offline report with metrics and storyboard."""
    cards = []
    for s in segments[:MAX_REPORT_SEGMENTS]:
        color = "#2f855a" if s["face_detected"] else "#2563eb" if "motion" in s["reason_kept"] else "#7c3aed"
        cards.append(
            f"""
            <article class=\"card\" style=\"border-top:4px solid {color};\">
              <img src=\"data:image/jpeg;base64,{s['thumbnail_b64']}\" alt=\"segment {s['segment_id']}\" />
              <div class=\"meta\">
                <span class=\"badge\" style=\"background:{color};\">{s['reason_kept']}</span>
                <h3>Segment {s['segment_id']}</h3>
                <p>{s['start_sec']:.2f}s to {s['end_sec']:.2f}s</p>
                <p>{s['frames_kept']} kept | avg motion {s['motion_score_avg']:.3f}</p>
              </div>
            </article>
            """
        )

    discard = stats["frames_discarded_reasons"]

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Compression Report</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --paper: #fffdf8;
      --ink: #1e293b;
      --muted: #5b6472;
      --line: #d8cfc0;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; padding: 28px; color: var(--ink); font-family: Georgia, "Times New Roman", serif;
      background: radial-gradient(circle at top left, rgba(201,107,59,0.12), transparent 28%), linear-gradient(180deg, #f8f3ea 0%, #efe7da 100%); }}
    .wrap {{ max-width: 1120px; margin: 0 auto; }}
    .hero {{ background: var(--paper); border: 1px solid var(--line); border-radius: 18px; padding: 24px; }}
    h1 {{ margin: 0 0 6px; font-size: 1.95rem; }}
    .sub {{ margin: 0; color: var(--muted); }}
    .stats {{ margin-top: 18px; display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }}
    .stat {{ background: #fffaf2; border: 1px solid var(--line); border-radius: 14px; padding: 14px; }}
    .stat .label {{ font-size: 0.78rem; text-transform: uppercase; color: var(--muted); letter-spacing: 0.06em; }}
    .stat strong {{ display: block; font-size: 1.45rem; margin-top: 4px; }}
    h2 {{ margin: 26px 0 12px; font-size: 1.35rem; }}
    .board {{ display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); }}
    .card {{ background: var(--paper); border: 1px solid var(--line); border-radius: 16px; overflow: hidden; }}
    .card img {{ width: 100%; display: block; }}
    .meta {{ padding: 12px; }}
    .meta h3 {{ margin: 8px 0 6px; font-size: 1rem; }}
    .meta p {{ margin: 4px 0; color: var(--muted); font-family: "Segoe UI", Arial, sans-serif; font-size: 0.9rem; }}
    .badge {{ color: #fff; border-radius: 999px; padding: 4px 8px; font-size: 0.72rem; font-family: "Segoe UI", Arial, sans-serif; }}
    .notes {{ margin-top: 24px; background: var(--paper); border: 1px solid var(--line); border-radius: 16px; padding: 16px 18px; }}
    .notes ul {{ margin: 10px 0 0 18px; }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <section class=\"hero\">
      <h1>Smart Behavioral Video Compression</h1>
      <p class=\"sub\">Offline assignment report.</p>
      <div class=\"stats\">
        <div class=\"stat\"><span class=\"label\">Original Size</span><strong>{stats['original_size_mb']:.2f} MB</strong></div>
        <div class=\"stat\"><span class=\"label\">Compressed Size</span><strong>{stats['compressed_size_mb']:.2f} MB</strong></div>
        <div class=\"stat\"><span class=\"label\">Reduction</span><strong>{stats['reduction_pct']:.1f}%</strong></div>
        <div class=\"stat\"><span class=\"label\">Frames Kept</span><strong>{stats['frames_kept']} / {stats['frames_original']}</strong></div>
        <div class=\"stat\"><span class=\"label\">Segments</span><strong>{len(stats['segments'])}</strong></div>
        <div class=\"stat\"><span class=\"label\">Processing Time</span><strong>{stats['processing_time_sec']:.2f}s</strong></div>
      </div>
    </section>

    <h2>Storyboard</h2>
    <section class=\"board\">{''.join(cards)}</section>

    <section class=\"notes\">
      <h2>Discard Summary</h2>
      <p>Near duplicates by pHash: <strong>{discard['near_duplicate_phash']}</strong></p>
      <p>Low-motion no-face: <strong>{discard['low_motion_no_face']}</strong></p>
      <p>Total discarded: <strong>{discard['total_discarded']}</strong></p>
      <h2>Algorithm</h2>
      <ul>
        <li>Step 1: pHash duplicate drop at similarity &gt; {PHASH_THRESHOLD:.2f}.</li>
        <li>Step 2: optical flow low-motion check at &lt; {MOTION_DISCARD_THRESH:.2f}.</li>
        <li>Step 3: Haar face override for low-motion frames.</li>
        <li>Step 4: context keep every {CONTEXT_EVERY_SEC:.0f} seconds.</li>
        <li>Step 5: re-encode kept frames to H.264 MP4 at {OUTPUT_FPS} fps.</li>
      </ul>
    </section>
  </div>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")


def extract_intelligent_frames(
    video_path: str,
    frame_step: int = DEFAULT_FRAME_STEP,
    compute_scale: float = DEFAULT_COMPUTE_SCALE,
    strict_mode: bool = False,
):
    """Run full pipeline and create compressed video, report, and JSON."""
    input_path = Path(video_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    started_at = time.time()

    cv2.setUseOptimized(True)
    try:
        cv2.setNumThreads(max(1, (cv2.getNumberOfCPUs() or 4) - 1))
    except Exception:
        pass

    frame_step = max(1, int(frame_step))
    if strict_mode:
        frame_step = 1

    compute_scale = max(0.05, min(1.0, float(compute_scale)))

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps_in if fps_in else 0.0
    original_size_mb = input_path.stat().st_size / 1_000_000

    cw = max(1, int(width * compute_scale))
    ch = max(1, int(height * compute_scale))

    temp_out = VIDEO_OUT.with_suffix(".tmp.avi")
    writer = cv2.VideoWriter(str(temp_out), cv2.VideoWriter_fourcc(*"mp4v"), OUTPUT_FPS, (width, height))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not initialize temporary video writer")

    print(
        f"Input: {input_path.name} | {total_frames} frames | {duration_sec:.2f}s | {original_size_mb:.2f} MB | frame_step={frame_step}"
    )
    sys.stdout.flush()

    prev_flow_gray = None
    prev_kept_hash = None
    last_kept_ts = -CONTEXT_EVERY_SEC

    frames_kept = 0
    discard_dup = 0
    discard_static = 0

    machine_segments = []
    report_segments = []
    active_segment = None

    processed_frames = 0
    frame_idx = 0

    while True:
        if frame_step > 1 and (frame_idx % frame_step != 0):
            if not cap.grab():
                break
            frame_idx += 1
            continue

        ok, frame = cap.read()
        if not ok:
            break

        ts = frame_idx / fps_in if fps_in else 0.0

        small = cv2.resize(frame, (cw, ch), interpolation=cv2.INTER_AREA)
        small_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        curr_hash = compute_phash(small_gray)
        motion = compute_motion_score(prev_flow_gray, small_gray)

        face_needed = strict_mode or (motion < DEFAULT_FACE_CHECK_MOTION_THRESH)
        if face_needed:
            face_input = small_gray
        else:
            # Placeholder; has_face will not run when not needed.
            face_input = None

        if face_needed:
            keep, reason, face_found = should_keep_frame(
                curr_hash,
                prev_kept_hash,
                motion,
                face_input,
                last_kept_ts,
                ts,
                cascade,
            )
        else:
            # Same step order, but skips face compute for higher-motion frames in fast mode.
            similarity = phash_similarity(curr_hash, prev_kept_hash) if prev_kept_hash is not None else 0.0
            if similarity > PHASH_THRESHOLD:
                keep, reason, face_found = False, "discarded_duplicate", False
            elif motion >= MOTION_DISCARD_THRESH:
                keep, reason, face_found = True, "motion_above_threshold", False
            elif (ts - last_kept_ts) >= CONTEXT_EVERY_SEC:
                keep, reason, face_found = True, "context_frame", False
            else:
                keep, reason, face_found = False, "discarded_static", False

        if keep:
            writer.write(frame)
            frames_kept += 1
            prev_kept_hash = curr_hash
            last_kept_ts = ts

            if active_segment is None or (ts - active_segment["end_sec"]) > SEGMENT_GAP_SEC:
                _append_segment(machine_segments, report_segments, active_segment)
                active_segment = _new_segment(
                    len(machine_segments) + 1,
                    frame_idx,
                    ts,
                    reason,
                    motion,
                    face_found,
                    frame_to_b64_thumb(frame),
                )
            else:
                active_segment["end_frame"] = frame_idx
                active_segment["end_sec"] = round(ts, 2)
                active_segment["frames_kept"] += 1
                active_segment["face_detected"] = active_segment["face_detected"] or face_found
                n = active_segment["frames_kept"]
                prev_avg = active_segment["motion_score_avg"]
                active_segment["motion_score_avg"] = round(((prev_avg * (n - 1)) + motion) / n, 3)
        else:
            if reason == "discarded_duplicate":
                discard_dup += 1
            else:
                discard_static += 1

        prev_flow_gray = small_gray
        processed_frames += 1
        frame_idx += 1

        if processed_frames % 600 == 0:
            elapsed = time.time() - started_at
            print(f"Processed {processed_frames} sampled frames | kept {frames_kept} | {elapsed:.2f}s")
            sys.stdout.flush()

    _append_segment(machine_segments, report_segments, active_segment)

    cap.release()
    writer.release()

    _transcode_h264(temp_out, VIDEO_OUT)

    compressed_size_mb = VIDEO_OUT.stat().st_size / 1_000_000 if VIDEO_OUT.exists() else 0.0
    elapsed_total = time.time() - started_at

    stats = {
        "source_video": input_path.name,
        "compressed_video": VIDEO_OUT.name,
        "original_size_mb": round(original_size_mb, 2),
        "compressed_size_mb": round(compressed_size_mb, 2),
        "reduction_pct": round((1.0 - (compressed_size_mb / max(original_size_mb, 1e-9))) * 100.0, 1),
        "original_duration_sec": round(duration_sec, 2),
        "compressed_duration_sec": round(frames_kept / OUTPUT_FPS, 2),
        "original_fps": round(fps_in, 2),
        "output_fps": OUTPUT_FPS,
        "frames_original": total_frames,
        "frames_processed": processed_frames,
        "frame_step": frame_step,
        "frames_kept": frames_kept,
        "processing_time_sec": round(elapsed_total, 2),
        "segments": machine_segments,
        "frames_discarded_reasons": {
            "near_duplicate_phash": discard_dup,
            "low_motion_no_face": discard_static,
            "total_discarded": max(0, processed_frames - frames_kept),
        },
    }

    SEGMENTS_JSON_OUT.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    generate_compression_report(report_segments, stats, REPORT_HTML_OUT)

    print("=" * 64)
    print(f"Done in {stats['processing_time_sec']:.2f}s")
    print(
        f"Size: {stats['original_size_mb']:.2f} MB -> {stats['compressed_size_mb']:.2f} MB "
        f"({stats['reduction_pct']:.1f}% smaller)"
    )
    print(f"Duration: {stats['original_duration_sec']:.2f}s -> {stats['compressed_duration_sec']:.2f}s")
    print(f"Frames kept: {stats['frames_kept']} / {stats['frames_processed']} processed")
    print(f"Outputs: {VIDEO_OUT.name}, {REPORT_HTML_OUT.name}, {SEGMENTS_JSON_OUT.name}")
    print("=" * 64)
    sys.stdout.flush()

    return stats


def _default_input_path() -> Path:
    if VIDEO_IN_DEFAULT.exists():
        return VIDEO_IN_DEFAULT
    demo = Path("demo.mp4")
    if demo.exists():
        return demo
    return VIDEO_IN_DEFAULT


def _parse_args():
    parser = argparse.ArgumentParser(description="Smart behavioral video compression")
    parser.add_argument("--input", default=str(_default_input_path()), help="Input video path")
    parser.add_argument("--frame-step", type=int, default=DEFAULT_FRAME_STEP, help="Process every Nth frame (fast mode)")
    parser.add_argument("--compute-scale", type=float, default=DEFAULT_COMPUTE_SCALE, help="Compute resolution scale (0.05-1.0)")
    parser.add_argument("--strict", action="store_true", help="Strict full-frame mode (frame-step=1, always face-check static)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    extract_intelligent_frames(
        video_path=args.input,
        frame_step=args.frame_step,
        compute_scale=args.compute_scale,
        strict_mode=args.strict,
    )
