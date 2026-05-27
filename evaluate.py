import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import lane_detection as ld

EVAL_DIR = ROOT_DIR / "results" / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = EVAL_DIR / "metrics.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def collect_files(path, extensions):
    path = Path(path)
    if path.is_dir():
        return sorted([p for p in path.iterdir() if p.suffix.lower() in extensions])
    if path.is_file() and path.suffix.lower() in extensions:
        return [path]
    return []


def discover_scenarios(data_path):
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    scenario_dirs = [p for p in sorted(data_path.iterdir()) if p.is_dir()]
    if scenario_dirs:
        return scenario_dirs
    return [data_path]


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def reset_lane_buffers():
    ld.left_curve_buffer.clear()
    ld.right_curve_buffer.clear()


def save_annotated_image(output_dir, image_path, lane_image):
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    output_path = output_dir / f"annotated_{image_path.stem}.png"
    cv2.imwrite(str(output_path), lane_image)
    return output_path


def save_annotated_video(output_dir, video_path, lane_frames, fps):
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    output_path = output_dir / f"annotated_{video_path.stem}.mp4"
    if not lane_frames:
        return None

    height, width = lane_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    for frame in lane_frames:
        writer.write(frame)
    writer.release()
    return output_path


def evaluate_image(path, output_dir=None):
    frame = cv2.imread(str(path))
    if frame is None:
        raise FileNotFoundError(f"Unable to read image: {path}")

    reset_lane_buffers()
    start_time = time.time()
    _, _, _, _, lane_image, metrics, curvature = ld.process_frame(frame, return_metrics=True)
    elapsed = time.time() - start_time

    if output_dir is not None:
        save_annotated_image(output_dir, path, lane_image)

    valid = metrics is not None and curvature is not None and np.isfinite(curvature.get("curvature_radius_m", np.nan))
    lane_width_m = metrics["lane_width_m"] if valid else None
    curvature_m = curvature["curvature_radius_m"] if valid else None

    return {
        "source": str(path),
        "type": "image",
        "samples": 1,
        "source_fps": None,
        "processing_fps": 1.0 / elapsed if elapsed > 0 else 0.0,
        "detection_rate": 100.0 if valid else 0.0,
        "valid_count": 1 if valid else 0,
        "invalid_count": 0 if valid else 1,
        "avg_lane_width_m": lane_width_m,
        "std_lane_width_m": 0.0 if valid else None,
        "avg_curvature_m": curvature_m,
        "std_curvature_m": 0.0 if valid else None,
        "curvature_cv": 0.0 if valid else None,
    }


def evaluate_video(path, output_dir=None):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {path}")

    reset_lane_buffers()
    frame_count = 0
    valid_count = 0
    lane_widths = []
    curvatures = []
    total_processing_time = 0.0
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    lane_frames = []

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frame_count += 1

        start_time = time.time()
        _, _, _, _, lane_image, metrics, curvature = ld.process_frame(frame, return_metrics=True)
        total_processing_time += time.time() - start_time

        if output_dir is not None:
            lane_frames.append(lane_image)

        valid = metrics is not None and curvature is not None and np.isfinite(curvature.get("curvature_radius_m", np.nan))
        if valid:
            valid_count += 1
            lane_widths.append(metrics["lane_width_m"])
            curvatures.append(curvature["curvature_radius_m"])

    capture.release()

    if output_dir is not None and lane_frames:
        save_annotated_video(output_dir, path, lane_frames, source_fps if source_fps > 0 else 30.0)

    detection_rate = 100.0 * valid_count / frame_count if frame_count else 0.0
    avg_width = float(np.mean(lane_widths)) if lane_widths else None
    std_width = float(np.std(lane_widths)) if lane_widths else None
    avg_curvature = float(np.mean(curvatures)) if curvatures else None
    std_curvature = float(np.std(curvatures)) if curvatures else None
    curvature_cv = float(std_curvature / avg_curvature) if avg_curvature and std_curvature is not None else None
    processing_fps = float(frame_count / total_processing_time) if total_processing_time > 0 else 0.0

    return {
        "source": str(path),
        "type": "video",
        "samples": frame_count,
        "source_fps": float(source_fps) if source_fps and source_fps > 0 else None,
        "processing_fps": processing_fps,
        "detection_rate": detection_rate,
        "valid_count": valid_count,
        "invalid_count": frame_count - valid_count,
        "avg_lane_width_m": avg_width,
        "std_lane_width_m": std_width,
        "avg_curvature_m": avg_curvature,
        "std_curvature_m": std_curvature,
        "curvature_cv": curvature_cv,
    }


def write_metrics(rows, csv_path):
    fieldnames = [
        "source",
        "type",
        "samples",
        "source_fps",
        "processing_fps",
        "detection_rate",
        "valid_count",
        "invalid_count",
        "avg_lane_width_m",
        "std_lane_width_m",
        "avg_curvature_m",
        "std_curvature_m",
        "curvature_cv",
    ]
    with open(csv_path, mode="w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("{:.4f}".format(v) if isinstance(v, float) else v) for k, v in row.items()})


def main():
    parser = argparse.ArgumentParser(description="Evaluate lane detection performance on images and videos.")
    parser.add_argument("--data", type=str, default=str(ROOT_DIR / "data"), help="Root data folder containing scenario subfolders.")
    parser.add_argument("--output", type=str, default=str(CSV_PATH), help="Path to output CSV metrics file.")
    args = parser.parse_args()

    scenarios = discover_scenarios(args.data)
    if not scenarios:
        raise SystemExit("No scenario directories found under data path.")

    aggregated_rows = []
    output_root = Path(args.output).parent
    output_root.mkdir(parents=True, exist_ok=True)

    for scenario_path in scenarios:
        scenario_name = scenario_path.name
        image_files = collect_files(scenario_path, IMAGE_EXTENSIONS)
        video_files = collect_files(scenario_path, VIDEO_EXTENSIONS)
        if not image_files and not video_files:
            continue

        scenario_output = output_root / scenario_name
        scenario_output.mkdir(parents=True, exist_ok=True)
        image_output = scenario_output / "images"
        video_output = scenario_output / "videos"
        ensure_dir(image_output)
        ensure_dir(video_output)

        rows = []
        print(f"Evaluating scenario: {scenario_name}")

        for image_path in image_files:
            print(f"  image: {image_path.name}")
            rows.append(evaluate_image(image_path, output_dir=image_output))

        for video_path in video_files:
            print(f"  video: {video_path.name}")
            rows.append(evaluate_video(video_path, output_dir=video_output))

        write_metrics(rows, scenario_output / "metrics.csv")
        aggregated_rows.extend(rows)
        print(f"  Scenario metrics saved to: {scenario_output / 'metrics.csv'}")

    if aggregated_rows:
        write_metrics(aggregated_rows, Path(args.output))
        print(f"Overall metrics saved to: {args.output}")

    print("Evaluation complete.")


if __name__ == "__main__":
    main()
