import cv2
import numpy as np
from collections import deque
from pathlib import Path

VIDEO_PATH = "data/road.mp4"
OUTPUT_VIDEO_PATH = "results/lane_output.mp4"
CENTER_REJECTION_RATIO = 0.15
LEFT_BOUNDARY_MAX_RATIO = 0.35
RIGHT_BOUNDARY_MIN_RATIO = 0.65
LANE_TOP_RATIO = 0.55
SMOOTHING_WINDOW = 5
THUMBNAIL_COUNT = 5
THUMBNAIL_STRIP_HEIGHT = 100
BOUNDARY_BOTTOM_OUTWARD_SHIFT = 20
BOUNDARY_TOP_OUTWARD_SHIFT = 70
FILL_BOTTOM_INSET_RATIO = 0.16
FILL_TOP_INSET_RATIO = 0.08

boundary_pair_history = deque(maxlen=SMOOTHING_WINDOW)
thumbnail_history = deque(maxlen=THUMBNAIL_COUNT)


def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    polygon = np.array(
        [[
            (0, height),
            (width, height),
            (int(width * 0.85), int(height * LANE_TOP_RATIO)),
            (int(width * 0.15), int(height * LANE_TOP_RATIO)),
        ]],
        dtype=np.int32,
    )

    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(edges, mask)


def filter_boundary_lines(lines, width):
    left_lines = []
    right_lines = []

    if lines is None:
        return left_lines, right_lines

    image_center = width // 2
    center_rejection_width = width * CENTER_REJECTION_RATIO
    left_boundary_max_x = width * LEFT_BOUNDARY_MAX_RATIO
    right_boundary_min_x = width * RIGHT_BOUNDARY_MIN_RATIO

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue

        slope = (y2 - y1) / (x2 - x1)
        mid_x = (x1 + x2) / 2
        bottom_x = x1 if y1 > y2 else x2

        if abs(mid_x - image_center) < center_rejection_width:
            continue

        if bottom_x < left_boundary_max_x and mid_x < image_center and slope < 0:
            left_lines.append((x1, y1, x2, y2))
        elif bottom_x > right_boundary_min_x and mid_x > image_center and slope > 0:
            right_lines.append((x1, y1, x2, y2))

    return left_lines, right_lines


def white_edge_mask(image, edges):
    white_mask = cv2.inRange(image, np.array([170, 170, 170]), np.array([255, 255, 255]))
    white_mask = cv2.GaussianBlur(white_mask, (5, 5), 0)
    white_mask = cv2.dilate(white_mask, np.ones((5, 5), dtype=np.uint8), iterations=1)
    return cv2.bitwise_and(edges, white_mask)


def score_line_on_white_edges(line, white_edges):
    x1, y1, x2, y2 = line
    length = np.hypot(x2 - x1, y2 - y1)
    samples = max(10, int(length))
    xs = np.linspace(x1, x2, samples).astype(np.int32)
    ys = np.linspace(y1, y2, samples).astype(np.int32)

    xs = np.clip(xs, 0, white_edges.shape[1] - 1)
    ys = np.clip(ys, 0, white_edges.shape[0] - 1)

    white_hits = np.count_nonzero(white_edges[ys, xs])
    return max(1.0, length + white_hits * 4.0)


def average_boundary_line(lines, height, white_edges):
    if len(lines) == 0:
        return None

    slopes = []
    intercepts = []
    weights = []

    for x1, y1, x2, y2 in lines:
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        slopes.append(slope)
        intercepts.append(intercept)
        weights.append(score_line_on_white_edges((x1, y1, x2, y2), white_edges))

    if len(slopes) == 0:
        return None

    slope = np.average(slopes, weights=weights)
    intercept = np.average(intercepts, weights=weights)

    y1 = height
    y2 = int(height * LANE_TOP_RATIO)

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return x1, y1, x2, y2


def validate_boundaries(left_boundary, right_boundary):
    if left_boundary is None or right_boundary is None:
        return None, None

    left_x_bottom = left_boundary[0]
    right_x_bottom = right_boundary[0]
    left_x_top = left_boundary[2]
    right_x_top = right_boundary[2]

    if left_x_bottom >= right_x_bottom or left_x_top >= right_x_top:
        return None, None

    return left_boundary, right_boundary


def constrain_boundary_to_outer_band(boundary, width, side):
    if boundary is None:
        return None

    x1, y1, x2, y2 = boundary
    if side == "left":
        max_x = int(width * LEFT_BOUNDARY_MAX_RATIO)
        x1 = int(np.clip(x1, 0, max_x))
        x2 = int(np.clip(x2, 0, max_x))
    else:
        min_x = int(width * RIGHT_BOUNDARY_MIN_RATIO)
        x1 = int(np.clip(x1, min_x, width - 1))
        x2 = int(np.clip(x2, min_x, width - 1))

    return x1, y1, x2, y2


def smooth_boundary_pair(left_boundary, right_boundary):
    left_boundary, right_boundary = validate_boundaries(left_boundary, right_boundary)
    if left_boundary is None or right_boundary is None:
        return None, None

    boundary_pair_history.append(
        np.array([left_boundary, right_boundary], dtype=np.float32)
    )
    smoothed_pair = np.mean(np.stack(boundary_pair_history), axis=0).astype(np.int32)

    return validate_boundaries(tuple(smoothed_pair[0]), tuple(smoothed_pair[1]))


def draw_boundary(img, boundary, color):
    if boundary is None:
        return
    x1, y1, x2, y2 = boundary
    x1 = max(0, min(img.shape[1] - 1, x1))
    x2 = max(0, min(img.shape[1] - 1, x2))

    cv2.line(img, (x1, y1), (x2, y2), color, 8)


def adjust_boundaries_for_display(left_boundary, right_boundary, width):
    left_boundary, right_boundary = validate_boundaries(left_boundary, right_boundary)
    if left_boundary is None or right_boundary is None:
        return None, None

    lx1, ly1, lx2, ly2 = left_boundary
    rx1, ry1, rx2, ry2 = right_boundary

    lx1 = int(np.clip(lx1 - BOUNDARY_BOTTOM_OUTWARD_SHIFT, 0, width - 1))
    lx2 = int(np.clip(lx2 - BOUNDARY_TOP_OUTWARD_SHIFT, 0, width - 1))
    rx1 = int(np.clip(rx1 + BOUNDARY_BOTTOM_OUTWARD_SHIFT, 0, width - 1))
    rx2 = int(np.clip(rx2 + BOUNDARY_TOP_OUTWARD_SHIFT, 0, width - 1))

    return validate_boundaries((lx1, ly1, lx2, ly2), (rx1, ry1, rx2, ry2))


def inset_lane_fill(left_boundary, right_boundary):
    lx1, ly1, lx2, ly2 = left_boundary
    rx1, ry1, rx2, ry2 = right_boundary

    bottom_width = rx1 - lx1
    top_width = rx2 - lx2

    fill_lx1 = int(lx1 + bottom_width * FILL_BOTTOM_INSET_RATIO)
    fill_rx1 = int(rx1 - bottom_width * FILL_BOTTOM_INSET_RATIO)
    fill_lx2 = int(lx2 + top_width * FILL_TOP_INSET_RATIO)
    fill_rx2 = int(rx2 - top_width * FILL_TOP_INSET_RATIO)

    return [(fill_lx1, ly1), (fill_lx2, ly2), (fill_rx2, ry2), (fill_rx1, ry1)]


def draw_lane_area(img, left_boundary, right_boundary):
    if left_boundary is None or right_boundary is None:
        return

    polygon = np.array([inset_lane_fill(left_boundary, right_boundary)], dtype=np.int32)

    overlay = img.copy()
    cv2.fillPoly(overlay, polygon, (80, 190, 55))
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)


def draw_translucent_panel(img, top_left, bottom_right, alpha=0.78):
    overlay = img.copy()
    cv2.rectangle(overlay, top_left, bottom_right, (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_legend(img):
    draw_translucent_panel(img, (8, 10), (260, 112))

    cv2.line(img, (24, 32), (58, 32), (255, 0, 0), 3)
    cv2.putText(img, "Left Lane Boundary", (74, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.line(img, (24, 64), (58, 64), (0, 255, 0), 3)
    cv2.putText(img, "Right Lane Boundary", (74, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.rectangle(img, (24, 84), (58, 102), (80, 190, 55), -1)
    cv2.putText(img, "Detected Lane Area", (74, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)


def estimate_dashboard_metrics(img, left_boundary, right_boundary):
    if left_boundary is None or right_boundary is None:
        return "N/A", "N/A", "Searching"

    lane_center = (left_boundary[0] + right_boundary[0]) / 2
    vehicle_center = img.shape[1] / 2
    lane_width_px = max(1, right_boundary[0] - left_boundary[0])
    offset_m = (vehicle_center - lane_center) / lane_width_px * 3.7

    left_angle = np.degrees(np.arctan2(left_boundary[1] - left_boundary[3], left_boundary[0] - left_boundary[2]))
    right_angle = np.degrees(np.arctan2(right_boundary[1] - right_boundary[3], right_boundary[0] - right_boundary[2]))
    curvature_km = max(0.25, min(9.99, 8.0 / (abs(left_angle - right_angle) + 1.0)))

    if abs(offset_m) < 0.25:
        status = "Lane Centered"
    elif offset_m > 0:
        status = "Shift Right"
    else:
        status = "Shift Left"

    return f"{curvature_km:.2f} km", f"{offset_m:+.2f} m", status


def draw_dashboard(img, left_boundary, right_boundary):
    h, w = img.shape[:2]
    x1 = max(0, w - 210)
    y1 = 10
    x2 = w - 8
    y2 = 214
    draw_translucent_panel(img, (x1, y1), (x2, y2))

    curvature, offset, status = estimate_dashboard_metrics(img, left_boundary, right_boundary)

    cv2.putText(img, "DASHBOARD", (x1 + 28, y1 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(img, (x1 + 16, y1 + 52), (x2 - 16, y1 + 52), (230, 230, 230), 1)

    yellow = (0, 230, 255)
    green = (40, 255, 40)
    white = (255, 255, 255)

    cv2.putText(img, "Curvature", (x1 + 28, y1 + 84), cv2.FONT_HERSHEY_SIMPLEX, 0.52, yellow, 1, cv2.LINE_AA)
    cv2.putText(img, curvature, (x1 + 28, y1 + 108), cv2.FONT_HERSHEY_SIMPLEX, 0.52, white, 1, cv2.LINE_AA)
    cv2.putText(img, "Vehicle Offset", (x1 + 28, y1 + 142), cv2.FONT_HERSHEY_SIMPLEX, 0.52, yellow, 1, cv2.LINE_AA)
    cv2.putText(img, offset, (x1 + 28, y1 + 166), cv2.FONT_HERSHEY_SIMPLEX, 0.52, white, 1, cv2.LINE_AA)
    cv2.putText(img, "Status", (x1 + 28, y1 + 198), cv2.FONT_HERSHEY_SIMPLEX, 0.52, yellow, 1, cv2.LINE_AA)
    cv2.putText(img, status, (x1 + 88, y1 + 198), cv2.FONT_HERSHEY_SIMPLEX, 0.52, green, 1, cv2.LINE_AA)


def compose_output_frame(main_view):
    thumbnail_history.append(main_view.copy())

    h, w = main_view.shape[:2]
    output = np.zeros((h + THUMBNAIL_STRIP_HEIGHT, w, 3), dtype=np.uint8)
    output[:h] = main_view

    thumb_width = w // THUMBNAIL_COUNT
    for index in range(THUMBNAIL_COUNT):
        if index < len(thumbnail_history):
            thumb = thumbnail_history[index]
        else:
            thumb = main_view

        thumb = cv2.resize(thumb, (thumb_width, THUMBNAIL_STRIP_HEIGHT))
        x1 = index * thumb_width
        x2 = w if index == THUMBNAIL_COUNT - 1 else x1 + thumb_width
        output[h:h + THUMBNAIL_STRIP_HEIGHT, x1:x2] = cv2.resize(thumb, (x2 - x1, THUMBNAIL_STRIP_HEIGHT))
        cv2.rectangle(output, (x1, h), (x2 - 1, h + THUMBNAIL_STRIP_HEIGHT - 1), (0, 0, 0), 2)

    cv2.line(output, (0, h), (w, h), (0, 0, 0), 3)
    return output


def process_frame(frame):
    image = cv2.resize(frame, (800, 500))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    white_edges = white_edge_mask(image, edges)
    roi = region_of_interest(white_edges)

    lines = cv2.HoughLinesP(
        roi,
        2,
        np.pi / 180,
        50,
        minLineLength=50,
        maxLineGap=100,
    )

    _, width = roi.shape
    left_lines, right_lines = filter_boundary_lines(lines, width)

    lane_image = image.copy()
    left_boundary = average_boundary_line(left_lines, image.shape[0], white_edges)
    right_boundary = average_boundary_line(right_lines, image.shape[0], white_edges)
    left_boundary = constrain_boundary_to_outer_band(left_boundary, image.shape[1], "left")
    right_boundary = constrain_boundary_to_outer_band(right_boundary, image.shape[1], "right")
    left_boundary, right_boundary = smooth_boundary_pair(left_boundary, right_boundary)
    left_boundary, right_boundary = adjust_boundaries_for_display(left_boundary, right_boundary, image.shape[1])

    draw_lane_area(lane_image, left_boundary, right_boundary)
    draw_boundary(lane_image, left_boundary, (255, 0, 0))
    draw_boundary(lane_image, right_boundary, (0, 255, 0))
    draw_legend(lane_image)
    draw_dashboard(lane_image, left_boundary, right_boundary)

    return compose_output_frame(lane_image)


def main():
    capture = cv2.VideoCapture(VIDEO_PATH)
    if not capture.isOpened():
        raise SystemExit(f"Error: Video not found or cannot be opened: {VIDEO_PATH}")

    Path(OUTPUT_VIDEO_PATH).parent.mkdir(parents=True, exist_ok=True)

    writer = None
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        lane_image = process_frame(frame)

        if writer is None:
            frame_height, frame_width = lane_image.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

        writer.write(lane_image)

    capture.release()
    if writer is not None:
        writer.release()


if __name__ == "__main__":
    main()
