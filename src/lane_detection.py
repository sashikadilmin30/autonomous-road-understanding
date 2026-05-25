import cv2
import numpy as np


LANE_TOP_RATIO = 0.52
CURVE_SAMPLE_COUNT = 25
VIDEO_PATH = "data/road.mp4"
OUTPUT_VIDEO_PATH = "results/lane_output.mp4"
PLAYBACK_SLOWDOWN = 2.0
DISPLAY_FPS = None
PROCESS_EVERY_NTH_FRAME = 3
COLOR_LABEL = (255, 255, 255)
COLOR_GOOD = (0, 220, 80)
COLOR_WARN = (0, 220, 255)
COLOR_ERROR = (0, 80, 255)
COLOR_PANEL = (20, 20, 20)
COLOR_LANE_FILL = (0, 200, 90)
COLOR_SECTION = (180, 180, 180)


def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    polygon = np.array(
        [[
            (int(width * 0.18), height),
            (int(width * 0.82), height),
            (int(width * 0.59), int(height * 0.52)),
            (int(width * 0.41), int(height * 0.52)),
        ]],
        dtype=np.int32,
    )

    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(edges, mask)


def average_line(lines):
    if not lines:
        return None

    slopes = []
    intercepts = []
    weights = []

    for x1, y1, x2, y2 in lines:
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        length = np.hypot(x2 - x1, y2 - y1)

        slopes.append(slope)
        intercepts.append(intercept)
        weights.append(length)

    return np.average(slopes, weights=weights), np.average(intercepts, weights=weights)


def classify_lane_lines(image, lines):
    left_lines = []
    right_lines = []

    _, width, _ = image.shape
    center_x = width / 2
    center_margin = width * 0.05

    if lines is None:
        return left_lines, right_lines

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x1 == x2:
            continue

        slope = (y2 - y1) / (x2 - x1)
        length = np.hypot(x2 - x1, y2 - y1)
        midpoint_x = (x1 + x2) / 2

        if length < 20:
            continue

        if abs(slope) < 0.15 or abs(slope) > 1.5:
            continue

        if abs(midpoint_x - center_x) < center_margin:
            continue

        if slope < 0 and midpoint_x < center_x:
            left_lines.append((x1, y1, x2, y2))
        elif slope > 0 and midpoint_x > center_x:
            right_lines.append((x1, y1, x2, y2))

    return left_lines, right_lines


def draw_full_line(image, line_params, color):
    if line_params is None:
        return

    slope, intercept = line_params
    y1 = image.shape[0]
    y2 = int(y1 * LANE_TOP_RATIO)

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    x1 = max(0, min(image.shape[1] - 1, x1))
    x2 = max(0, min(image.shape[1] - 1, x2))

    cv2.line(image, (x1, y1), (x2, y2), color, 6)


def sample_points_from_lines(lines, samples_per_line=CURVE_SAMPLE_COUNT):
    points = []

    for x1, y1, x2, y2 in lines:
        xs = np.linspace(x1, x2, samples_per_line)
        ys = np.linspace(y1, y2, samples_per_line)

        for x, y in zip(xs, ys):
            points.append((float(x), float(y)))

    return points


def fit_lane_curve(points):
    if len(points) < 6:
        return None

    y = np.array([point[1] for point in points], dtype=np.float32)
    x = np.array([point[0] for point in points], dtype=np.float32)

    if len(np.unique(y.astype(np.int32))) < 3:
        return None

    return np.polyfit(y, x, 2)


def generate_curve_points(image, curve, num_points=60):
    if curve is None:
        return None

    y_top = int(image.shape[0] * LANE_TOP_RATIO)
    y_bottom = image.shape[0] - 1
    y_values = np.linspace(y_top, y_bottom, num_points)
    x_values = np.polyval(curve, y_values)

    curve_points = []
    for x, y in zip(x_values, y_values):
        x = int(np.clip(x, 0, image.shape[1] - 1))
        y = int(np.clip(y, 0, image.shape[0] - 1))
        curve_points.append((x, y))

    return curve_points


def line_points_from_params(image, line_params):
    if line_params is None:
        return None

    slope, intercept = line_params
    if abs(slope) < 1e-6:
        return None

    y_bottom = image.shape[0] - 1
    y_top = int(image.shape[0] * LANE_TOP_RATIO)

    x_bottom = int((y_bottom - intercept) / slope)
    x_top = int((y_top - intercept) / slope)

    x_bottom = max(0, min(image.shape[1] - 1, x_bottom))
    x_top = max(0, min(image.shape[1] - 1, x_top))

    return (x_bottom, y_bottom), (x_top, y_top)


def fill_lane_area(image, left_points, right_points, left_curve_points=None, right_curve_points=None):
    if left_points is None or right_points is None:
        return image.copy(), None

    overlay = image.copy()
    if left_curve_points is not None and right_curve_points is not None:
        polygon_points = left_curve_points + list(reversed(right_curve_points))
    else:
        polygon_points = [
            left_points[0],
            left_points[1],
            right_points[1],
            right_points[0],
        ]

    lane_polygon = np.array([polygon_points], dtype=np.int32)

    cv2.fillPoly(overlay, lane_polygon, COLOR_LANE_FILL)
    filled = cv2.addWeighted(overlay, 0.18, image, 0.82, 0)
    return filled, lane_polygon


def calculate_lane_metrics(image, left_points, right_points):
    if left_points is None or right_points is None:
        return None

    vehicle_center_x = image.shape[1] // 2
    lane_center_x = int((left_points[0][0] + right_points[0][0]) / 2)
    offset_px = vehicle_center_x - lane_center_x
    lane_width_px = max(1, right_points[0][0] - left_points[0][0])
    offset_percent = (offset_px / lane_width_px) * 100

    if offset_px > 0:
        direction = "right"
    elif offset_px < 0:
        direction = "left"
    else:
        direction = "center"

    return {
        "vehicle_center_x": vehicle_center_x,
        "lane_center_x": lane_center_x,
        "offset_px": offset_px,
        "offset_percent": offset_percent,
        "direction": direction,
    }


def calculate_curvature_radius(curve, y_eval):
    if curve is None:
        return None

    a, b, _ = curve
    denominator = abs(2 * a)
    if denominator < 1e-6:
        return float("inf")

    return ((1 + (2 * a * y_eval + b) ** 2) ** 1.5) / denominator


def estimate_lane_curvature(image, left_curve, right_curve, left_curve_points, right_curve_points):
    if left_curve is None or right_curve is None:
        return None

    y_eval = image.shape[0] - 1
    left_radius = calculate_curvature_radius(left_curve, y_eval)
    right_radius = calculate_curvature_radius(right_curve, y_eval)

    if left_radius is None or right_radius is None:
        return None

    lane_center_bottom = int((left_curve_points[-1][0] + right_curve_points[-1][0]) / 2)
    lane_center_top = int((left_curve_points[0][0] + right_curve_points[0][0]) / 2)
    center_shift = lane_center_top - lane_center_bottom
    straight_threshold = image.shape[1] * 0.02

    if abs(center_shift) < straight_threshold:
        road_direction = "Straight"
    elif center_shift > 0:
        road_direction = "Right Curve"
    else:
        road_direction = "Left Curve"

    finite_radii = [radius for radius in (left_radius, right_radius) if np.isfinite(radius)]
    curvature_radius = float(np.mean(finite_radii)) if finite_radii else float("inf")

    return {
        "left_radius": left_radius,
        "right_radius": right_radius,
        "curvature_radius": curvature_radius,
        "road_direction": road_direction,
    }


def get_status_color(level):
    if level == "error":
        return COLOR_ERROR
    if level == "warning":
        return COLOR_WARN
    return COLOR_GOOD


def draw_dashboard_row(image, label, value, y, level):
    cv2.putText(image, label, (32, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_LABEL, 1)
    cv2.putText(image, value, (175, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, get_status_color(level), 2)


def draw_dashboard_section(image, title, y):
    cv2.putText(image, title, (32, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_SECTION, 1)
    cv2.line(image, (135, y - 5), (332, y - 5), (70, 70, 70), 1)


def draw_info_panel(image, metrics, curvature):
    panel = image.copy()
    x1, y1 = 15, 15
    x2, y2 = 365, 255
    cv2.rectangle(panel, (x1, y1), (x2, y2), COLOR_PANEL, -1)
    cv2.addWeighted(panel, 0.5, image, 0.5, 0, image)

    cv2.putText(image, "Autonomous Road System", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.72, COLOR_LABEL, 2)
    cv2.putText(image, "Lane Guidance Dashboard", (31, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_SECTION, 1)

    draw_dashboard_section(image, "Offset", 92)
    if metrics is None:
        draw_dashboard_row(image, "Lane Center", "Unavailable", 118, "error")
        draw_dashboard_row(image, "Vehicle Offset", "No lane lock", 145, "error")
    else:
        offset_abs = abs(metrics["offset_px"])
        offset_ratio = abs(metrics["offset_percent"])
        if metrics["direction"] == "center":
            offset_text = "Centered"
        else:
            offset_text = f"{offset_abs}px {metrics['direction']}"

        if offset_ratio <= 5:
            offset_level = "safe"
        elif offset_ratio <= 12:
            offset_level = "warning"
        else:
            offset_level = "error"

        draw_dashboard_row(image, "Lane Center", f"{metrics['lane_center_x']} px", 118, "safe")
        draw_dashboard_row(image, "Vehicle Offset", offset_text, 145, offset_level)

    draw_dashboard_section(image, "Curvature", 177)
    if curvature is None:
        draw_dashboard_row(image, "Radius", "Unavailable", 203, "error")
    else:
        if np.isfinite(curvature["curvature_radius"]):
            curvature_text = f"{curvature['curvature_radius']:.1f} px"
        else:
            curvature_text = "Very large"
        draw_dashboard_row(image, "Radius", curvature_text, 203, "safe")

    draw_dashboard_section(image, "Direction", 235)
    if curvature is None:
        draw_dashboard_row(image, "Road Shape", "Unknown", 251, "error")
    else:
        if curvature["road_direction"] == "Straight":
            direction_level = "safe"
        else:
            direction_level = "warning"
        draw_dashboard_row(image, "Road Shape", curvature["road_direction"], 251, direction_level)


def get_display_delay_ms(source_fps):
    target_fps = DISPLAY_FPS if DISPLAY_FPS is not None else source_fps / PLAYBACK_SLOWDOWN
    if target_fps <= 0:
        target_fps = 15.0

    return max(1, int(1000 / target_fps))


def draw_lane_guidance(image, left_points, right_points, metrics, curvature=None):
    guided = image.copy()

    if left_points is not None and right_points is not None:
        lane_center_top = (
            int((left_points[1][0] + right_points[1][0]) / 2),
            left_points[1][1],
        )
        lane_center_bottom = (metrics["lane_center_x"], left_points[0][1])

        cv2.line(guided, lane_center_top, lane_center_bottom, (255, 255, 255), 2)
        cv2.circle(guided, lane_center_bottom, 7, (255, 255, 255), -1)

    vehicle_center = (metrics["vehicle_center_x"], image.shape[0] - 1)
    lane_center = (metrics["lane_center_x"], image.shape[0] - 1)

    cv2.line(
        guided,
        (metrics["vehicle_center_x"], int(image.shape[0] * 0.62)),
        vehicle_center,
        (0, 0, 255),
        2,
    )
    cv2.circle(guided, vehicle_center, 7, (0, 0, 255), -1)
    cv2.line(guided, vehicle_center, lane_center, (0, 255, 255), 2)

    draw_info_panel(guided, metrics, curvature)

    return guided


def process_frame(frame):
    image = cv2.resize(frame, (800, 500))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 120)
    roi = region_of_interest(edges)

    lines = cv2.HoughLinesP(
        roi,
        1,
        np.pi / 180,
        threshold=15,
        minLineLength=20,
        maxLineGap=50,
    )

    left_lines, right_lines = classify_lane_lines(image, lines)
    left_avg = average_line(left_lines)
    right_avg = average_line(right_lines)
    left_sample_points = sample_points_from_lines(left_lines)
    right_sample_points = sample_points_from_lines(right_lines)
    left_curve = fit_lane_curve(left_sample_points)
    right_curve = fit_lane_curve(right_sample_points)
    left_curve_points = generate_curve_points(image, left_curve)
    right_curve_points = generate_curve_points(image, right_curve)
    left_points = line_points_from_params(image, left_avg)
    right_points = line_points_from_params(image, right_avg)

    if left_curve_points is not None:
        left_points = (left_curve_points[-1], left_curve_points[0])
    if right_curve_points is not None:
        right_points = (right_curve_points[-1], right_curve_points[0])

    line_image = np.zeros_like(image)
    debug = np.zeros_like(image)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(debug, (x1, y1), (x2, y2), (0, 255, 255), 2)

    if left_curve_points is not None:
        cv2.polylines(line_image, [np.array(left_curve_points, dtype=np.int32)], False, (255, 0, 0), 6)
    else:
        draw_full_line(line_image, left_avg, (255, 0, 0))

    if right_curve_points is not None:
        cv2.polylines(line_image, [np.array(right_curve_points, dtype=np.int32)], False, (0, 255, 0), 6)
    else:
        draw_full_line(line_image, right_avg, (0, 255, 0))

    lane_overlay, _ = fill_lane_area(
        image,
        left_points,
        right_points,
        left_curve_points=left_curve_points,
        right_curve_points=right_curve_points,
    )
    lane_image = cv2.addWeighted(lane_overlay, 0.8, line_image, 1, 1)

    metrics = calculate_lane_metrics(image, left_points, right_points)
    curvature = estimate_lane_curvature(image, left_curve, right_curve, left_curve_points, right_curve_points)
    if metrics is not None:
        lane_image = draw_lane_guidance(lane_image, left_points, right_points, metrics, curvature)

    return image, edges, roi, debug, lane_image


def main():
    capture = cv2.VideoCapture(VIDEO_PATH)
    if not capture.isOpened():
        raise SystemExit(f"Error: Video not found or cannot be opened: {VIDEO_PATH}")

    writer = None
    frame_count = 0
    latest_views = None
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    display_delay = get_display_delay_ms(fps)

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        frame_count += 1
        should_process = (
            latest_views is None
            or PROCESS_EVERY_NTH_FRAME <= 1
            or frame_count % PROCESS_EVERY_NTH_FRAME == 0
        )

        if should_process:
            latest_views = process_frame(frame)

        image, edges, roi, debug, lane_image = latest_views

        if writer is None:
            frame_height, frame_width = lane_image.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

        writer.write(lane_image)

        cv2.imshow("Original", image)
        cv2.imshow("Edges", edges)
        cv2.imshow("ROI", roi)
        cv2.imshow("Raw Lines", debug)
        cv2.imshow("Lane Detection", lane_image)

        if cv2.waitKey(display_delay) & 0xFF == 27:
            break

    capture.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
