import cv2
import numpy as np
from collections import deque
from pathlib import Path

VIDEO_PATH = "data/road.mp4"
OUTPUT_VIDEO_PATH = "results/lane_output.mp4"
DEBUG_WINDOWS = False
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
MIN_VEHICLE_AREA = 280
MAX_VEHICLE_AREA_RATIO = 0.08
MIN_VEHICLE_ASPECT_RATIO = 0.55
MAX_VEHICLE_ASPECT_RATIO = 4.0
MIN_VEHICLE_EXTENT = 0.36
MAX_LANE_MARKING_WHITE_RATIO = 0.18
MIN_VEHICLE_DARK_RATIO = 0.08
LANE_WIDTH_METERS = 3.7
LANE_DEPARTURE_WARNING_THRESHOLD_M = 0.35
DASHBOARD_SMOOTHING_WINDOW = 8
LOW_LIGHT_BRIGHTNESS_GAIN = 1.18
LOW_LIGHT_BRIGHTNESS_BIAS = 14
CLAHE_CLIP_LIMIT = 2.4
CLAHE_TILE_GRID_SIZE = (8, 8)
LOW_LIGHT_MEAN_THRESHOLD = 115
LOW_LIGHT_VALUE_THRESHOLD = 130
MIN_SIGN_AREA = 90
MAX_SIGN_AREA_RATIO = 0.025
MIN_SIGN_CIRCULARITY = 0.62
SIGN_ASPECT_RATIO_MIN = 0.55
SIGN_ASPECT_RATIO_MAX = 1.45

boundary_pair_history = deque(maxlen=SMOOTHING_WINDOW)
thumbnail_history = deque(maxlen=THUMBNAIL_COUNT)
curvature_history = deque(maxlen=DASHBOARD_SMOOTHING_WINDOW)
offset_history = deque(maxlen=DASHBOARD_SMOOTHING_WINDOW)
vehicle_count_history = deque(maxlen=DASHBOARD_SMOOTHING_WINDOW)


class VehicleDetector:
    def __init__(self):
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=350,
            varThreshold=48,
            detectShadows=True,
        )
        self.open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 7))
        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))

    def detect(self, frame, road_polygon=None):
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        foreground = self.background_subtractor.apply(blurred)
        _, motion_mask = cv2.threshold(foreground, 200, 255, cv2.THRESH_BINARY)

        road_mask = self.create_road_mask(frame.shape[:2], road_polygon)
        motion_mask = cv2.bitwise_and(motion_mask, road_mask)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, self.open_kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, self.close_kernel, iterations=2)
        motion_mask = cv2.dilate(motion_mask, self.dilate_kernel, iterations=1)

        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = self.filter_vehicle_contours(contours, frame)
        boxes.extend(self.detect_static_candidates(frame, road_mask))
        return self.merge_overlapping_boxes(boxes), motion_mask

    def create_road_mask(self, shape, road_polygon):
        height, width = shape
        mask = np.zeros((height, width), dtype=np.uint8)

        if road_polygon is None:
            road_polygon = np.array(
                [[
                    (int(width * 0.08), height),
                    (int(width * 0.92), height),
                    (int(width * 0.62), int(height * LANE_TOP_RATIO)),
                    (int(width * 0.38), int(height * LANE_TOP_RATIO)),
                ]],
                dtype=np.int32,
            )

        cv2.fillPoly(mask, road_polygon, 255)
        return mask

    def filter_vehicle_contours(self, contours, frame):
        frame_shape = frame.shape
        height, width = frame_shape[:2]
        max_area = width * height * MAX_VEHICLE_AREA_RATIO
        boxes = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_VEHICLE_AREA or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if h == 0:
                continue

            extent = area / max(1, w * h)
            if extent < MIN_VEHICLE_EXTENT:
                continue

            aspect_ratio = w / h
            if aspect_ratio < MIN_VEHICLE_ASPECT_RATIO or aspect_ratio > MAX_VEHICLE_ASPECT_RATIO:
                continue
            if y < int(height * LANE_TOP_RATIO) or y + h > height - 8:
                continue
            if self.is_lane_marking_false_positive(frame[y:y + h, x:x + w]):
                continue

            boxes.append((x, y, w, h))

        return self.merge_overlapping_boxes(boxes)

    def is_lane_marking_false_positive(self, patch):
        if patch.size == 0:
            return True

        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 165]), np.array([180, 90, 255]))
        dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 95]))

        patch_area = patch.shape[0] * patch.shape[1]
        white_ratio = np.count_nonzero(white_mask) / max(1, patch_area)
        dark_ratio = np.count_nonzero(dark_mask) / max(1, patch_area)

        if white_ratio > MAX_LANE_MARKING_WHITE_RATIO and dark_ratio < MIN_VEHICLE_DARK_RATIO:
            return True

        edges = cv2.Canny(white_mask, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            18,
            minLineLength=max(18, int(min(patch.shape[:2]) * 0.65)),
            maxLineGap=8,
        )
        if lines is None:
            return False

        diagonal = np.hypot(patch.shape[1], patch.shape[0])
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_length = np.hypot(x2 - x1, y2 - y1)
            if line_length > diagonal * 0.48 and white_ratio > 0.06:
                return True

        return False

    def merge_overlapping_boxes(self, boxes):
        if len(boxes) <= 1:
            return boxes

        rectangles = [[x, y, w, h] for x, y, w, h in boxes]
        rectangles, _ = cv2.groupRectangles(rectangles + rectangles, groupThreshold=1, eps=0.35)
        return [tuple(rect) for rect in rectangles]

    def detect_static_candidates(self, frame, road_mask):
        height, width = frame.shape[:2]
        roi_mask = road_mask.copy()
        roi_mask[:int(height * LANE_TOP_RATIO)] = 0
        roi_mask[int(height * 0.9):] = 0

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 115, 120]))
        dark_mask = cv2.bitwise_and(dark_mask, roi_mask)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, self.open_kernel)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, self.open_kernel, iterations=1)

        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 18 or area > 1800:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / max(1, h)
            if aspect_ratio < 0.45 or aspect_ratio > 3.8:
                continue
            if w < 4 or h < 4:
                continue

            boxes.append((x, y, w, h))

        return self.merge_overlapping_boxes(boxes)


vehicle_detector = VehicleDetector()


def enhance_low_light_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mean_value = float(np.mean(hsv[:, :, 2]))
    low_light_ratio = np.count_nonzero(hsv[:, :, 2] < LOW_LIGHT_VALUE_THRESHOLD) / hsv[:, :, 2].size

    if mean_value > LOW_LIGHT_MEAN_THRESHOLD and low_light_ratio < 0.58:
        denoised = cv2.GaussianBlur(frame, (3, 3), 0)
        brightness_gain = 1.03
        brightness_bias = 2
        clahe_clip_limit = 1.4
    else:
        denoised = cv2.bilateralFilter(frame, 5, 35, 35)
        brightness_gain = LOW_LIGHT_BRIGHTNESS_GAIN
        brightness_bias = LOW_LIGHT_BRIGHTNESS_BIAS
        clahe_clip_limit = CLAHE_CLIP_LIMIT

    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip_limit,
        tileGridSize=CLAHE_TILE_GRID_SIZE,
    )
    equalized_lightness = clahe.apply(lightness)
    enhanced_lab = cv2.merge((equalized_lightness, channel_a, channel_b))
    contrast_enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    brightened = cv2.convertScaleAbs(
        contrast_enhanced,
        alpha=brightness_gain,
        beta=brightness_bias,
    )
    detail = cv2.GaussianBlur(brightened, (0, 0), 1.0)
    return cv2.addWeighted(brightened, 1.25, detail, -0.25, 0)


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


def get_lane_area_polygon(left_boundary, right_boundary):
    if left_boundary is None or right_boundary is None:
        return None

    return np.array([inset_lane_fill(left_boundary, right_boundary)], dtype=np.int32)


def draw_lane_area(img, left_boundary, right_boundary):
    if left_boundary is None or right_boundary is None:
        return

    polygon = get_lane_area_polygon(left_boundary, right_boundary)

    overlay = img.copy()
    cv2.fillPoly(overlay, polygon, (80, 190, 55))
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)


def draw_vehicle_boxes(img, vehicle_boxes):
    for index, (x, y, w, h) in enumerate(vehicle_boxes, start=1):
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 180, 255), 2)
        cv2.rectangle(img, (x, y - 22), (x + 72, y), (0, 180, 255), -1)
        cv2.putText(img, f"Vehicle {index}", (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)


def red_sign_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = cv2.inRange(hsv, np.array([0, 70, 70]), np.array([12, 255, 255]))
    upper_red = cv2.inRange(hsv, np.array([168, 70, 70]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(lower_red, upper_red)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8), iterations=2)
    return mask


def classify_sign_shape(contour):
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None

    area = cv2.contourArea(contour)
    circularity = 4.0 * np.pi * area / (perimeter * perimeter)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

    if len(approx) == 3:
        return "Triangular"
    if circularity >= MIN_SIGN_CIRCULARITY and len(approx) >= 6:
        return "Circular"

    return None


def detect_traffic_signs(frame):
    mask = red_sign_mask(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = frame.shape[:2]
    max_area = width * height * MAX_SIGN_AREA_RATIO
    detected_signs = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_SIGN_AREA or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / max(1, h)
        if aspect_ratio < SIGN_ASPECT_RATIO_MIN or aspect_ratio > SIGN_ASPECT_RATIO_MAX:
            continue
        if y > int(height * 0.82):
            continue

        shape = classify_sign_shape(contour)
        if shape is None:
            continue

        detected_signs.append({
            "box": (x, y, w, h),
            "shape": shape,
            "area": area,
        })

    return detected_signs, mask


def draw_traffic_signs(img, traffic_signs):
    for sign in traffic_signs:
        x, y, w, h = sign["box"]
        label = "Traffic Sign Detected"
        label_y = max(18, y - 8)
        text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(img, (x, label_y - 18), (min(img.shape[1] - 1, x + text_width + 8), label_y + 2), (0, 0, 255), -1)
        cv2.putText(img, label, (x + 4, label_y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)


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


def boundary_x_at_y(boundary, y):
    x1, y1, x2, y2 = boundary
    if y2 == y1:
        return float(x1)

    ratio = (y - y1) / (y2 - y1)
    return float(x1 + (x2 - x1) * ratio)


def smooth_numeric_value(history, value):
    if value is None:
        return None

    history.append(float(value))
    return float(np.mean(history))


def smooth_vehicle_count(vehicle_count):
    vehicle_count_history.append(int(vehicle_count))
    return int(round(float(np.median(vehicle_count_history))))


def compute_lane_departure(img, left_boundary, right_boundary):
    if left_boundary is None or right_boundary is None:
        return {
            "lane_center_px": None,
            "vehicle_center_px": img.shape[1] / 2,
            "offset_m": None,
            "lane_status": "Searching",
            "vehicle_position": "N/A",
            "warning_state": "No Lane Lock",
        }

    measurement_y = img.shape[0] - 1
    left_x = boundary_x_at_y(left_boundary, measurement_y)
    right_x = boundary_x_at_y(right_boundary, measurement_y)
    lane_center = (left_x + right_x) / 2
    vehicle_center = img.shape[1] / 2
    lane_width_px = max(1, right_x - left_x)
    raw_offset_m = (vehicle_center - lane_center) / lane_width_px * LANE_WIDTH_METERS
    offset_m = smooth_numeric_value(offset_history, raw_offset_m)

    if offset_m <= -LANE_DEPARTURE_WARNING_THRESHOLD_M:
        lane_status = "Left Deviation"
        vehicle_position = "Left"
        warning_state = "Lane Departure Left"
    elif offset_m >= LANE_DEPARTURE_WARNING_THRESHOLD_M:
        lane_status = "Right Deviation"
        vehicle_position = "Right"
        warning_state = "Lane Departure Right"
    else:
        lane_status = "Centered"
        vehicle_position = "Center"
        warning_state = "None"

    return {
        "lane_center_px": lane_center,
        "vehicle_center_px": vehicle_center,
        "offset_m": offset_m,
        "lane_status": lane_status,
        "vehicle_position": vehicle_position,
        "warning_state": warning_state,
    }


def estimate_curvature_km(img, left_boundary, right_boundary):
    if left_boundary is None or right_boundary is None:
        return None

    bottom_y = img.shape[0] - 1
    mid_y = int(img.shape[0] * 0.75)
    top_y = int(img.shape[0] * LANE_TOP_RATIO)

    centers = []
    widths = []
    for y in (bottom_y, mid_y, top_y):
        left_x = boundary_x_at_y(left_boundary, y)
        right_x = boundary_x_at_y(right_boundary, y)
        centers.append((left_x + right_x) / 2)
        widths.append(max(1.0, right_x - left_x))

    average_width = max(1.0, float(np.mean(widths)))
    center_shift_ratio = abs(centers[0] - centers[-1]) / average_width
    bend_change_ratio = abs((centers[0] - centers[1]) - (centers[1] - centers[2])) / average_width

    curvature_km = 9.99 / (1.0 + center_shift_ratio * 3.0 + bend_change_ratio * 18.0)
    curvature_km = float(np.clip(curvature_km, 0.25, 9.99))
    return smooth_numeric_value(curvature_history, curvature_km)


def estimate_dashboard_metrics(img, left_boundary, right_boundary, lane_departure=None):
    if lane_departure is None:
        lane_departure = compute_lane_departure(img, left_boundary, right_boundary)
    if lane_departure["offset_m"] is None:
        return "N/A", "N/A", lane_departure

    curvature_km = estimate_curvature_km(img, left_boundary, right_boundary)
    offset_m = lane_departure["offset_m"]

    return f"{curvature_km:.2f} km", f"{offset_m:+.2f} m", lane_departure


def draw_lane_departure_warning(img, lane_departure):
    warning_state = lane_departure["warning_state"]
    if warning_state not in {"Lane Departure Left", "Lane Departure Right"}:
        return

    h, w = img.shape[:2]
    text_size, _ = cv2.getTextSize(warning_state, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    box_width = min(w - 32, text_size[0] + 52)
    x1 = (w - box_width) // 2
    y1 = 18
    x2 = x1 + box_width
    y2 = y1 + 54

    draw_translucent_panel(img, (x1, y1), (x2, y2), alpha=0.62)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(img, warning_state, (x1 + 26, y1 + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)


def draw_dashboard(img, left_boundary, right_boundary, vehicle_count=0, lane_departure=None):
    h, w = img.shape[:2]
    x1 = max(0, w - 250)
    y1 = 10
    x2 = w - 8
    y2 = 360
    draw_translucent_panel(img, (x1, y1), (x2, y2))

    curvature, offset, lane_departure = estimate_dashboard_metrics(img, left_boundary, right_boundary, lane_departure)
    stable_vehicle_count = smooth_vehicle_count(vehicle_count)

    cv2.putText(img, "DASHBOARD", (x1 + 28, y1 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(img, (x1 + 16, y1 + 52), (x2 - 16, y1 + 52), (230, 230, 230), 1)

    yellow = (0, 230, 255)
    green = (40, 255, 40)
    red = (0, 0, 255)
    white = (255, 255, 255)
    state_color = red if lane_departure["warning_state"].startswith("Lane Departure") else green

    cv2.putText(img, "Curvature", (x1 + 28, y1 + 84), cv2.FONT_HERSHEY_SIMPLEX, 0.52, yellow, 1, cv2.LINE_AA)
    cv2.putText(img, curvature, (x1 + 28, y1 + 108), cv2.FONT_HERSHEY_SIMPLEX, 0.52, white, 1, cv2.LINE_AA)
    cv2.putText(img, "Lateral Offset", (x1 + 28, y1 + 142), cv2.FONT_HERSHEY_SIMPLEX, 0.52, yellow, 1, cv2.LINE_AA)
    cv2.putText(img, offset, (x1 + 28, y1 + 166), cv2.FONT_HERSHEY_SIMPLEX, 0.52, white, 1, cv2.LINE_AA)
    cv2.putText(img, "Lane Status", (x1 + 28, y1 + 198), cv2.FONT_HERSHEY_SIMPLEX, 0.52, yellow, 1, cv2.LINE_AA)
    cv2.putText(img, lane_departure["lane_status"], (x1 + 28, y1 + 222), cv2.FONT_HERSHEY_SIMPLEX, 0.52, state_color, 1, cv2.LINE_AA)
    cv2.putText(img, "Vehicle Position", (x1 + 28, y1 + 254), cv2.FONT_HERSHEY_SIMPLEX, 0.52, yellow, 1, cv2.LINE_AA)
    cv2.putText(img, lane_departure["vehicle_position"], (x1 + 28, y1 + 278), cv2.FONT_HERSHEY_SIMPLEX, 0.52, white, 1, cv2.LINE_AA)
    cv2.putText(img, "Warning State", (x1 + 28, y1 + 310), cv2.FONT_HERSHEY_SIMPLEX, 0.52, yellow, 1, cv2.LINE_AA)
    cv2.putText(img, lane_departure["warning_state"], (x1 + 28, y1 + 334), cv2.FONT_HERSHEY_SIMPLEX, 0.42, state_color, 1, cv2.LINE_AA)
    cv2.putText(img, "Vehicles", (x1 + 168, y1 + 84), cv2.FONT_HERSHEY_SIMPLEX, 0.52, yellow, 1, cv2.LINE_AA)
    cv2.putText(img, str(stable_vehicle_count), (x1 + 168, y1 + 108), cv2.FONT_HERSHEY_SIMPLEX, 0.52, white, 1, cv2.LINE_AA)


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


def process_frame(frame, include_debug=False):
    original_image = cv2.resize(frame, (800, 500))
    enhanced_image = enhance_low_light_frame(original_image)
    image = enhanced_image

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

    lane_image = original_image.copy()
    left_boundary = average_boundary_line(left_lines, image.shape[0], white_edges)
    right_boundary = average_boundary_line(right_lines, image.shape[0], white_edges)
    left_boundary = constrain_boundary_to_outer_band(left_boundary, image.shape[1], "left")
    right_boundary = constrain_boundary_to_outer_band(right_boundary, image.shape[1], "right")
    left_boundary, right_boundary = smooth_boundary_pair(left_boundary, right_boundary)
    metric_left_boundary, metric_right_boundary = left_boundary, right_boundary
    display_left_boundary, display_right_boundary = adjust_boundaries_for_display(left_boundary, right_boundary, image.shape[1])

    lane_polygon = get_lane_area_polygon(display_left_boundary, display_right_boundary)
    vehicle_boxes, vehicle_mask = vehicle_detector.detect(image, lane_polygon)
    traffic_signs, traffic_sign_mask = detect_traffic_signs(original_image)

    draw_lane_area(lane_image, display_left_boundary, display_right_boundary)
    draw_boundary(lane_image, display_left_boundary, (255, 0, 0))
    draw_boundary(lane_image, display_right_boundary, (0, 255, 0))
    draw_vehicle_boxes(lane_image, vehicle_boxes)
    draw_traffic_signs(lane_image, traffic_signs)
    draw_legend(lane_image)
    lane_departure = compute_lane_departure(lane_image, metric_left_boundary, metric_right_boundary)
    draw_lane_departure_warning(lane_image, lane_departure)
    draw_dashboard(lane_image, metric_left_boundary, metric_right_boundary, len(vehicle_boxes), lane_departure)

    output_frame = compose_output_frame(lane_image)

    if include_debug:
        debug_views = {
            "Lane Detection": output_frame,
            "Original Frame": original_image,
            "Enhanced Frame": enhanced_image,
            "Vehicle Motion Mask": vehicle_mask,
            "Traffic Sign Red Mask": traffic_sign_mask,
            "Lane ROI Edges": roi,
        }
        return output_frame, debug_views

    return output_frame


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

        if DEBUG_WINDOWS:
            lane_image, debug_views = process_frame(frame, include_debug=True)
        else:
            lane_image = process_frame(frame)
            debug_views = {}

        if writer is None:
            frame_height, frame_width = lane_image.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

        writer.write(lane_image)

        if DEBUG_WINDOWS:
            for name, view in debug_views.items():
                cv2.imshow(name, view)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    capture.release()
    if writer is not None:
        writer.release()
    if DEBUG_WINDOWS:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
