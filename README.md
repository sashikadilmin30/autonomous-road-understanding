# Autonomous Road Understanding System Using Image Processing Techniques

## Project Overview

This project implements a robust lane detection and road curvature estimation pipeline using classical computer vision techniques. It is designed to detect and track left and right lane boundaries, smooth temporal lane predictions, compute vehicle offset, and estimate road curvature across multiple driving scenarios.

The system supports evaluation across scenario-specific datasets such as highway, curved roads, night driving, and rainy weather, while generating annotated outputs and CSV metrics for analysis.

## Features

* Lane Detection
* Road Curvature Estimation
* Vehicle Detection
* Pedestrian Detection
* Traffic Sign Detection
* Distance Estimation
* Collision Warning Generation
* Lane Departure Warning
* Road Scene Dashboard


## Tech Stack

- Python 3
- OpenCV for image/video processing and visualization
- NumPy for numeric operations and polynomial fitting
- CSV output for evaluation metrics

## Pipeline

1. **Input Processing**
   - Load image or video frame from `data/`
   - Resize to a standard working resolution
   - Convert to grayscale and blur for noise reduction
2. **Edge and ROI Extraction**
   - Apply Canny edge detection
   - Mask the image to a road-centered region of interest
3. **Lane Line Detection**
   - Use probabilistic Hough line transform to detect candidate lane segments
   - Classify segments into left and right lanes using slope and position
4. **Curve Fitting**
   - Sample points from lane segments
   - Fit second-order polynomials with `numpy.polyfit`
   - Generate smooth lane boundary curves
5. **Temporal Smoothing**
   - Maintain recent polynomial coefficients
   - Apply moving-average smoothing across frames to reduce flicker
6. **Metrics and Visualization**
   - Compute lane center, vehicle offset, lane width stability, and curvature
   - Fill the lane area and overlay guidance information
   - Present dashboard labels for lane offset, vehicle position, and road curvature

## Algorithms

- **Edge Detection**: Canny as a robust pre-processing step for line detection.
- **Hough Line Transform**: Extracts candidate line segments from edges.
- **Line Classification**: Separates left/right lane candidates based on slope, length, and position relative to the frame center.
- **Polynomial Lane Fitting**: Uses `np.polyfit(y, x, 2)` to fit quadratic lane boundaries in pixel space.
- **Curve Generation**: Samples the fitted polynomial to form smooth lane boundaries.
- **Crossing Removal**: Detects and trims overlapping lane curves to keep left lane left of right lane.
- **Temporal Averaging**: Maintains a short buffer of recent polynomial coefficients for stable lane predictions.
- **Curvature Estimation**: Computes radius of curvature from lane polynomial coefficients and reports it in meters.
- **Offset Estimation**: Computes vehicle offset from lane center and converts it to meters using lane width assumptions.

## Results

- Annotated lane detection output is saved separately for each scenario.
- Evaluation reports include FPS, detection rate, lane width stability, and curvature consistency.
- Metrics are written to `results/evaluation/metrics.csv` and per-scenario CSVs.

## How to Run

1. Activate your Python environment.
2. Install dependencies if needed (OpenCV, NumPy).
3. Run the main lane detection pipeline:

```bash
python src/lane_detection.py
```

4. Run scenario evaluation:

```bash
python evaluate.py --data data --output results/evaluation/metrics.csv
```

5. Place scenario datasets under the `data/` folder, for example:

- `data/highway/`
- `data/curved/`
- `data/night/`
- `data/rainy/`

## Future Improvements

- Add more robust lane detection for shadows, glare, and faded markings.
- Integrate perspective transformation for real-world lane geometry.
- Use camera calibration to improve meter-scale accuracy.
- Add ground truth comparison and per-frame accuracy metrics.
- Extend to multi-lane and junction detection.
- Add automated visualization dashboards or plots for scenario comparisons.
