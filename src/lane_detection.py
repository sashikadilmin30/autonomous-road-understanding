import cv2
import numpy as np

# Load image
image = cv2.imread("data/road.jpg")

if image is None:
    print("Error: Image not found")
    exit()

# Resize image
image = cv2.resize(image, (800, 500))

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blur
blur = cv2.GaussianBlur(gray, (5,5), 0)

# Edge detection
edges = cv2.Canny(blur, 50, 150)

# -------- REGION OF INTEREST --------

height = edges.shape[0]
width = edges.shape[1]

mask = np.zeros_like(edges)

polygon = np.array([[
    (0, height),
    (width, height),
    (width//2, height//2)
]])

cv2.fillPoly(mask, polygon, 255)

roi = cv2.bitwise_and(edges, mask)

# ------------------------------------

cv2.imshow("Original", image)
cv2.imshow("Edges", edges)
cv2.imshow("ROI", roi)

cv2.imwrite("../results/roi_edges.jpg", roi)

cv2.waitKey(0)
cv2.destroyAllWindows()