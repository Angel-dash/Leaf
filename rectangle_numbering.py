import cv2
import numpy as np

# Load your image (replace 'image_path.jpg' with your actual image path)
image = cv2.imread('Image/rectangles.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect edges using Canny
edges = cv2.Canny(gray, 50, 150)

# Find contours (rectangles)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize a dictionary to store rectangle lengths
rectangle_lengths = {}

for i, contour in enumerate(contours):
    # Approximate the contour to a polygon
    epsilon = 0.05 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Assuming each rectangle has only one line (adjust as needed)
    if len(approx) == 4:
        # Calculate line length (you can customize this part)
        p1, p2 = approx[0][0], approx[1][0]
        length = np.linalg.norm(p2 - p1)

        # Store the length in the dictionary
        rectangle_lengths[i] = length

# Sort rectangles by length
sorted_rectangles = sorted(rectangle_lengths.keys(), key=lambda k: rectangle_lengths[k])

# Assign numbers
numbering = {sorted_rectangles[i]: i + 1 for i in range(len(sorted_rectangles))}

# Print the assigned numbers
for rect, number in numbering.items():
    print(f"Rectangle {rect + 1}: Number {number}")
