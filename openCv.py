import cv2
import numpy as np

# Load the image
image = cv2.imread('Image/rectangles.png')
img_copy = image.copy()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# Apply Canny edge detection
canny = cv2.Canny(gray, 125, 175)
cv2.imshow('Canny', canny)

# Find outer rectangle contours
outer_contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"The number of contours in the image: {len(outer_contours)}")

# Draw contours on the image copy
cv2.drawContours(img_copy, outer_contours, -1, (0, 255, 0), 2)  # Draw all contours in green color

# Define thresholds for filtering inner contours
min_area = 10  # Adjust this value based on your requirements
min_aspect_ratio = 10  # Adjust this value based on your requirements

# Create a blank image to draw inner contours
inner_contours_img = np.zeros_like(image)

for outer_contour in outer_contours:
    # Create a mask for the current outer rectangle
    mask = np.zeros_like(thresh)
    cv2.fillPoly(mask, [outer_contour], 255)

    # Find inner contours within the masked region
    inner_contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter inner contours to find line segments
    for inner_contour in inner_contours:
        area = cv2.contourArea(inner_contour)
        length = cv2.arcLength(inner_contour, True)
        aspect_ratio = length / area  # or any other property you need

        if area > min_area and aspect_ratio > min_aspect_ratio:
            # This inner contour is likely a line segment
            # Draw the inner contour on the separate image
            cv2.drawContours(inner_contours_img, [inner_contour], -1, (0, 0, 255), 2)

# Display the image with contours
cv2.imshow('Outer_Contours', img_copy)
cv2.imshow('Inner Contours', inner_contours_img)
cv2.waitKey(0)
cv2.destroyAllWindows()