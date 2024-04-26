import cv2

# Load the image
img = cv2.imread('Image/rectangles.png')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop through contours
for cnt in contours:
    # Filter out the rectangles
    if cv2.contourArea(cnt) > 100:  # Adjust the area threshold as needed
        # Find the line inside the rectangle
        line_contour = max(cv2.findContours(thresh[cv2.boundingRect(cnt)], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.arcLength)
        
        # Calculate the length of the line
        line_length = cv2.arcLength(line_contour, True)
        print(f"Line length: {line_length:.2f}")