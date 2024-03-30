import cv2
import numpy as np

# Load your image (replace 'image_path.jpg' with your actual image path)
image = cv2.imread('Image/rectangles.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect edges using Canny
edges = cv2.Canny(gray, 50, 150)

# Find contours (rectangles)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize a dictionary to store rectangle lengths and their corresponding approximated contours
rectangle_data = {}

for i, contour in enumerate(contours):
    # Approximate the contour to a polygon
    epsilon = 0.05 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Assuming each rectangle has only one line (adjust as needed)
    if len(approx) == 4:
        # Calculate line length (you can customize this part)
        p1, p2 = approx[0][0], approx[1][0]
        length = np.linalg.norm(p2 - p1)

        # Store the length and the approximated contour in the dictionary
        rectangle_data[i] = {'length': length, 'approx': approx}

# Sort rectangles by length
sorted_rectangles = sorted(rectangle_data.keys(), key=lambda k: rectangle_data[k]['length'])

# Assign numbers and write them below the rectangles
for rect, number in enumerate(sorted_rectangles, start=1):
    # Safely get the approximated contour for the current rectangle
    approx = rectangle_data.get(rect, {}).get('approx')

    if approx is not None:
        # Get the center of the rectangle
        center_x = int((approx[0][0][0] + approx[2][0][0]) / 2)
        center_y = int((approx[0][0][1] + approx[2][0][1]) / 2)

        # Write the number below the image
        cv2.putText(image, str(number), (center_x, center_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    else:
        print(f"No data found for rectangle {rect}")
# Show the image with numbers
cv2.imshow('Numbered Rectangles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
