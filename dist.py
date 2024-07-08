import cv2
import numpy as np

def find_marker(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 35, 125)

    # Find the contours of the edged image
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)

    return cv2.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
    # Compute and return the distance from the marker to the camera
    if perWidth == 0:
        return 0
    return (knownWidth * focalLength) / perWidth

# Known parameters (adjust these to your known reference dimensions)
KNOWN_DISTANCE = 24.0  # distance from camera to object in inches
KNOWN_WIDTH = 11.0     # width of the object in inches

# Start video capture
cap = cv2.VideoCapture(0)  # 0 is usually the default camera

# Read the first frame to initialize the focal length calculation
ret, reference_image = cap.read()
if not ret:
    print("Failed to capture initial frame from camera.")
    cap.release()
    exit(1)

marker = find_marker(reference_image)
if marker is not None:
    focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
else:
    print("No marker found in the initial frame.")
    cap.release()
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    marker = find_marker(frame)
    if marker is not None:
        inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])

        box = cv2.boxPoints(marker)
        box = np.int0(box)
        cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)
        cv2.putText(frame, f"{inches:.2f}in", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
