import streamlit as st
import cv2
import numpy as np

def find_objects(image):
    # Convert the image to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 50, 100)

    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_dimensions(contour, pixelsPerMetric):
    # Calculate the minimum area rectangle for the contour
    box = cv2.minAreaRect(contour)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    # Order the points in the contour
    rect = np.zeros((4, 2), dtype="float32")
    s = box.sum(axis=1)
    rect[0] = box[np.argmin(s)]
    rect[2] = box[np.argmax(s)]

    diff = np.diff(box, axis=1)
    rect[1] = box[np.argmin(diff)]
    rect[3] = box[np.argmax(diff)]

    # Calculate the distances between the points
    width = np.linalg.norm(rect[1] - rect[0])
    height = np.linalg.norm(rect[2] - rect[1])

    if pixelsPerMetric is None:
        # Return pixel dimensions if no scale is provided
        return width, height, rect
    # Convert dimensions to real-world measurements
    return width / pixelsPerMetric, height / pixelsPerMetric, rect

def main():
    st.title("Object Dimension Measurement App")
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        # Convert the uploaded file to an OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # User input for calibration
        KNOWN_WIDTH = st.number_input("Enter the known width of a reference object in cm:", value=21.59, format="%.2f")
        pixelsPerMetric = None

        if st.button("Measure Dimensions"):
            contours = find_objects(image)
            for contour in contours:
                if cv2.contourArea(contour) < 100:
                    continue

                if pixelsPerMetric is None:
                    # Initialize the pixel-to-metric conversion factor
                    pixelsPerMetric = cv2.arcLength(contour, True) / KNOWN_WIDTH

                width, height, rect = get_dimensions(contour, pixelsPerMetric)
                cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
                cv2.putText(image, f"{width:.1f}cm x {height:.1f}cm", (int(rect[0][0]), int(rect[0][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            # Convert the colors from BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image, caption='Processed Image with Dimensions', use_column_width=True)

if __name__ == "__main__":
    main()
