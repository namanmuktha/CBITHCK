import streamlit as st
import cv2
import numpy as np
import string
import joblib
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Define the image processing functions
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

# Load models and preprocessors for text processing
glove_model = joblib.load('glove_model.pkl')
svd_model = joblib.load('svd_model.pkl')
scaler = joblib.load('scaler.pkl')
restored_model = load_model('my_model.h5')

# Load NLTK resources
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Define text processing functions
def document_vector(words):
    valid_words = [word for word in words if word in glove_model]
    if not valid_words:
        return np.zeros(glove_model.vector_size)
    return np.mean(glove_model[valid_words], axis=0)

def adding_stemming(words):
    return [stemmer.stem(word) for word in words]

def preprocess_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    text = text.split()
    text = [word for word in text if word not in stop_words]
    text = [lemmatizer.lemmatize(word) for word in text]
    text = adding_stemming(text)
    text = ' '.join(text)
    return text

def predict_length(text):
    processed_words = preprocess_text(text)
    doc_vec = document_vector(processed_words.split())
    doc_vec = doc_vec.reshape(1, -1)
    reduced_vec = svd_model.transform(doc_vec)
    scaled_vec = scaler.transform(reduced_vec)
    prediction = restored_model.predict(scaled_vec)
    return prediction[0][0]  # Assuming the model outputs a single value

# Streamlit UI
st.title('Product Dimension Analysis App')
user_input = st.text_area("Enter your text here for product description", "Type Here")
predicted_length = None

if st.button('Predict Length'):
    predicted_length = predict_length(user_input)
    st.write(f'Predicted Product Length: {predicted_length:.2f} cm')

uploaded_file = st.file_uploader("Upload an image of the product", type=['png', 'jpg', 'jpeg'])

if uploaded_file and predicted_length:
    # Convert the uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    contours = find_objects(image)
    pixelsPerMetric = None

    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue

        if pixelsPerMetric is None:
            # Initialize the pixel-to-metric conversion factor with the predicted length
            pixelsPerMetric = cv2.arcLength(contour, True) / predicted_length

        width, height, rect = get_dimensions(contour, pixelsPerMetric)
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
        cv2.putText(image, f"{width:.1f}cm x {height:.1f}cm", (int(rect[0][0]), int(rect[0][1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    # Convert the colors from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image, caption='Processed Image with Dimensions', use_column_width=True)
