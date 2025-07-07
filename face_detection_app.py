import streamlit as st
from opencv-python-headless import cv2
import numpy as np
from PIL import Image
import io
import os

# Set page configuration
st.set_page_config(page_title="Face Detection App", layout="wide")

# Title and instructions
st.title("Face Detection with Viola-Jones Algorithm")
st.markdown("""
### Instructions
1. **Upload an Image**: Use the file uploader to select an image (JPG, PNG, etc.) containing faces.
2. **Adjust Parameters**:
   - **Scale Factor**: Controls how much the image size is reduced at each scale (1.01–2.0). Higher values speed up detection but may miss faces.
   - **Min Neighbors**: Specifies how many neighbors each candidate rectangle should have (1–10). Higher values reduce false positives but may miss faces.
   - **Rectangle Color**: Choose the color for the rectangles drawn around detected faces.
3. **View Results**: The app will display the image with detected faces.
4. **Download Image**: Click the "Download Image" button to save the processed image with detected faces.
""")

# Load the Haar Cascade Classifier for face detection
@st.cache_resource
def load_cascade():
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(cascade_path)

face_cascade = load_cascade()

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Parameters for face detection
st.subheader("Detection Parameters")
scale_factor = st.slider("Scale Factor", min_value=1.01, max_value=2.0, value=1.3, step=0.01)
min_neighbors = st.slider("Min Neighbors", min_value=1, max_value=10, value=5, step=1)
rect_color = st.color_picker("Rectangle Color", value="#FF0000")  # Default red

def hex_to_bgr(hex_color):
    """Convert hex color to BGR format for OpenCV."""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])  # Convert RGB to BGR

def detect_faces(image, scale_factor, min_neighbors, rect_color_bgr):
    """Detect faces in the image and draw rectangles."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), rect_color_bgr, 2)
    
    return image, len(faces)

# Main logic
if uploaded_file is not None:
    try:
        # Read and process the image
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Detect faces
        rect_color_bgr = hex_to_bgr(rect_color)
        processed_image, num_faces = detect_faces(image_bgr, scale_factor, min_neighbors, rect_color_bgr)
        
        # Convert back to RGB for display
        processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        
        # Display results
        st.subheader("Processed Image")
        st.image(processed_image_rgb, caption=f"Detected {num_faces} face(s)", use_column_width=True)
        
        # Save and download option
        st.subheader("Download Processed Image")
        output_filename = f"detected_faces_{uploaded_file.name}"
        _, img_buffer = cv2.imencode(".png", processed_image)
        img_bytes = img_buffer.tobytes()
        
        st.download_button(
            label="Download Image",
            data=img_bytes,
            file_name=output_filename,
            mime="image/png"
        )
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please upload an image to proceed.")
