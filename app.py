import streamlit as st
from PIL import Image
import os
from yolo_detector import YOLODetector
from llm_processor import LLMProcessor

# Initialize models
@st.cache_resource
def load_models():
    return YOLODetector(), LLMProcessor()

def save_uploaded_file(uploaded_file):
    """Save uploaded file to the uploads folder and return the path."""
    try:
        os.makedirs("uploads", exist_ok=True)
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Failed to save file: {str(e)}")
        return None

def main():
    st.title("Image Analysis and Text Generation App")
    st.write("Upload an image and provide a text prompt to get object detection results and a generated text response.")

    # Load models
    try:
        yolo_detector, llm_processor = load_models()
    except Exception as e:
        st.error(f"Failed to initialize models: {str(e)}")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    user_prompt = st.text_area("Enter your text prompt:", "Describe a scene involving the objects in the image.")

    if st.button("Process"):
        if uploaded_file is None:
            st.error("Please upload an image.")
            return
        if not user_prompt.strip():
            st.error("Please provide a text prompt.")
            return

        # Save and process image
        file_path = save_uploaded_file(uploaded_file)
        if file_path:
            try:
                # Display uploaded image
                image = Image.open(file_path)
                st.image(image, caption="Uploaded Image", use_column_width=True)

                # Detect objects
                with st.spinner("Detecting objects..."):
                    detections = yolo_detector.detect_objects(image)
                if isinstance(detections, dict) and "error" in detections:
                    st.error(detections["error"])
                    return
                st.subheader("Detected Objects:")
                for detection in detections:
                    st.write(f"- {detection['label']}: {detection['confidence']:.2f}")

                # Generate text response
                with st.spinner("Generating text response..."):
                    response = llm_processor.generate_description(user_prompt, detections)
                st.subheader("Generated Text Response:")
                st.write(response)

            except Exception as e:
                st.error(f"Processing failed: {str(e)}")

if __name__ == "__main__":
    main()
