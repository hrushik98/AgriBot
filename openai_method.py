import streamlit as st
import cv2
import numpy as np
import base64
from openai import OpenAI
import os
import time
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Weed Detection System",
    page_icon="🌿",
    layout="wide"
)

# OpenAI API key
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

# Create necessary directories if they don't exist
if not os.path.exists("to_train"):
    os.makedirs("to_train")

# Function to encode image for OpenAI API
def encode_image(image_array):
    # Convert numpy array to PIL Image
    image = Image.fromarray(image_array)
    # Create a bytes buffer for the image
    buffered = io.BytesIO()
    # Save the image to the buffer in JPEG format
    image.save(buffered, format="JPEG")
    # Get the byte data
    img_bytes = buffered.getvalue()
    # Encode to base64
    return base64.b64encode(img_bytes).decode("utf-8")

# Function to analyze image with OpenAI
def detect_weed(image):
    base64_image = encode_image(image)
    
    prompt = """
    Analyze this image and determine if it shows a weed.

    Definition: A weed is an unwanted plant growing in a location where it competes with cultivated plants for resources (water, light, nutrients) or interferes with agricultural operations.

    Instructions:
    1. Examine the plant species visible in the image
    2. Consider the context (agricultural field, garden, natural area, etc.)
    3. Apply botanical knowledge to classify the plant
    4. Respond with ONLY one of these options:
       - "YES" - Confirmed weed species in an agricultural/garden context
       - "NO" - Beneficial or intentionally cultivated plant
       - "IDK" - Unable to determine due to image quality, incomplete view, or uncertain plant identity

    Do not provide explanations, descriptions, or additional text beyond the single-word response.
    """
    
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
        )
        
        result = completion.choices[0].message.content.strip()
        return result
    except Exception as e:
        st.error(f"Error with OpenAI API: {e}")
        return "ERROR"

# Function to save image to training folder
def save_to_train(image, weed_name=None):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{weed_name}_{timestamp}.jpg" if weed_name else f"weed_{timestamp}.jpg"
    filepath = os.path.join("to_train", filename)
    
    # Convert the image from BGR to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Save the image
    cv2.imwrite(filepath, image_rgb)
    return filepath

# Sidebar for navigation
st.sidebar.title("Weed Detection System")
st.sidebar.image("https://img.icons8.com/color/96/000000/plant.png", width=100)

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Stream", "Test and Train"])

# Main content
if page == "Stream":
    st.header("Bot Video Stream")
    
    # Information about the stream
    st.info("This section will display the live video feed from the Solar-Powered Autonomous Weed Removal Machine.")
    
    # Placeholder for video stream
    video_placeholder = st.empty()
    
    # Simulated video stream (placeholder)
    st.warning("Video stream is not connected. This is a placeholder.")
    
    # Display static placeholder image
    placeholder_image = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add text to the placeholder
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(placeholder_image, "Waiting for video stream...", (120, 240), font, 1, (255, 255, 255), 2)
    
    video_placeholder.image(placeholder_image, channels="BGR", use_column_width=True)
    
    # Additional information about the project
    st.subheader("About the System")
    st.write("""
    The Solar-Powered Autonomous Weed Removal Machine is an eco-friendly farming solution that uses 
    solar energy and smart technology to detect and remove weeds without chemicals or manual work.
    
    This system uses YOLOv11 for weed detection with a Raspberry Pi 5 as the core processing unit.
    """)

elif page == "Test and Train":
    st.header("Test and Train Weed Detection")
    
    # Instructions
    st.write("""
    This section allows you to test the weed detection system and train it when false negatives occur.
    1. Use your camera to capture plants
    2. The system will analyze the image and determine if it's a weed
    3. If the system fails to identify a weed (false negative), you can save the image for training
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Camera Input")
        # Camera input options
        input_method = st.radio("Select input method:", ["Webcam", "Upload Image"])
        
        if input_method == "Webcam":
            # Add a button to capture image
            img_file_buffer = st.camera_input("Take a picture of a plant")
            
            if img_file_buffer is not None:
                # Convert the file buffer to opencv Image
                bytes_data = img_file_buffer.getvalue()
                img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                
                # Display the captured image
                st.image(img, channels="BGR", caption="Captured Image", use_column_width=True)
                
                # Store the image in session state for analysis
                st.session_state.current_image = img
                st.session_state.analysis_ready = True
                
        else:  # Upload Image
            uploaded_file = st.file_uploader("Upload an image of a plant", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                # Convert the file buffer to opencv Image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                # Display the uploaded image
                st.image(img, channels="BGR", caption="Uploaded Image", use_column_width=True)
                
                # Store the image in session state for analysis
                st.session_state.current_image = img
                st.session_state.analysis_ready = True
    
    with col2:
        st.subheader("Analysis Results")
        
        # Check if an image is ready for analysis
        if 'analysis_ready' in st.session_state and st.session_state.analysis_ready:
            with st.spinner("Analyzing image with AI..."):
                # Analyze the image
                result = detect_weed(st.session_state.current_image)
                
                # Display the result
                if result == "YES":
                    st.success("✅ WEED DETECTED")
                    st.session_state.is_weed = True
                elif result == "NO":
                    st.error("❌ NOT A WEED")
                    st.session_state.is_weed = False
                elif result == "IDK":
                    st.warning("⚠️ UNABLE TO DETERMINE")
                    st.session_state.is_weed = None
                else:
                    st.error(f"Unexpected response: {result}")
                    st.session_state.is_weed = None
                
                # Option to train if the user disagrees
                st.write("---")
                st.write("Is this result incorrect? (False negative)")
                
                col_yes, col_no = st.columns(2)
                
                with col_yes:
                    if st.button("Yes, this is a weed"):
                        if 'is_weed' in st.session_state and not st.session_state.is_weed:
                            # It's a false negative, allow training
                            weed_name = st.text_input("Enter weed name (optional):")
                            if st.button("Save for Training"):
                                saved_path = save_to_train(st.session_state.current_image, weed_name)
                                st.success(f"Image saved for training at {saved_path}")
                                # Reset for next image
                                st.session_state.analysis_ready = False
                        else:
                            st.info("No action needed - already identified as a weed or undetermined.")
                
                with col_no:
                    if st.button("No, result is correct"):
                        st.info("Thank you for confirming! Ready for next image.")
                        # Reset for next image
                        st.session_state.analysis_ready = False
        else:
            st.info("Capture or upload an image to begin analysis.")
    
    # Display training stats
    st.subheader("Training Data Statistics")
    try:
        num_training_images = len([f for f in os.listdir("to_train") if f.endswith(('.jpg', '.jpeg', '.png'))])
        st.write(f"Number of images saved for training: {num_training_images}")
    except:
        st.write("Training data directory not accessible.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>Solar-Powered Autonomous Weed Removal Machine Project</p>
    <p>Department of Electronics and Communication Engineering</p>
    <p>MATRUSRI ENGINEERING COLLEGE, AUTONOMOUS</p>
</div>
""", unsafe_allow_html=True)