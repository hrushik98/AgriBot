from inference import get_model
import supervision as sv
import cv2
import time
import os
import threading
import subprocess
import signal
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
import socket

# Global variables
latest_jpg = None
stop_server = False
weed_detected = False
platform = "laptop"  # Default to laptop mode, will be set in main function

# Roboflow API key for weed detection model
os.environ["ROBOFLOW_API_KEY"] = "78zUzBtrpgjMypEYxfHI"

# Fixed port for ngrok
PORT = 80

# Conditional imports and setup based on platform
class DummyLCD:
    def clear(self): pass
    def write_string(self, text): print(f"LCD would display: {text}")
    def close(self): pass

# Initialize LCD - will be properly set based on platform
lcd = DummyLCD()

def setup_platform(platform_type):
    """Set up environment based on platform (laptop or rpi)"""
    global lcd, platform
    
    platform = platform_type.lower()
    print(f"Setting up for platform: {platform}")
    
    if platform == "rpi":
        try:
            # Check for required packages
            try:
                import psutil
            except ImportError:
                print("Installing required package: psutil")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
                
            try:
                from RPLCD.i2c import CharLCD
                import smbus
            except ImportError:
                print("Installing required package: RPLCD")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "RPLCD"])
                import smbus
                from RPLCD.i2c import CharLCD
            
            # Initialize LCD display
            try:
                # I2C bus=1 for Raspberry Pi
                lcd = CharLCD(i2c_expander='PCF8574', address=0x27, port=1, 
                            cols=16, rows=2, dotsize=8,
                            charmap='A02',
                            auto_linebreaks=True,
                            backlight_enabled=True)
                lcd.clear()
                lcd.write_string("Starting up...\nWeed Detection")
                print("LCD initialized successfully")
            except Exception as e:
                print(f"Error initializing LCD: {e}")
                lcd = DummyLCD()
        except Exception as e:
            print(f"Error setting up RPI platform: {e}")
            lcd = DummyLCD()
    else:
        # Laptop mode
        lcd = DummyLCD()
        print("Running in laptop mode (no LCD display)")

# Function to make sure camera is released on exit (RPI specific)
def ensure_camera_released(device="/dev/video0"):
    if platform != "rpi":
        return False  # Skip on laptop
    
    try:
        # Find processes using the camera
        cmd = f"lsof {device}"
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode == 0:  # Camera is in use
            for line in result.stdout.strip().split('\n')[1:]:  # Skip header line
                parts = line.split()
                if len(parts) > 1:
                    pid = parts[1]
                    try:
                        print(f"Killing process {pid} using camera")
                        subprocess.run(f"kill -9 {pid}", shell=True)
                    except Exception as e:
                        print(f"Error killing process {pid}: {e}")
            
            print(f"Released camera {device}")
            return True
        return False
    except Exception as e:
        print(f"Error releasing camera: {e}")
        return False

# Start ngrok service (RPI specific)
def start_ngrok():
    if platform != "rpi":
        print("Skipping ngrok in laptop mode")
        return
    
    try:
        # Kill any existing ngrok processes
        subprocess.run("pkill -f ngrok", shell=True, stderr=subprocess.DEVNULL)
        time.sleep(1)
        
        # Start ngrok with custom domain
        cmd = f"ngrok http --url=verified-allegedly-ram.ngrok-free.app {PORT}"
        subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Started ngrok with custom domain: verified-allegedly-ram.ngrok-free.app")
        
        # Update LCD
        lcd.clear()
        lcd.write_string("Ngrok started\nWaiting for cam")
    except Exception as e:
        print(f"Failed to start ngrok: {e}")

# HTTP Server for displaying frames
class FrameHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global latest_jpg
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = f'''
            <html>
            <head>
                <title>Weed Detection</title>
                <style>
                    body {{ margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; height: 100vh; background-color: #222; }}
                    img {{ max-width: 100%; max-height: 100vh; }}
                    .status {{ position: fixed; top: 10px; right: 10px; padding: 10px; background-color: rgba(0,0,0,0.7); color: white; border-radius: 5px; }}
                    .platform {{ position: fixed; top: 10px; left: 10px; padding: 10px; background-color: rgba(0,0,0,0.7); color: white; border-radius: 5px; }}
                </style>
                <script>
                    function refreshImage() {{
                        const img = document.getElementById('stream');
                        img.src = "/frame?" + new Date().getTime();
                        
                        // Also fetch detection status
                        fetch('/status')
                            .then(response => response.text())
                            .then(text => {{
                                document.getElementById('status').innerText = text;
                            }});
                    }}
                    setInterval(refreshImage, 100);
                </script>
            </head>
            <body>
                <div class="platform">Platform: {platform.upper()}</div>
                <img id="stream" src="/frame" alt="Weed Detection Stream" />
                <div id="status" class="status">No weeds detected</div>
            </body>
            </html>
            '''
            
            self.wfile.write(html.encode())
            
        elif self.path.startswith('/frame'):
            if latest_jpg is not None:
                self.send_response(200)
                self.send_header('Content-type', 'image/jpeg')
                self.end_headers()
                self.wfile.write(latest_jpg)
            else:
                self.send_response(503)
                self.end_headers()
                
        elif self.path == '/status':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            status = "WEEDS DETECTED!" if weed_detected else "No weeds detected"
            self.wfile.write(status.encode())
    
    def log_message(self, format, *args):
        return

# Start HTTP server in a thread
def run_server():
    global stop_server
    host = '0.0.0.0' if platform == 'rpi' else 'localhost'
    server = HTTPServer((host, PORT), FrameHandler)
    print(f"Web server started at http://{host}:{PORT}")
    
    while not stop_server:
        try:
            server.handle_request()
        except Exception as e:
            print(f"Server error: {e}")
            time.sleep(1)

# Enhanced camera initialization with improved focus
def initialize_camera():
    global latest_jpg
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        lcd.clear()
        lcd.write_string("Camera Error!\nCheck connection")
        exit(1)
    
    # Performance optimization - use smaller resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Camera warm-up time
    print("Warming up camera...")
    lcd.clear()
    lcd.write_string("Camera warmup...\nPlease wait")
    
    # Take several frames to let camera adjust
    for i in range(30):
        ret, frame = cap.read()
        time.sleep(0.1)
    
    # Try to improve focus - platform specific
    try:
        print("Optimizing camera focus...")
        
        if platform == "rpi":
            # RPI specific camera optimizations
            
            # Enable autofocus
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            
            # For Raspberry Pi camera, try using v4l2-ctl command for better focus control
            try:
                # Disable auto focus
                subprocess.run("v4l2-ctl --set-ctrl=focus_auto=0", shell=True)
                # Set focus to infinity (or appropriate value)
                subprocess.run("v4l2-ctl --set-ctrl=focus_absolute=0", shell=True)
                # Set other camera parameters for better quality
                subprocess.run("v4l2-ctl --set-ctrl=sharpness=100", shell=True)
                subprocess.run("v4l2-ctl --set-ctrl=brightness=55", shell=True)
                subprocess.run("v4l2-ctl --set-ctrl=contrast=60", shell=True)
                print("Applied v4l2-ctl camera optimizations")
            except Exception as v4l2_err:
                print(f"v4l2-ctl adjustments failed: {v4l2_err}. Continuing with CV2 settings.")
            
            # Alternative focus methods if v4l2-ctl fails
            focus_values = [0, 5, 10, 20, 50, 100, 150, 200, 250]
            best_focus = 0
            best_variance = -1
            
            for val in focus_values:
                # Set focus
                cap.set(cv2.CAP_PROP_FOCUS, val)
                time.sleep(0.5)  # Let camera adjust
                
                # Take a test frame
                ret, frame = cap.read()
                if ret:
                    # Convert to grayscale
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # Calculate Laplacian variance (measure of focus)
                    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
                    print(f"Focus {val} - Variance: {variance}")
                    
                    if variance > best_variance:
                        best_variance = variance
                        best_focus = val
                        
                    # Update latest_jpg for server
                    _, buffer = cv2.imencode('.jpg', frame)
                    latest_jpg = buffer.tobytes()
            
            # Set the best focus value
            print(f"Setting optimal focus value: {best_focus}")
            cap.set(cv2.CAP_PROP_FOCUS, best_focus)
        else:
            # Laptop mode - simpler focus approach
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus
            # Take a test frame for server
            ret, frame = cap.read()
            if ret:
                _, buffer = cv2.imencode('.jpg', frame)
                latest_jpg = buffer.tobytes()
        
    except Exception as e:
        print(f"Focus adjustment error: {e}")
    
    lcd.clear()
    lcd.write_string("Camera ready\nStarting detection")
    
    return cap

# Initialize model
def initialize_model():
    try:
        print("Loading Roboflow model (this may take a moment)...")
        lcd.clear()
        lcd.write_string("Loading model...\nPlease wait")
        model = get_model(model_id="weeds-nxe1w/1")
        print("Roboflow model loaded successfully")
        lcd.clear()
        lcd.write_string("Model loaded\nStarting detection")
        return model
    except Exception as e:
        print(f"Error initializing Roboflow model: {e}")
        lcd.clear()
        lcd.write_string("Model Error!\nCheck connection")
        exit(1)

# Use threading for inference to improve performance
def process_frame(input_frame, model, bounding_box_annotator, label_annotator, fps):
    global latest_jpg, weed_detected
    try:
        # Copy the frame to avoid modification during processing
        frame_copy = input_frame.copy()
        
        # Run inference
        results = model.infer(frame_copy)[0]
        detections = sv.Detections.from_inference(results)
        
        # Update weed detection status
        current_weed_detected = len(detections) > 0
        
        # If weed status changed, update LCD
        if current_weed_detected != weed_detected:
            weed_detected = current_weed_detected
            lcd.clear()
            if weed_detected:
                lcd.write_string("WEED DETECTED!\n" + time.strftime("%H:%M:%S"))
            else:
                lcd.write_string("No weeds found\nMonitoring...")
        
        # Annotate
        if len(detections) > 0:
            frame_copy = bounding_box_annotator.annotate(scene=frame_copy, detections=detections)
            frame_copy = label_annotator.annotate(scene=frame_copy, detections=detections)
        
        # Add platform and FPS info
        cv2.putText(frame_copy, f"Platform: {platform.upper()}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_copy, f"FPS: {fps:.1f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Compress with lower quality for better performance
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        _, buffer = cv2.imencode('.jpg', frame_copy, encode_param)
        latest_jpg = buffer.tobytes()
    except Exception as e:
        print(f"Processing error: {e}")

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    global stop_server
    print("\nCaught Ctrl+C! Shutting down gracefully...")
    
    # Show shutdown message on LCD
    lcd.clear()
    lcd.write_string("Shutting down...\nPlease wait")
    
    # Set flag to stop server
    stop_server = True
    
    # Make sure camera is released if on RPI
    if platform == "rpi":
        ensure_camera_released()
    
    # Clean up LCD
    try:
        lcd.clear()
        lcd.close()
    except:
        pass
    
    print("Cleaned up resources, exiting now.")
    sys.exit(0)

def main(platform_type="laptop"):
    # Setup platform-specific components
    setup_platform(platform_type)
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Release camera if it's already in use (RPI only)
    if platform == "rpi":
        ensure_camera_released()
    
    try:
        # Configure ngrok auth token (RPI only)
        if platform == "rpi":
            try:
                subprocess.run("ngrok config add-authtoken 2TV2268vI6itn4VYkHqYYKgIHF2_7ArR7XzqZ3wBLh6pX8gmv", 
                            shell=True, check=True)
                print("Configured ngrok auth token")
                
                # Start ngrok in a separate thread
                ngrok_thread = threading.Thread(target=start_ngrok)
                ngrok_thread.daemon = True
                ngrok_thread.start()
            except Exception as e:
                print(f"Ngrok setup error (non-critical): {e}")
        
        # Start server thread
        server_thread = threading.Thread(target=run_server)
        server_thread.daemon = True
        server_thread.start()
        
        # Initialize camera with improved focus
        cap = initialize_camera()
        
        # Initialize model
        model = initialize_model()
        
        # Create supervision annotators
        bounding_box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        
        # FPS calculation
        prev_time = time.time()
        frame_count = 0
        fps = 0
        
        # For skipping frames to improve performance
        frame_skip = 0
        max_skip = 2 if platform == "rpi" else 0  # Skip frames only on RPI
        
        # Processing thread
        processing_thread = None
        
        print(f"Starting detection on {platform} - system is ready")
        lcd.clear()
        lcd.write_string("System Ready\nMonitoring...")
        
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            # Update FPS calculation
            frame_count += 1
            current_time = time.time()
            if current_time - prev_time >= 1.0:
                fps = frame_count / (current_time - prev_time)
                frame_count = 0
                prev_time = current_time
                print(f"FPS: {fps:.1f}")
            
            # Skip frames to improve performance on Raspberry Pi
            frame_skip += 1
            if frame_skip <= max_skip:
                continue
            frame_skip = 0
            
            # Only start a new processing thread if previous one is done
            if processing_thread is None or not processing_thread.is_alive():
                processing_thread = threading.Thread(
                    target=process_frame, 
                    args=(frame, model, bounding_box_annotator, label_annotator, fps)
                )
                processing_thread.start()
            
            # Small delay to reduce CPU usage
            time.sleep(0.01)
    
    except Exception as e:
        print(f"Main error: {e}")
        lcd.clear()
        lcd.write_string("System Error!\n" + str(e)[:16])
        time.sleep(5)
        
        # Make sure camera is released on error
        if platform == "rpi":
            ensure_camera_released()
        
        try:
            lcd.close()
        except:
            pass
        
        print("Error handled, resources cleaned up")

# Make sure script exits properly and releases camera on exit
def run_with_protection(platform_type):
    try:
        # Run the main function with the specified platform
        main(platform_type)
    except KeyboardInterrupt:
        # Already handled by signal handler
        pass
    except Exception as e:
        print(f"Unhandled exception: {e}")
    finally:
        # One final check to make sure camera is released
        if platform == "rpi":
            ensure_camera_released()
        print("Final cleanup complete")

# Example usage of the script - user will set platform here
if __name__ == "__main__":
    # Set platform here - choose "laptop" or "rpi"
    selected_platform = "laptop"  # Change this as needed
    
    # Check for command line argument
    if len(sys.argv) > 1:
        if sys.argv[1].lower() in ["laptop", "rpi"]:
            selected_platform = sys.argv[1].lower()
    
    print(f"Starting in {selected_platform} mode")
    run_with_protection(selected_platform)