# main.py

import cv2
from ultralytics import YOLO

# --- 1. Initialization and Model Loading ---
# Load the pre-trained YOLOv8 'nano' model. 
# The 'ultralytics' library handles downloading the model weights automatically.
print("Loading Project Aegis model...")
model = YOLO('yolov8n.pt')
print("Model loaded successfully.")

# --- 2. Video Input ---
# Define the path to your test video.
video_path = 'test_video.mp4'
# Create a VideoCapture object to read frames from the video file.
cap = cv2.VideoCapture(video_path)

# Check if the video file was opened successfully.
if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    exit()

# --- 3. Main Processing Loop ---
# This loop reads the video frame by frame until the end.
while True:
    # Read a single frame from the video. 'success' is a boolean, 'frame' is the image.
    success, frame = cap.read()

    # If a frame was read successfully, process it.
    if success:
        # --- 4. AI Inference ---
        # Pass the frame to the loaded YOLO model for object detection.
        results = model(frame)
        # The 'results' object contains all the information about detected objects.
        result = results[0] # Get the results for the first image (our frame).

        # --- 5. Data Extraction ---
        # Iterate through each detected bounding box in the current frame.
        for box in result.boxes:
            # Extract the class ID (e.g., 0 for 'person') and convert to an integer.
            class_id = int(box.cls.item())
            # Get the human-readable class name (e.g., 'person') using the model's names dictionary.
            class_name = model.names[class_id]
            # Get the confidence score (how sure the model is) and convert to a float.
            confidence = float(box.conf.item())

            # Only process detections with a confidence score higher than 0.5 (50%).
            if confidence > 0.5:
                # Print the extracted data to the terminal. This is our structured output.
                print(f"[INFO] Detected: {class_name} | Confidence: {confidence:.2f}")

        # --- 6. Visualization ---
        # Use the '.plot()' method to draw the bounding boxes and labels on the frame.
        annotated_frame = result.plot()
        # Display the frame with the detections in a window with a standard title.
        cv2.imshow("Project Aegis - Live Feed", annotated_frame)

        # --- 7. Exit Condition ---
        # Wait for 1 millisecond. If the 'q' key is pressed during that time, exit the loop.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # If 'success' is false, it means we've reached the end of the video.
        break

# --- 8. Cleanup ---
# Release the video capture object to free up resources.
cap.release()
# Close all the windows created by OpenCV.
cv2.destroyAllWindows()
print("Application shut down gracefully.")