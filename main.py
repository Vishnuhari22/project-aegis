# Step 1: Import the necessary libraries
import cv2  # This is the OpenCV library for video and image processing.
from ultralytics import YOLO # This imports the YOLO model class from the ultralytics library.

# Step 2: Load the pre-trained YOLOv8 model
# 'yolov8n.pt' is the "nano" version of YOLOv8. It's small and fast, perfect for starting.
# The .pt file will be downloaded automatically the first time this line runs.
print("Loading YOLOv8 model...")
model = YOLO('yolov8n.pt')
print("Model loaded successfully.")

# Step 3: Open the video file
video_path = 'test_video.mp4'
# cv2.VideoCapture creates a video capture object. You can also put 0 here to use your webcam.
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully. If not, exit.
if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    exit()

# Step 4: Loop through the video frames
while True:
    # cap.read() reads one frame from the video.
    # 'success' is a boolean (True/False) that tells us if a frame was read correctly.
    # 'frame' is the actual image data of the frame.
    success, frame = cap.read()

    if success:
        # Step 5: Run YOLOv8 inference on the frame
        # This is the magic line where the AI analyzes the image.
        # It returns a list of result objects.
        results = model(frame)

        # Step 6: Visualize the results on the frame
        # '.plot()' is a handy function from ultralytics that draws all the bounding boxes
        # and labels on the frame for us.
        annotated_frame = results[0].plot()

        # Step 7: Display the annotated frame in a window
        # 'cv2.imshow()' displays an image in a window.
        # The first argument is the window name, the second is the image to display.
        cv2.imshow("Aegis - Real-Time Anomaly Detection", annotated_frame)

        # Step 8: Wait for a key press to exit
        # 'cv2.waitKey(1)' waits for 1 millisecond for a key press.
        # '& 0xFF == ord("q")' checks if the key pressed was 'q'.
        # If it was, we break out of the loop to close the application.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # If 'success' is False, it means we've reached the end of the video.
        print("Reached the end of the video.")
        break

# Step 9: Clean up
# Release the video capture object and close all OpenCV windows.
print("Cleaning up and closing application.")
cap.release()
cv2.destroyAllWindows()