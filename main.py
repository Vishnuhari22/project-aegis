# main.py

import cv2
from ultralytics import YOLO
import math # We need the math library to calculate distance

# --- 1. Initialization and Model Loading ---
print("Loading Project Aegis model...")
model = YOLO('yolov8n.pt')
print("Model loaded successfully.")

# --- 2. Video Input ---
video_path = 'test_video.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    exit()

# --- NEW: Object Tracker Initialization ---
# This dictionary will be our short-term memory.
# Key: A unique object ID (e.g., 1, 2, 3...)
# Value: The last known center coordinates (x, y) of that object.
tracker = {}
next_object_id = 1 # Start assigning IDs from 1.

# --- 3. Main Processing Loop ---
while True:
    success, frame = cap.read()

    if success:
        # --- 4. AI Inference ---
        results = model(frame)
        result = results[0]

        # --- NEW: Prepare a list for current frame's detections ---
        # We'll store the center point and class name of each detected object in this list.
        current_detections = []

        # --- 5. Data Extraction ---
        for box in result.boxes:
            confidence = float(box.conf.item())
            if confidence > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls.item())
                class_name = model.names[class_id]

                # We only want to track objects relevant to our anomaly
                if class_name in ['person', 'backpack', 'handbag', 'suitcase', 'bag']:
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    current_detections.append(((center_x, center_y), class_name))
                    # Draw the bounding box for visualization
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # --- NEW: Object Tracking Logic ---
        # This dictionary will hold the tracked objects for the current frame.
        updated_tracker = {}
        
        for det_center, det_class_name in current_detections:
            object_found = False
            for obj_id, tracked_center in tracker.items():
                # Calculate the Euclidean distance between a new detection and an existing object
                distance = math.sqrt((det_center[0] - tracked_center[0])**2 + (det_center[1] - tracked_center[1])**2)
                
                # If the distance is below a threshold, it's the same object.
                if distance < 35: # This threshold may need tuning based on video resolution and movement speed.
                    updated_tracker[obj_id] = det_center
                    object_found = True
                    # Display the object's ID in green for a successfully tracked object
                    cv2.putText(frame, f"{det_class_name} ID: {obj_id}", (det_center[0], det_center[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    break
            
            # If after checking all tracked objects, none were close enough, it's a new object.
            if not object_found:
                updated_tracker[next_object_id] = det_center
                # Display the new object's ID in red to signify it's new
                cv2.putText(frame, f"{det_class_name} ID: {next_object_id}", (det_center[0], det_center[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                next_object_id += 1
        
        # Update our main tracker with the latest information for the next frame.
        tracker = updated_tracker

        # --- 6. Visualization ---
        cv2.imshow("Project Aegis - Object Tracking", frame)

        # --- 7. Exit Condition ---
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# --- 8. Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("Application shut down gracefully.")
