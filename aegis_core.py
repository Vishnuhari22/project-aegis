# aegis_core.py

import cv2
from ultralytics import YOLO
import math
import time
import asyncio

# --- Helper Function to Group Classes ---
def get_class_group(class_name):
    if class_name in ['backpack', 'handbag', 'suitcase', 'bag']:
        return 'bag'
    if class_name == 'person':
        return 'person'
    return None

async def process_video(websocket):
    """
    Main function to process the video, perform tracking, and send alerts.
    """
    print("Loading Project Aegis model...")
    model = YOLO('yolov8n.pt')
    print("Model loaded successfully.")

    video_path = 'test_video1.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        await websocket.send_json({"type": "error", "message": "Could not open video file."})
        return

    tracker = {}
    next_object_id = 1
    UNATTENDED_THRESHOLD_SECONDS = 2
    GHOST_CLEANUP_SECONDS = 3
    OWNER_GRACE_PERIOD_SECONDS = 1

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        result = results[0]
        
        current_detections = []
        for box in result.boxes:
            confidence = float(box.conf.item())
            if confidence > 0.5:
                class_name = model.names[int(box.cls.item())]
                class_group = get_class_group(class_name)
                
                if class_group:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    current_detections.append({
                        'center': (center_x, center_y), 'class_group': class_group, 
                        'display_name': class_name, 'bbox': (x1, y1, x2, y2)
                    })

        # --- Tracking Logic ---
        for obj_id in tracker:
            tracker[obj_id]['matched_in_frame'] = False

        unmatched_detections = []
        for det in current_detections:
            match_found = False
            for obj_id, obj_data in tracker.items():
                if det['class_group'] == obj_data['class_group']:
                    distance = math.sqrt((det['center'][0] - obj_data['center'][0])**2 + (det['center'][1] - obj_data['center'][1])**2)
                    if distance < 75:
                        obj_data.update({
                            'center': det['center'], 'bbox': det['bbox'], 'last_seen': time.time(),
                            'matched_in_frame': True, 'display_name': det['display_name']
                        })
                        match_found = True
                        break
            if match_found: continue
        
        # Correctly identify unmatched detections to be added as new objects
        matched_centers = {tuple(obj['center']) for obj in tracker.values() if obj['matched_in_frame']}
        for det in current_detections:
            if tuple(det['center']) not in matched_centers:
                 tracker[next_object_id] = {
                    'center': det['center'], 'bbox': det['bbox'], 'class_group': det['class_group'],
                    'display_name': det['display_name'], 'status': 'new', 'timer_start': 0, 
                    'owner': None, 'last_seen': time.time(), 'matched_in_frame': True
                }
                 next_object_id += 1

        current_time = time.time()
        ids_to_delete = [obj_id for obj_id, data in tracker.items() if (current_time - data['last_seen']) > GHOST_CLEANUP_SECONDS]
        for obj_id in ids_to_delete:
            if obj_id in tracker: del tracker[obj_id]

        people_visible = {obj_id: data for obj_id, data in tracker.items() if data['class_group'] == 'person' and data['matched_in_frame']}
        all_tracked_bags = {obj_id: data for obj_id, data in tracker.items() if data['class_group'] == 'bag'}

        for bag_id, bag_data in all_tracked_bags.items():
            if bag_data['owner'] is None or bag_data['owner'] not in tracker:
                min_dist = float('inf')
                potential_owner = None
                for person_id, person_data in people_visible.items():
                    distance = math.sqrt((bag_data['center'][0] - person_data['center'][0])**2 + (bag_data['center'][1] - person_data['center'][1])**2)
                    if distance < 250:
                        if distance < min_dist:
                            min_dist = distance
                            potential_owner = person_id
                if potential_owner:
                    bag_data['owner'] = potential_owner
                    bag_data['status'] = 'owned'

        for bag_id, bag_data in all_tracked_bags.items():
            owner_id = bag_data['owner']
            if owner_id:
                if owner_id not in tracker or (time.time() - tracker[owner_id]['last_seen']) > OWNER_GRACE_PERIOD_SECONDS:
                    if bag_data['status'] != 'unattended':
                        bag_data['status'] = 'unattended'
                        bag_data['timer_start'] = time.time()
                else:
                    bag_data['status'] = 'owned'
                    bag_data['timer_start'] = 0

        for obj_id, data in tracker.items():
            if data['matched_in_frame']:
                x1, y1, x2, y2 = data['bbox']
                center_x, center_y = data['center']
                text = f"ID:{obj_id} {data['display_name']}"
                color = (0, 255, 0)

                if data['status'] == 'unattended':
                    time_unattended = time.time() - data['timer_start']
                    text += f" Unattended: {int(time_unattended)}s"
                    color = (0, 165, 255)
                    if time_unattended > UNATTENDED_THRESHOLD_SECONDS:
                        color = (0, 0, 255)
                        text += " - ANOMALY!"
                        alert_message = f"Unattended package detected! Object ID: {obj_id}"
                        print(f"ALERT: {alert_message}")
                        await websocket.send_json({"type": "alert", "message": alert_message})
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (center_x - 50, center_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        cv2.imshow("Project Aegis - Live Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        await asyncio.sleep(0.01) # Allow other tasks to run

    cap.release()
    cv2.destroyAllWindows()
    print("Video processing finished.")
    await websocket.send_json({"type": "status", "message": "Video processing finished."})
