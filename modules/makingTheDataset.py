import cv2
import os
from util_New import get_parking_spots_bboxes

# Paths
video_path = 'Carparking_lot_Aerial view.mp4'
mask_path = 'Mask_Carparking_lot_Aerialview.png'
output_dir = 'cropped_spots_more'
os.makedirs(output_dir, exist_ok=True)

# Load mask and open video
mask = cv2.imread(mask_path, 0)
cap = cv2.VideoCapture(video_path)

# Read first frame to get size
ret, first_frame = cap.read()
if not ret:
    cap.release()
    raise Exception("Couldn't read video.")
frame_height, frame_width = first_frame.shape[:2]

# Resize mask to match video frame
mask = cv2.resize(mask, (frame_width, frame_height))
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)
print(f"🅿️ Detected {len(spots)} parking spots")

# Settings
frame_skip = 60  # extract every 30 frames (~1 second for 30fps)
frame_count = 0
extracted = 0

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_skip == 0:
        for idx, (x, y, w, h) in enumerate(spots):
            crop = frame[y:y+h, x:x+w]
            filename = f'spot_{idx}_frame_{frame_count}.png'
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, crop)
            extracted += 1

    frame_count += 1

cap.release()
print(f"\n✅ Saved {extracted} cropped spot images into '{output_dir}'")
