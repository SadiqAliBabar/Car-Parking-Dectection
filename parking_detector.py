import cv2
import matplotlib.pyplot as plt
import numpy as np

from modules.util_New import get_parking_spots_bboxes, empty_or_not, extract_features

def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

mask = r'C:\Users\SadiqAliBabar\Downloads\ParkingDetectorComplete\ParkingDetectorComplete\utils\Mask_Carparking_lot_Aerialview.png'
video_path = 'Carparking_lot_Aerial view.mp4'

# Load the mask
mask = cv2.imread(mask, 0)
if mask is None:
    raise ValueError("Could not read mask file. Please check the path.")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError("Could not open video file. Please check the path.")

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

spots = get_parking_spots_bboxes(connected_components)
print(f"Detected {len(spots)} parking spots")

# Initialize all spots as None (unknown status)
spots_status = [None for _ in spots]
diffs = [None for _ in spots]

previous_frame = None
frame_nmr = 0
step = 1  # Evaluate every 15 frames
first_full_eval = False

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    if frame_nmr % 100 == 0:
        print(f"Processing frame {frame_nmr}")

    if not first_full_eval:
        print("Doing initial full evaluation of all spots")
        for spot_indx, spot in enumerate(spots):
            x1, y1, w, h = spot
            y2 = min(y1 + h, frame.shape[0])
            x2 = min(x1 + w, frame.shape[1])
            if y2 <= y1 or x2 <= x1:
                continue
            spot_crop = frame[y1:y2, x1:x2, :]
            spot_status = empty_or_not(spot_crop)
            spots_status[spot_indx] = spot_status
        first_full_eval = True

    if frame_nmr % step == 0 and previous_frame is not None:
        for spot_indx, spot in enumerate(spots):
            x1, y1, w, h = spot
            y2 = min(y1 + h, frame.shape[0])
            x2 = min(x1 + w, frame.shape[1])
            if y2 <= y1 or x2 <= x1:
                continue
            spot_crop = frame[y1:y2, x1:x2, :]
            prev_crop = previous_frame[y1:y2, x1:x2, :]
            diffs[spot_indx] = calc_diff(spot_crop, prev_crop)

    if frame_nmr % step == 0:
        if previous_frame is None:
            arr_ = range(len(spots))
        else:
            valid_diffs = [d for d in diffs if d is not None]
            if valid_diffs and np.amax(valid_diffs) > 0:
                arr_ = [j for j in np.argsort(diffs) if diffs[j] is not None and diffs[j] / np.amax(valid_diffs) > 0.4]
            else:
                arr_ = []

        for spot_indx in arr_:
            x1, y1, w, h = spots[spot_indx]
            y2 = min(y1 + h, frame.shape[0])
            x2 = min(x1 + w, frame.shape[1])
            if y2 <= y1 or x2 <= x1:
                continue
            spot_crop = frame[y1:y2, x1:x2, :]
            spot_status = empty_or_not(spot_crop)
            spots_status[spot_indx] = spot_status

    if frame_nmr % step == 0:
        previous_frame = frame.copy()

    available_spots = 0  # RESET before counting green rectangles
    for spot_indx, spot in enumerate(spots):
        spot_status = spots_status[spot_indx]
        x1, y1, w, h = spots[spot_indx]
        if spot_status is None:
            continue
        color = (0, 255, 0) if spot_status else (0, 0, 255)
        if spot_status:
            available_spots += 1  # count dynamically
        frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)

    total_spots = sum(1 for status in spots_status if status is not None)

    # Draw background rectangle and display available count
    cv2.rectangle(frame, (50, 20), (450, 70), (0, 0, 0), -1)
    cv2.putText(frame, f'Available spots: {available_spots} / {total_spots}',
            (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


    # Optionally print to console for debugging
    print(f"Available: {available_spots}, Total: {total_spots}")

    # Resize for display if needed
    display_frame = cv2.resize(frame, (1280, 720))
    cv2.imshow('frame', display_frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_nmr += 1

cap.release()
cv2.destroyAllWindows()
