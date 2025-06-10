import cv2
import numpy as np

# Load the first frame from the video
video_path = 'Carparking_lot_Aerial view.mp4'
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    raise Exception("Failed to load video frame.")

mask = np.zeros(frame.shape[:2], dtype=np.uint8)
drawing = False
x_start, y_start = -1, -1
rectangles = []

def draw_rectangle(event, x, y, flags, param):
    global x_start, y_start, drawing, mask

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x_start, y_start = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x_end, y_end = x, y
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        cv2.rectangle(mask, (x_start, y_start), (x_end, y_end), 255, -1)
        rectangles.append(((x_start, y_start), (x_end, y_end)))

cv2.namedWindow('Draw Parking Slots')
cv2.setMouseCallback('Draw Parking Slots', draw_rectangle)

while True:
    cv2.imshow('Draw Parking Slots', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# Save the binary mask
cv2.imwrite('Mask_Carparking_lot_Aerialview.png', mask)
print("Mask saved as 'Mask_Carparking_lot_Aerialview.png'")
