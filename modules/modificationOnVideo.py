import cv2
import os

def process_video(input_path, output_path, mode="slow", factor=2, reverse=False, reverse_position='end'):
    """
    Process video by slowing down or speeding up, with optional reversed playback.

    Args:
        input_path (str): Path to input video.
        output_path (str): Path to save the processed video.
        mode (str): 'slow' to slow down, 'fast' to speed up.
        factor (int): Speed factor (2 = 2x slower or 2x faster).
        reverse (bool): Whether to include reversed video.
        reverse_position (str): 'start', 'end', or 'both' — position of reversed clip.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise Exception("Failed to open video.")

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Load all frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise Exception("No frames found in the video.")

    # Adjust frames for speed
    processed_frames = []
    if mode == "slow":
        for frame in frames:
            processed_frames.extend([frame] * factor)
        out_fps = fps / factor
    elif mode == "fast":
        processed_frames = frames[::factor]
        out_fps = fps * factor
    else:
        raise ValueError("Mode must be either 'slow' or 'fast'.")

    # Reverse logic
    reversed_frames = list(reversed(processed_frames)) if reverse else []

    if reverse and reverse_position == 'start':
        final_frames = reversed_frames + processed_frames
    elif reverse and reverse_position == 'end':
        final_frames = processed_frames + reversed_frames
    elif reverse and reverse_position == 'both':
        final_frames = reversed_frames + processed_frames + reversed_frames
    else:
        final_frames = processed_frames

    # Save output
    out = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))
    for frame in final_frames:
        out.write(frame)
    out.release()

    print(f"✅ Saved processed video to: {output_path}")


input_video_path = "Carparking_lot_Aerial view.mp4"  # Replace with your input video path
output_video_path = "Carparking_lot_Aerial view_with_modification.mp4"  # Replace with your desired output video path

# Example usage
process_video(input_video_path, output_video_path, mode="slow", factor=2, reverse=True, reverse_position='end')