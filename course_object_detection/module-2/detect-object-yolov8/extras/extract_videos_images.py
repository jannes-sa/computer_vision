import cv2
from pathlib import Path
import os.path

#  Settings
FPS = 1 # Insert the FPS number you want to extract from the video.
OUTPUT_PATH = "DATASET" # Insert folder path where you want to save the frames.
VIDEO_PATH = r"../demo/highway.mp4"  # Insert the path of your video ex. myvideo.mp4
IMAGE_MAX_SIZE = 1920 # Select max width/height that the video can have.
########################


# Check if video exists
video_path = VIDEO_PATH
if os.path.isfile(video_path) is False:
    raise FileNotFoundError(video_path)

# Get video file name
base = os.path.basename(video_path)
filename = os.path.splitext(base)[0]

output_path = OUTPUT_PATH
output_path = os.path.join(output_path, filename + "\\")
if os.path.isdir(output_path) is False:
    print("Creating folder: {}".format(output_path))
    path = Path(output_path)
    path.mkdir(parents=True)



# python  video_annotation.py -o "dataset" -v "videos/broccoli_video.mp4"

cap = cv2.VideoCapture(video_path)
fps_count = cap.get(cv2.CAP_PROP_FPS)
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
cap_width = cap.get(3)
cap_height = cap.get(4)
print("Video size: {}x{}".format(int(cap_width), int(cap_height)))

# Get filename
filename = Path(video_path).stem

# Settings
save_fps = FPS
maximum_width = IMAGE_MAX_SIZE
maximum_height = IMAGE_MAX_SIZE


# Get Video orientation
orientation = "horizontal"
if cap_width < cap_height:
    orientation = "vertical"

# 4 Cases
resized = False
if cap_width > maximum_width or cap_height > maximum_height:
    if orientation == "vertical":
        new_height = maximum_height
        new_width = new_height / cap_height * cap_width
    else:
        new_width = maximum_width
        new_height = new_width / cap_width * cap_height
    print("Resizing video to {}x{}".format(int(new_width), int(new_height)))
    resized = True


frame_to_skip = int(fps_count / save_fps)

all_frames_to_save = int(total_frames / frame_to_skip)
current_frame = 0 # current frame to grab from the video
frame_count = 0 # count the frames one by one to print them
while True:
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()

    if not ret:
        break

    if resized:
        frame = cv2.resize(frame, (int(new_width), int(new_height)))

    cv2.imshow("Frame", frame)

    cv2.imwrite("{}{}_frame_{}.jpg".format(output_path, filename, current_frame), frame)

    # Update count
    current_frame += frame_to_skip
    frame_count += 1

    print("Saving Frame {} out of {}".format(frame_count, all_frames_to_save + 1))

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

