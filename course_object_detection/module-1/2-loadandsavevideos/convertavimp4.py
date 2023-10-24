from moviepy.editor import *
# must install ffmpeg in server first
# for mac just brew instal ffmpeg

# Load the AVI file
clip = VideoFileClip("transport_flipped2.avi")

# Write it to MP4 format
clip.write_videofile("video_example.mp4", codec="libx264")
