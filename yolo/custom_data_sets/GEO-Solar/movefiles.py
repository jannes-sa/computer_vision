import os
import shutil

# Define the source (folder A) and destination (folder B) directories
src_dir = '/Users/user/work/experiment/computer_vision/yolo/custom_data_sets/GEO-Solar/train/images'
dst_dir = '/Users/user/work/experiment/computer_vision/yolo/custom_data_sets/GEO-Solar/train_2k/images'

# List all files in the source directory
files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]

# Ask the user for the percentage of files to move
percentage = float(50)

# Calculate the number of files to move based on the input percentage
num_files_to_move = int((percentage / 100) * len(files))

# Move the specified percentage of files
for i in range(num_files_to_move):
    src_file_path = os.path.join(src_dir, files[i])
    dst_file_path = os.path.join(dst_dir, files[i])
    shutil.move(src_file_path, dst_file_path)

print(f"Moved {num_files_to_move} files from {src_dir} to {dst_dir}")
