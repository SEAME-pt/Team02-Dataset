# Rename all files in 'frames_signals' from frame_02* to frame_03*
import os

folder = "frames_road"

for filename in os.listdir(folder):
    if filename.startswith("frame_00"):
        new_name = "frame_04" + filename[len("frame_0"):]
        os.rename(os.path.join(folder, filename), os.path.join(folder, new_name))
        print(f"Renamed {filename} -> {new_name}")