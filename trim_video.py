import cv2
import os
import cv2

path_video_input = "/media/anlabadmin/setup/dungtd/carton/video_test/2004008_02_IMG_0837.MOV"

cap = cv2.VideoCapture(path_video_input)

fps = cap.get(cv2.CAP_PROP_FPS)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("fps", fps, "num frame = ", num_frames, num_frames / fps)

# exit()
# Get the width and height of the input video
w_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(w_vid, h_vid)
# exit()
# Define the start and end times of the trimmed video
start_time = 37  # in seconds
end_time = 47  # in seconds

# Calculate the start and end frame numbers of the trimmed video
start_frame = int(start_time * fps)
end_frame = int(end_time * fps)

# Create a VideoWriter object to write the trimmed video to a file
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
path_video_output = "/media/anlabadmin/setup/dungtd/carton/video_test/video_2_trim_test2.avi"
out = cv2.VideoWriter(path_video_output, fourcc, fps, (w_vid, h_vid))
folder_save_frame = "/media/anlabadmin/Data/SonG/moving-recog/results_carton/frame_debug/"
# Loop through the frames of the input video and write the trimmed frames to the output video
for i in range(0, num_frames):
    # Read the next frame from the input video
    ret, frame = cap.read()

    # If the frame was successfully read
    if ret:
        if i in range(start_frame, end_frame):
            # Write the frame to the output video
            # print(i,"==check--", frame.shape)
            path_save_img = os.path.join(folder_save_frame, f'{str(i)}.jpg')
            out.write(frame)
            
            cv2.imwrite(path_save_img, frame)
    else:
        # If the frame could not be read, break out of the loop
        break

# Release the VideoCapture and VideoWriter objects
cap.release()
out.release()