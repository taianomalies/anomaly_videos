import cv2
import os

directory = './data/track2-dataset'

tif_files = [f for f in os.listdir(directory) if f.endswith('.tif')]
tif_files.sort()  # Sort the files if necessary

output_directory = './output_videos'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

output_file_prefix = 'output_video'
output_file_extension = '.avi'

fps = 25.0  # Adjust the frame rate as needed
#frame_width, frame_height = None, None  # Set frame size if required

first_image = cv2.imread(os.path.join(directory, tif_files[0]))
frame_height, frame_width = first_image.shape[:2]

video_index = 0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = f"{output_directory}/{output_file_prefix}_{video_index}{output_file_extension}"
video = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
for file in tif_files:
    #output_file = f"{output_file_prefix}_{video_index}{output_file_extension}"
    #video = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
    image = cv2.imread(os.path.join(directory, file))
    video.write(image)
    #video.release()
    video_index += 1


video.release()
cv2.destroyAllWindows()








