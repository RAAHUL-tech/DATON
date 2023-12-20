import cv2
import numpy as np
import matplotlib.pyplot as plt
from local_utils import detect_lp
from os.path import splitext,basename
from keras.models import model_from_json
import glob


def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)


wpod_net_path = "wpod-net.json"
wpod_net = load_model(wpod_net_path)

def preprocess_image(frame,resize=False):

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img


# forward image through model and return plate's image and coordinates
# if error "No Licensese plate is founded!" pop up, try to adjust Dmin
def get_plate(frame, Dmax=608, Dmin=256):
    vehicle = preprocess_image(frame)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    box_image = draw_box(vehicle, cor)
    # Write the processed frame to the output video
    output_video.write(box_image)
    return LpImg, cor


def draw_box(image_path, cor, thickness=3):
    pts = []
    x_coordinates = cor[0][0]
    y_coordinates = cor[0][1]
    # store the top-left, top-right, bottom-left, bottom-right
    # of the plate license respectively
    for i in range(4):
        pts.append([int(x_coordinates[i]), int(y_coordinates[i])])

    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1, 1, 2))
    vehicle_image = preprocess_image(image_path)

    cv2.polylines(vehicle_image, [pts], True, (0, 255, 0), thickness)
    return vehicle_image



# Open a video file
video_capture = cv2.VideoCapture('Plate_examples/sample_video.mp4')

# Check if the video file opened successfully
if not video_capture.isOpened():
    print("Error: Couldn't open video file.")
    exit()

# Get the video's frames per second (fps) and frame dimensions
fps = int(video_capture.get(cv2.CAP_PROP_FPS))
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a VideoWriter object to save the processed video in MP4 format
output_video = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Process each frame in the video
while True:
    # Read a frame from the video
    ret, frame = video_capture.read()

    # Break the loop if the video is finished
    if not ret:
        break
    # Perform some image processing (grayscale conversion in this case)
    LpImg, cor = get_plate(frame)
    print("Detect %i plate(s) in" % len(LpImg), splitext(basename(frame))[0])
    print("Coordinate of plate(s) in image: \n", cor)


# Release the VideoCapture and VideoWriter objects
video_capture.release()
output_video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()





