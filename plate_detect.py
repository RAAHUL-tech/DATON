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

def preprocess_image(image_path,resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img



# Create a list of image paths
image_paths = glob.glob("Plate_examples/*.jpg")
print(image_paths)
# forward image through model and return plate's image and coordinates
# if error "No Licensese plate is founded!" pop up, try to adjust Dmin

def draw_box(image_path, cor, thickness=3):
    pts = []
    x_coordinates = cor[0][0]
    y_coordinates = cor[0][1]
    print("X:", x_coordinates)
    print("Y:", y_coordinates)
    # store the top-left, top-right, bottom-left, bottom-right
    # of the plate license respectively
    for i in range(4):
        pts.append([int(x_coordinates[i]), int(y_coordinates[i])])

    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1, 1, 2))
    vehicle_image = preprocess_image(image_path)
    print("Pts:", pts)
    cv2.polylines(vehicle_image, [pts], True, (0, 255, 0), thickness)
    return vehicle_image

def get_plate(image_path, Dmax=608, Dmin=256):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return LpImg, cor

# Obtain plate image and its coordinates from an image
print(image_paths[1])
test_image = image_paths[2]
LpImg,cor = get_plate(test_image)
if not LpImg or not cor:
    print("No detections...")
else:
    print("Detect %i plate(s) in"%len(LpImg),splitext(basename(test_image))[0])
    print("Coordinate of plate(s) in image: \n", cor)
    # Visualize our result
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.axis(False)
    plt.imshow(preprocess_image(test_image))
    plt.subplot(1, 2, 2)
    plt.axis(False)
    plt.imshow(LpImg[0])
    plt.show()
    # plt.savefig("part1_result.jpg",dpi=300)

    plt.figure(figsize=(8, 8))
    plt.axis(False)
    plt.imshow(draw_box(test_image, cor))
    plt.show()






