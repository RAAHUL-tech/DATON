from ultralytics import YOLO
import cv2
import numpy as np
import util
import local_utils
from os.path import splitext, basename
from keras.models import model_from_json
from sort.sort import *
from util import get_car, read_license_plate, write_csv
from local_utils import detect_lp
from plate_detection import load_model, preprocess_image, draw_box, get_plate


results = {}

motor_tracker = Sort()


# load models
coco_model = YOLO('yolov8n.pt')    # for detecting vehicles like car,bike
wpod_net_path = "wpod-net.json"     # for detecting license plate
wpod_net = load_model(wpod_net_path)

# load video
cap = cv2.VideoCapture('Plate_examples/sample_video1.mp4')

vehicles = [2, 3, 5, 7]    # 2-car, 3-motorbike, 5-bus, 7-truck  in coco model dataset

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        print("Frame no is:", frame_nmr)
        if frame_nmr > 10:
            break
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        vehicle_detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                vehicle_detections_.append([x1, y1, x2, y2, score])
                print("Vehicle detected:", vehicle_detections_)

        # track vehicles
        track_ids = motor_tracker.update(np.asarray(vehicle_detections_))
        print("Tracking id:", track_ids)

        # detect license plates
        LpImg, cor = get_plate(frame, wpod_net)
        if not cor or not LpImg:
            print("No detections...")
            continue
        else:
            print("Number of license plate detected is:", len(LpImg))
            print("license plate coordinate:", cor)

            for license_plate in cor:
                pts = draw_box(frame, cor)
                print(pts)
                x1, y1, x2, y2 = pts
                print("License plate detected:", x1, y1, x2, y2)
                # assign license plate to car
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(pts, track_ids)
                print("get_car:", xcar1, ycar1, car_id)
                if car_id != -1:
                    # crop license plate

                    license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                    # process license plate

                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255,
                                                                 cv2.THRESH_BINARY_INV)  # normalizing

                    plt.figure(figsize=(12, 5))
                    plt.subplot(1, 2, 1)
                    plt.axis(False)
                    plt.imshow(license_plate_crop_gray)
                    plt.subplot(1, 2, 2)
                    plt.axis(False)
                    plt.imshow(license_plate_crop_thresh)
                    plt.show()

                    # read license plate number
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                    print("license no is:", license_plate_text)

                    if license_plate_text is not None:

                        results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                      'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                        'text': license_plate_text,
                                                                        'text_score': license_plate_text_score}}

# write results
write_csv(results, 'results/test1.csv')