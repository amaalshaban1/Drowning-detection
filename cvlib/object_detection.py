#import necessary packages
import cv2
import os
import numpy as np
from .utils import download_file
from playsound import playsound
import threading
import pygame

initialize = True
net = None
dest_dir = os.path.expanduser('~') + os.path.sep + '.cvlib' + os.path.sep + 'object_detection' + os.path.sep + 'yolo' + os.path.sep + 'yolov3'
classes = None
COLORS = [(0, 255, 0), (0, 0, 255), (255, 255, 255)]  # Add a third color (white) to the list

def populate_class_labels():
    class_file_name = 'yolov3_classes.txt'
    class_file_abs_path = dest_dir + os.path.sep + class_file_name
    url = 'https://raw.githubusercontent.com/randhana/Drowning-Detection-/master/yolov3.txt'
    if not os.path.exists(class_file_abs_path):
        download_file(url=url, file_name=class_file_name, dest_dir=dest_dir)
    f = open(class_file_abs_path, 'r')
    classes = [line.strip() for line in f.readlines()]
    return classes

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_bbox(img, bbox, labels, confidence, Drowning, write_conf=False):
    global COLORS
    global classes

    if classes is None:
        classes = populate_class_labels()

    for i, label in enumerate(labels):
        # If the person is drowning, the box will be drawn red instead of green
        if label == 'person' and Drowning:
            color = COLORS[1]
            label = 'ALERT DROWNING'
            threading.Thread(target=play_alert_sound).start()
        else:
            color = COLORS[0]
            label = 'Normal'

        if write_conf:
            label += ' ' + str(format(confidence[i] * 100, '.2f')) + '%'

        # Only need two points (the opposite corners) to draw a rectangle. These points
        # are stored in the variable bbox
        cv2.rectangle(img, (bbox[i][0],bbox[i][1]), (bbox[i][2],bbox[i][3]), color, 2)
        cv2.putText(img, label, (bbox[i][0],bbox[i][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img

def detect_common_objects(image, confidence=0.5, nms_thresh=0.3):
    try:
        if image is None:
            raise ValueError("Received invalid image")
        Height, Width = image.shape[:2]
        scale = 0.00392
        global classes, dest_dir, net

        # Check if YOLO files are already loaded
        if net is None:
            # Initialize YOLO model
            config_file_name = 'yolov3.cfg'
            config_file_abs_path = os.path.join(dest_dir, config_file_name)
            weights_file_name = 'yolov3.weights'
            weights_file_abs_path = os.path.join(dest_dir, weights_file_name)

            if not os.path.exists(config_file_abs_path):
                download_file(url='https://raw.githubusercontent.com/randhana/Drowning-Detection-/master/yolov3.cfg', file_name=config_file_name, dest_dir=dest_dir)

            if not os.path.exists(weights_file_abs_path):
                download_file(url='https://pjreddie.com/media/files/yolov3.weights', file_name=weights_file_name, dest_dir=dest_dir)

            classes = populate_class_labels()
            net = cv2.dnn.readNet(weights_file_abs_path, config_file_abs_path)

        # Process the image for detection
        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))

        class_ids, confidences, boxes = [], [], []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                max_conf = scores[class_id]
                if max_conf > confidence:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(max_conf))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence, nms_thresh)
        bbox, label, conf = [], [], []

        if len(indices) > 0:
            indices = indices.flatten()
            for i in indices:
                box = boxes[i]
                x, y, w, h = box
                bbox.append([round(x), round(y), round(x + w), round(y + h)])
                label.append(classes[class_ids[i]])
                conf.append(confidences[i])

        return bbox, label, conf
    except Exception as e:
        print(f"Error in object detection: {e}")
        return [], [], []

def play_sound(file_path):
    try:
        pygame.mixer.init()
        sound = pygame.mixer.Sound(file_path)
        sound.play()
    except Exception as e:
        print(f"Error playing sound: {e}")

def play_alert_sound():
    file_path = r"D:\coding\python\cv\Drowning-Detection--master\Drowning-Detection--master\cvlib\alarm.mp3"  # تأكد من أن المسار صحيح
    print(f"Playing sound from: {file_path}")
    thread = threading.Thread(target=play_sound, args=(file_path,))
    thread.start()

# Example usage: 
# This would be called when a person is detected to be drowning
play_alert_sound()
