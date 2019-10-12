import sys
sys.path.insert(0, 'v3/handtracking')
from utils import detector_utils as detector_utils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import tensorflow as tf
from keras.models import load_model

class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', 'space']

def make_1080():
    cap.set(3, 1920)
    cap.set(4, 1080)

def draw_label(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)

model = load_model('v2/resnet1.h5')

detection_graph, sess = detector_utils.load_inference_graph()

cap = cv2.VideoCapture(0)
make_1080()

while(cap.isOpened):
    
    ret, frame = cap.read()

    im_width, im_height = (cap.get(3), cap.get(4))
    num_hands_detect = 1
    score_thresh = 0.5

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes, scores = detector_utils.detect_objects(frame, detection_graph, sess)

    crop = detector_utils.get_crop_image(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, frame)

    detector_utils.draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, frame)

    dim = (200, 200)
    crop = cv2.resize(crop, dim, interpolation=cv2.INTER_AREA)
    crop = crop.reshape((1, 200, 200, 3))
    crop = crop / 255

    predicted = model.predict(crop)

    draw_label(frame, class_names[np.argmax(predicted)], (100,100), (0,255,0))
    
    cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("frame",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
    