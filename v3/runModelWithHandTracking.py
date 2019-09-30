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

model = load_model('v2/resnet1.h5')

detection_graph, sess = detector_utils.load_inference_graph()

def adjust_gamma(image, gamma=1.0):

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)

cap = cv2.VideoCapture(0)

while(cap.isOpened):
    
    ret, frame = cap.read()

    im_width, im_height = (cap.get(3), cap.get(4))
    num_hands_detect = 1
    score_thresh = 0.1

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes, scores = detector_utils.detect_objects(frame, detection_graph, sess)

    crop = detector_utils.get_crop_image(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, frame)

    detector_utils.draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, frame)

    dim = (200, 200)
    crop = cv2.resize(crop, dim, interpolation=cv2.INTER_AREA)
    crop = crop.reshape((1, 200, 200, 3))
    crop = crop / 255

    predicted = model.predict(crop)
    print(class_names[np.argmax(predicted)])

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
    