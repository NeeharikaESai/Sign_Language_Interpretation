##############
import numpy as np
import cv2
import tensorflow as tf
import time
from keras.models import load_model

# Add the class names
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', 'space']

model = load_model('resnet1.h5')

# illumination correction


def adjust_gamma(image, gamma=1.0):

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


cap = cv2.VideoCapture(0)

while(cap.isOpened):
    # Capture frame-by-frame
    ret, frame = cap.read()

    cv2.rectangle(frame, (100, 50), (500, 450), (0, 0, 255))
    cv2.imshow('frame', frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = adjust_gamma(frame, gamma=0.7)
    crop = np.array(frame[51:450, 101:500, :])

    dim = (200, 200)
    img = cv2.resize(crop, dim, interpolation=cv2.INTER_AREA)
    img = img.reshape((1, 200, 200, 3))
    img = img / 255.0
    predicted = model.predict(img)
    print(class_names[np.argmax(predicted)])
    # time.sleep(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
