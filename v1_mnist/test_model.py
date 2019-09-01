import cv2
from tensorflow.keras.models import load_model
import numpy as np

class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

cap = cv2.VideoCapture(0)


def adjust_gamma(image, gamma=1.0):

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)

#Copy the relative path for the mnist_model here
model = load_model('Sign_Language_Interpretation/v1_mnist/mnist_model.h5')

while cap.isOpened:

    ret, frame = cap.read()

    cv2.rectangle(frame, (100, 50), (500, 450), (0, 0, 255))
    cv2.imshow('frame', frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = adjust_gamma(frame, gamma=0.7)
    crop = np.array(frame[51:450, 101:500])

    dim = (28, 28)
    img = cv2.resize(crop, dim, interpolation=cv2.INTER_AREA)
    img = img.reshape((1, 28, 28, 1))
    img = img / 255.0
    predicted = model.predict(img)
    print(class_names[np.argmax(predicted)])
    # time.sleep(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()