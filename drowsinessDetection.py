import cv2
import os

import imutils
from keras.models import load_model
import numpy as np
from pygame import mixer
import pyfirmata as fir
import time

a = fir.Arduino('COM3')
a.digital[9].mode = fir.OUTPUT
a.digital[2].mode = fir.OUTPUT

mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

lbl = ['Close', 'Open']

model = load_model('models/cnncat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]
m = 0;

while (True):
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        count = count + 1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye / 255
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)
        rpred = model.predict_classes(r_eye)
        if (rpred[0] == 1):
            lbl = 'Open'
        if (rpred[0] == 0):
            lbl = 'Closed'
        break

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        count = count + 1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)
        lpred = model.predict_classes(l_eye)
        if (lpred[0] == 1):
            lbl = 'Open'
        if (lpred[0] == 0):
            lbl = 'Closed'
        break

    if (rpred[0] == 0 and lpred[0] == 0):
        if m == 0:
            now = time.time()
            future = now + 1.5

        m = 1;
        t = time.time()

        diffTxt = '{:.2f}s'.format(float(1.5 - (future - time.time())))

        cv2.putText(frame, diffTxt, (10, height - 40), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        if (time.time() >= future):
            cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
            cv2.putText(frame, diffTxt, (10, height - 40), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

            try:
                sound.play()
                a.digital[2].write(0)
                a.digital[9].write(1)
                a.digital[3].write(1)
            except:
                pass



        else:
            pass
            a.digital[2].write(1)
            a.digital[9].write(0)
            a.digital[3].write(0)

        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
        cv2.putText(frame, "Driver sleeps", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        # sound.stop()
        m = 0;
        cv2.putText(frame, "Awake driver", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    frame = imutils.resize(frame, width=1200)
    cv2.imshow('Sleep Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
