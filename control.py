import time
import cv2
import numpy as np
from keras.models import load_model
from prepare_data import img2array
from grab_pics import capture, get_window_pos
from directkeys import PressKey, ReleaseKey, get_key, key_to_index, W, A, S, D
from utils import IMG_SIZE

autogta = load_model('autogta.h5')

def predict(delay = 0.5):
    pause = True
    while True:
        time.sleep(delay)
        if get_key() == ['P']:
            pause = not(pause)
            print('start to autodrive' if not pause else 'pause')

        if not pause:
            image = capture(get_key(), get_window_pos(), save=False)
            image = image.resize((IMG_SIZE, IMG_SIZE))
            image = np.asarray(image)
            prediction = autogta.predict(image.reshape(1, *image.shape))
            make_decision(prediction)
            print(prediction)

def make_decision(prediction):
    index = int(np.argwhere(prediction[0]==1))
    if index == 1:
        PressKey(W)
        print('直行\n')
        time.sleep(1)
        ReleaseKey(W)
    elif index == 2:
        PressKey(A)
        time.sleep(1)
        ReleaseKey(A)
        print('左拐\n')
    elif index == 3:
        PressKey(D)
        time.sleep(1)
        ReleaseKey(D)
        print('右拐\n')
    elif index == 4:
        PressKey(S)
        time.sleep(1)
        ReleaseKey(S)
        print('减速\n')
  
    get_key() # 吸收PressKey()
    
if __name__ == '__main__':
    predict()
    