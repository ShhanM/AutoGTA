import win32gui
import time
import numpy as np
import os
from directkeys import PressKey, ReleaseKey, get_key, key_to_index # PressKey/ReleaseKey will be used in control
from PIL import ImageGrab

PIC_COUNT = 0

def get_window_pos():
    classname = "Grand theft auto San Andreas"
    titlename = "GTA: San Andreas"

    # classname = 'CabinetWClass'
    # titlename = 'Tools'
    hwnd = win32gui.FindWindow(classname, titlename)
    position = win32gui.GetWindowRect(hwnd)
    # position:(x, y, w, h)
    # print(position)
    return position
    
def capture(key, position, save=True):
    key = key_to_index(key)
    if not key in [0, 1, 2, 3]:
        return
       
    global PIC_COUNT
    PIC_COUNT += 1
    pos = list(position)

    pos[0] = pos[0] + 100
    pos[1] = pos[1] + 40
    pos[2] = pos[0] + 600
    pos[3] = pos[1] + 600
    # pos[0] += 0.4 * w
    # pos[1] += 0.4 * h
    # pos[2] -= 0.4 * w
    # pos[3] -= 0.4 * h
    
    
    image = ImageGrab.grab(pos)
    if save:
        name =  str(key) + 'k' + str(PIC_COUNT) + '.png'
        image.save(name)
        return name
    else:
        return image

def grab_window(position=get_window_pos(), delay=0.05, save=True):
    print('Press p to start/pause')
    pause = True
    os.chdir('Pics')
    
    while True:
        time.sleep(delay)
        if get_key() == ['P']:
            pause = not(pause)
            print('start to grab' if not pause else 'pause')

        if not pause:
            key = get_key()
            pic = capture(key, position, save)
            if pic:
                print(pic)
if __name__ == '__main__':
    grab_window()