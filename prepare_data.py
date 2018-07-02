import os
import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from sklearn.cross_validation import train_test_split
from utils import CLASS_NUM, IMG_SIZE

def img2array(file_name, img_size):
    image = Image.open(file_name)
    image = image.resize((img_size, img_size))
    image = np.asarray(image)
    return image
    
def load_data(class_num, img_size=IMG_SIZE):
    '''
    读取路径下所有图片及其标签，转化为 np.array
    参数：
        class_num:种类数
        img_size:网络入口尺寸
    返回:
        train_X, test_X, train_y, test_y
    '''
    X = []
    y = []
    for path, dir, files in os.walk(os.getcwd()):
            for f in files:
                file_name = os.path.join(path, f)
                if file_name.endswith('png'):
                    image = img2array(file_name, img_size)
                    X.append(image)
                    
                    label = int(file_name.split('\\')[-1].split('k')[0]) - 1
                    if label == 4: # A
                        label = 1
                    elif label == 5: # D
                        label = 2
                    y.append(label)

    X = np.array(X, dtype='float') / 255.0
    y = np.array(y)
    y = to_categorical(y, class_num) # one-hot encoding
    
    np.save('X.npy', X)
    np.save('y.npy', y)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

    return train_X, test_X, train_y, test_y
    
if __name__ == '__main__':
    load_data(CLASS_NUM, IMG_SIZE)
    