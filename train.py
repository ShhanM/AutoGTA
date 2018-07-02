from models import LeNet, AlexNet, VGG13, ResNet34, TestNet
from keras.models import load_model
from keras import optimizers
from prepare_data import load_data
from utils import CLASS_NUM, IMG_SIZE
import os

train_X, test_X, train_y, test_y = load_data(class_num=CLASS_NUM, img_size=IMG_SIZE)

if not os.path.exists('autogta.h5'):
    autogta = AlexNet(train_X[0].shape)
    autogta.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    autogta.fit(x=train_X, y=train_y, epochs=10, batch_size=16)
    autogta.save('autogta.h5')
    
else:
    autogta = load_model('autogta.h5')
    
loss, accu = autogta.evaluate(x=test_X, y=test_y)

print('loss\t{}\naccuracy\t{}'.format(loss, accu))