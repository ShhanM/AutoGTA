import numpy as np
from keras import layers
from keras.models import Model
from keras.layers import MaxPooling2D, AveragePooling2D, ZeroPadding2D, Conv2D, Flatten, add
from keras.layers import Input, Dense, Activation, Dropout, BatchNormalization
from keras.utils.np_utils import to_categorical
from utils import CLASS_NUM

def TestNet(input_shape):
    X_input = Input(input_shape)
    X = Conv2D(32, (3, 3), strides=(1, 1), name='conv0')(X_input)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool')(X)
    X = Flatten()(X)
    X = Dense(CLASS_NUM, activation='softmax', name='fc')(X)
    model = Model(inputs=X_input, outputs=X, name='test')
    
    return model
    
def LeNet(input_shape):
	'''
	input size:(32, 32, 3)
	'''
	X_input = Input(input_shape)
	X = Conv2D(32,(5,5),strides=(1,1),input_shape=(28,28,1),padding='valid',activation='relu',kernel_initializer='uniform')(X_input)
	X = MaxPooling2D(pool_size=(2,2))(X)
	X = Conv2D(64,(5,5),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform')(X)
	X = MaxPooling2D(pool_size=(2,2))(X)
	X = Flatten()(X)
	X = Dense(100,activation='relu')(X)
	X = Dense(CLASS_NUM,activation='softmax')(X)
	
	model = Model(inputs=X_input, outputs=X, name='AlexNet')
	return model
	
def AlexNet(input_shape):
	'''
	input size:(224, 224, 3)
	'''
	X_input = Input(input_shape)
	X = Conv2D(96,(11,11),strides=(4,4),input_shape=(227,227,3),padding='valid',activation='relu',kernel_initializer='uniform')(X_input)
	X = MaxPooling2D(pool_size=(3,3),strides=(2,2))(X)
	X = Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform')(X)
	X = MaxPooling2D(pool_size=(3,3),strides=(2,2))(X)
	X = Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform')(X)
	X = Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform')(X)
	X = Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform')(X)
	X = MaxPooling2D(pool_size=(3,3),strides=(2,2))(X)
	X = Flatten()(X)
	X = Dense(4096,activation='relu')(X)
	X = Dropout(0.5)(X)
	X = Dense(4096,activation='relu')(X)
	X = Dropout(0.5)(X)
	X = Dense(CLASS_NUM,activation='softmax')(X) #!!!输出类别改动
	
	model = Model(inputs=X_input, outputs=X, name='AlexNet')
	return model

def VGG13(input_shape):
	'''
	input size:(224, 224, 3)
	'''
	X_input = Input(input_shape)
	X = Conv2D(64,(3,3),strides=(1,1),input_shape=(224,224,3),padding='same',activation='relu',kernel_initializer='uniform')(X_input)
	X = Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform')(X)
	X = MaxPooling2D(pool_size=(2,2))(X)
	X = Conv2D(128,(3,2),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform')(X)
	X = Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform')(X)
	X = MaxPooling2D(pool_size=(2,2))(X)
	X = Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform')(X)
	X = Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform')(X)
	X = MaxPooling2D(pool_size=(2,2))(X)
	X = Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform')(X)
	X = Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform')(X)
	X = MaxPooling2D(pool_size=(2,2))(X)
	X = Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform')(X)
	X = Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform')(X)
	X = MaxPooling2D(pool_size=(2,2))(X)
	X = Flatten()(X)
	X = Dense(4096,activation='relu')(X)
	X = Dropout(0.5)(X)
	X = Dense(4096,activation='relu')(X)
	X = Dropout(0.5)(X)
	X = Dense(CLASS_NUM,activation='softmax')(X)
	
	model = Model(inputs=X_input, outputs=X, name='AlexNet')
	return model
	
def ResNet34(input_shape):
	'''
	input size:(224, 224, 3)
	'''
	def Conv2d_BN(x, nb_filter,kernel_size, strides=(1,1), padding='same',name=None):
		if name is not None:
			bn_name = name + '_bn'
			conv_name = name + '_conv'
		else:
			bn_name = None
			conv_name = None
	  
		x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)
		x = BatchNormalization(axis=3,name=bn_name)(x)
		return x
		
	def Conv_Block(inpt,nb_filter,kernel_size,strides=(1,1), with_conv_shortcut=False):
		x = Conv2d_BN(inpt,nb_filter=nb_filter,kernel_size=kernel_size,strides=strides,padding='same')
		x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size,padding='same')
		if with_conv_shortcut:
			shortcut = Conv2d_BN(inpt,nb_filter=nb_filter,strides=strides,kernel_size=kernel_size)
			x = add([x,shortcut])
			return x
		else:
			x = add([x,inpt])
			return x

	X_input = Input(input_shape)  
	X = ZeroPadding2D((3,3))(X_input) 
	X = Conv2d_BN(X,nb_filter=64,kernel_size=(7,7),strides=(2,2),padding='valid')  
	X = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(X)  
	#(56,56,64)  
	X = Conv_Block(X,nb_filter=64,kernel_size=(3,3))  
	X = Conv_Block(X,nb_filter=64,kernel_size=(3,3))  
	X = Conv_Block(X,nb_filter=64,kernel_size=(3,3))  
	#(28,28,128)  
	X = Conv_Block(X,nb_filter=128,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)  
	X = Conv_Block(X,nb_filter=128,kernel_size=(3,3))  
	X = Conv_Block(X,nb_filter=128,kernel_size=(3,3))  
	X = Conv_Block(X,nb_filter=128,kernel_size=(3,3))  
	#(14,14,256)  
	X = Conv_Block(X,nb_filter=256,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)  
	X = Conv_Block(X,nb_filter=256,kernel_size=(3,3))  
	X = Conv_Block(X,nb_filter=256,kernel_size=(3,3))  
	X = Conv_Block(X,nb_filter=256,kernel_size=(3,3))  
	X = Conv_Block(X,nb_filter=256,kernel_size=(3,3))  
	X = Conv_Block(X,nb_filter=256,kernel_size=(3,3))  
	#(7,7,512)  
	X = Conv_Block(X,nb_filter=512,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)  
	X = Conv_Block(X,nb_filter=512,kernel_size=(3,3))  
	X = Conv_Block(X,nb_filter=512,kernel_size=(3,3))  
	X = AveragePooling2D(pool_size=(7,7))(X)  
	X = Flatten()(X)
	X = Dense(CLASS_NUM,activation='softmax')(X)  
	
	model = Model(inputs=X_input,outputs=X)  
	return model