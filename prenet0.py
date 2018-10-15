# -*- coding: utf-8 -*-
"""
Created on Wed May 10 12:32:25 2017

@author: wqx
"""
import os
from keras.utils import multi_gpu_model
from PIL import Image
import threading
import pandas as pd
from keras import applications
from keras.utils import plot_model
import keras
from keras import backend
backend.set_image_dim_ordering('tf')
from keras import regularizers
import numpy as np
from keras.utils import np_utils
from keras.models import model_from_json
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential,Model
from keras.models import load_model
from keras.layers import Dense,Dropout,Flatten, Conv1D,ZeroPadding1D,Activation,Input,Flatten,Conv2D,SeparableConv2D,Reshape,Multiply,add,Lambda,SpatialDropout2D,Conv2DTranspose
from keras.layers.merge import concatenate
from keras.layers.convolutional import MaxPooling1D,AveragePooling1D,MaxPooling2D,AveragePooling2D,UpSampling2D,Cropping2D
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from keras.optimizers import Adam, SGD
from keras.layers import add
from keras.layers.advanced_activations import PReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import GlobalAveragePooling1D,GlobalAveragePooling2D
import keras.backend as K
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import cv2
import sys
sys.setrecursionlimit(1000000)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)
shape_d = 512
val_acc = 0.


class CustomModelCheckpoint(keras.callbacks.Callback): 
	def __init__(self, model, path, **kwargs):
		super(CustomModelCheckpoint, self).__init__(**kwargs)
		self.model = model 
		self.path = path 
		self.best_loss = np.inf

	def on_epoch_end(self, epoch, logs=None): 
		val_loss = logs['val_loss'] 
		if val_loss < self.best_loss: 
			self.model.save_weights(self.path, overwrite=True) 
			self.best_loss = val_loss


def se_block(x):
	SE = GlobalAveragePooling2D()(x)
	SE = Dense(int(int(x.shape[-1])/2), kernel_initializer='he_uniform',activation='relu')(SE)
	SE = Dense(int(x.shape[-1]), kernel_initializer='he_uniform',activation='sigmoid')(SE)
	SE = Reshape([1,1,int(x.shape[-1])])(SE)
	while SE.shape[1] < x.shape[1]:
		SE = UpSampling2D((3, 3))(SE)
	SE = Cropping2D(cropping=((int(SE.shape[1])-int(x.shape[1]), 0), ((int(SE.shape[2])-int(x.shape[2]), 0))))(SE)
	x = Multiply()([x,SE])

	return x


def conv_block(ip, nb_filter, bottleneck=False, weight_decay=1E-4):

	dropout_rate = 0.1
	x = BatchNormalization(axis=-1, gamma_regularizer=regularizers.l2(weight_decay),beta_regularizer=regularizers.l2(weight_decay))(ip)
	#x = se_block(x)
	x = Activation('relu')(x)
	#x = Dropout(0.5)(x)
	#x = SpatialDropout2D(dropout_rate)(x)

	if bottleneck:
		inter_channel = nb_filter * 4
		x = Conv2D(inter_channel, (1,1), kernel_initializer='he_uniform', padding='same', use_bias=False,kernel_regularizer=regularizers.l2(weight_decay))(x)
		x = BatchNormalization(axis=-1, gamma_regularizer=regularizers.l2(weight_decay),beta_regularizer=regularizers.l2(weight_decay))(x)
		x = se_block(x)
		x = Activation('relu')(x)
		#x = Dropout(0.5)(x)
		#x = SpatialDropout2D(dropout_rate)(x)


	#x1 = Conv2D(nb_filter, 1, dilation_rate=(1,1), kernel_initializer='he_uniform', padding='same', use_bias=False,kernel_regularizer=regularizers.l2(weight_decay))(x)
	x = Conv2D(nb_filter, 3, dilation_rate=(1,1), kernel_initializer='he_uniform', padding='same', use_bias=False,kernel_regularizer=regularizers.l2(weight_decay))(x)
	#x3 = Conv2D(nb_filter, 5, dilation_rate=(1,1), kernel_initializer='he_uniform', padding='same', use_bias=False,kernel_regularizer=regularizers.l2(weight_decay))(x)
	#x4 = Conv2D(nb_filter, 7, dilation_rate=(1,1), kernel_initializer='he_uniform', padding='same', use_bias=False,kernel_regularizer=regularizers.l2(weight_decay))(x)
	#x = SeparableConv2D(nb_filter, 3, dilation_rate=(1,1), kernel_initializer='he_uniform', padding='same', use_bias=False,kernel_regularizer=regularizers.l2(weight_decay))(x)
	#x2 = SeparableConv2D(nb_filter, 3, dilation_rate=(2,2), kernel_initializer='he_uniform', padding='same', use_bias=False,kernel_regularizer=regularizers.l2(weight_decay))(x)
	#x3 = SeparableConv2D(nb_filter, 3, dilation_rate=(3,3), kernel_initializer='he_uniform', padding='same', use_bias=False,kernel_regularizer=regularizers.l2(weight_decay))(x)


	'''
	x = concatenate([x1,x2,x3], axis=-1)
	x = BatchNormalization(axis=-1, gamma_regularizer=regularizers.l2(weight_decay),beta_regularizer=regularizers.l2(weight_decay))(x)
	#x = se_block(x)
	x = Activation('relu')(x)
	#x = Dropout(0.5)(x) #- best
	#x = SpatialDropout2D(dropout_rate)(x)
	x = Conv2D(nb_filter, 1, kernel_initializer='he_uniform', padding='same', use_bias=False, kernel_regularizer=regularizers.l2(weight_decay))(x)
	'''



	return x


def transition_block(ip, nb_filter, strides=[2,2], compression=1.0, weight_decay=1E-4):

	x = BatchNormalization(axis=-1, gamma_regularizer=regularizers.l2(weight_decay),beta_regularizer=regularizers.l2(weight_decay))(ip)
	x = se_block(x)
	x = Activation('relu')(x)
	x = Conv2D(int(nb_filter*compression), 3, kernel_initializer='he_uniform', padding='same', use_bias=False, kernel_regularizer=regularizers.l2(weight_decay),strides=[2,2])(x)
	return x,int(nb_filter * compression)


def transition_block_up(x, x_u, nb_filter, nb_filter_u, strides=[2, 2], compression=1.0, weight_decay=1E-4):

	x = Conv2DTranspose(nb_filter, 3, kernel_initializer='he_uniform', padding='same', use_bias=False, kernel_regularizer=regularizers.l2(weight_decay),strides=[2,2])(x)
	#x = UpSampling2D((2,2))(x)
	x = concatenate([x,x_u], axis=-1)
	x = BatchNormalization(axis=-1, gamma_regularizer=regularizers.l2(weight_decay),beta_regularizer=regularizers.l2(weight_decay))(x)
	x = Activation('relu')(x)
	x = Conv2D(int((nb_filter+nb_filter_u)*compression/2), 1, kernel_initializer='he_uniform', padding='same', use_bias=False, kernel_regularizer=regularizers.l2(weight_decay))(x)
	#print (nb_filter+nb_filter_u)*compression/2 , nb_filter, nb_filter_u
	return x,int((nb_filter+nb_filter_u)*compression/2)


def dense_block(x, nb_layers, nb_filter, growth_rate, weight_decay=1E-4, bottleneck=False):
	x_list = [x]
	for i in range(nb_layers):
		x = conv_block(x, growth_rate, bottleneck=bottleneck)
		x_list.append(x)
		x = concatenate(x_list, axis=-1)
		nb_filter += growth_rate

	return x, nb_filter


def get_model():
	try:
		model = load_model('./model0.h5')
		#print 'loaded model'
		#model = load_model('./des.h5')

	except IOError:


		nb_filter = -1
		growth_rate = 24
		compression = 0.5
		weight_decay = 1e-4
		nb_layers = 2
		global shape_d
		img_shape = shape_d

		if nb_filter <= 0:
			nb_filter = 4*growth_rate


		inputs = Input([img_shape,img_shape,31])
		x = inputs
		x = Conv2D(nb_filter, (7,7), kernel_initializer='he_uniform', padding='same',use_bias=False, kernel_regularizer=regularizers.l2(weight_decay),strides=[2,2])(x)
		x_256, nb_filter_256 = x, nb_filter

		x,nb_filter = transition_block(x, nb_filter, compression=compression)
		x,nb_filter = dense_block(x, 2, nb_filter, growth_rate, bottleneck=True)
		x_128, nb_filter_128 = x, nb_filter

		x,nb_filter = transition_block(x, nb_filter, compression=compression)
		x,nb_filter = dense_block(x, 4, nb_filter, growth_rate, bottleneck=True)
		x_64, nb_filter_64 = x, nb_filter

		x,nb_filter = transition_block(x, nb_filter, compression=compression)
		x,nb_filter = dense_block(x, 4, nb_filter, growth_rate, bottleneck=True)
		x_32, nb_filter_32 = x, nb_filter

		x,nb_filter = transition_block(x, nb_filter, compression=compression)
		x,nb_filter = dense_block(x, 6, nb_filter, growth_rate, bottleneck=True)
		x_16, nb_filter_16 = x, nb_filter

		x,nb_filter = transition_block(x, nb_filter, compression=compression)
		x,nb_filter = dense_block(x, 8, nb_filter, growth_rate, bottleneck=True)


		x,nb_filter = transition_block_up(x,x_16,nb_filter,nb_filter_16, compression=compression)
		x,nb_filter = dense_block(x, 6, nb_filter, growth_rate, bottleneck=True)

		x,nb_filter = transition_block_up(x,x_32,nb_filter,nb_filter_32, compression=compression)
		x,nb_filter = dense_block(x, 4, nb_filter, growth_rate, bottleneck=True)

		x,nb_filter = transition_block_up(x,x_64,nb_filter,nb_filter_64, compression=compression)
		x,nb_filter = dense_block(x, 4, nb_filter, growth_rate, bottleneck=True)

		x,nb_filter = transition_block_up(x,x_128,nb_filter,nb_filter_128, compression=compression)
		x,nb_filter = dense_block(x, 2, nb_filter, growth_rate, bottleneck=True)

		x,nb_filter = transition_block_up(x,x_256,nb_filter,nb_filter_256, compression=compression)
		x,nb_filter = dense_block(x, 2, nb_filter, growth_rate, bottleneck=True)


		x = BatchNormalization(axis=-1, gamma_regularizer=regularizers.l2(weight_decay),beta_regularizer=regularizers.l2(weight_decay))(x)
		x = Activation('relu')(x)
		#x = UpSampling2D((2,2))(x)
		x = Conv2DTranspose(nb_filter, 3, kernel_initializer='he_uniform', padding='same', kernel_regularizer=regularizers.l2(weight_decay),strides=[2,2])(x)
		x = Conv2D(int(nb_filter*compression), (3,3), kernel_initializer='he_uniform', padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
		x = BatchNormalization(axis=-1, gamma_regularizer=regularizers.l2(weight_decay),beta_regularizer=regularizers.l2(weight_decay))(x)
		x = Activation('relu')(x)
		x = Conv2D(int(nb_filter*compression), (3,3), kernel_initializer='he_uniform', padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
		x = BatchNormalization(axis=-1, gamma_regularizer=regularizers.l2(weight_decay),beta_regularizer=regularizers.l2(weight_decay))(x)
		x = Activation('relu')(x)
		#x = SpatialDropout2D(0.1)(x)


		x1 = Conv2D(1, (1,1), kernel_initializer='he_uniform', padding='same', kernel_regularizer=regularizers.l2(weight_decay),name='1',activation='relu')(x)
		x2 = Conv2D(1, (1,1), kernel_initializer='he_uniform', padding='same', kernel_regularizer=regularizers.l2(weight_decay),name='2')(x)
		x3 = Conv2D(1, (1,1), kernel_initializer='he_uniform', padding='same', kernel_regularizer=regularizers.l2(weight_decay),name='3',activation='sigmoid')(x)
		x4 = Conv2D(1, (1,1), kernel_initializer='he_uniform', padding='same',use_bias=False, kernel_regularizer=regularizers.l2(weight_decay),name='4')(x)
		x5 = Conv2D(1, (1,1), kernel_initializer='he_uniform', padding='same',use_bias=False, kernel_regularizer=regularizers.l2(weight_decay),name='5')(x)
		x6 = Conv2D(1, (1,1), kernel_initializer='he_uniform', padding='same',use_bias=False, kernel_regularizer=regularizers.l2(weight_decay),name='6')(x)




		'''
		nb_filter = 512
		x = Conv2D(nb_filter, (1,1), kernel_initializer='he_uniform', padding='same', use_bias=False, kernel_regularizer=regularizers.l2(weight_decay))(x)
		x, nb_filter = dense_block(x, 8, nb_filter, growth_rate, bottleneck=True)
		#x, nb_filter = transition_block2(x, nb_filter, strides=[1,1],compression=compression)
		#x, nb_filter = dense_block(x, 8, nb_filter, growth_rate, bottleneck=True)
		x = BatchNormalization(axis=-1, gamma_regularizer=regularizers.l2(weight_decay),beta_regularizer=regularizers.l2(weight_decay))(x)
		x = Activation('relu')(x)
		'''



		model = Model(inputs, [x1,x2,x3,x4,x5,x6])
	return model


def res_block(ip, nb_filter, weight_decay=1e-4):
    x = Conv2D(int(nb_filter*0.5), 1, kernel_initializer='he_uniform', padding='same',)(ip)
    # x = BatchNormalization(axis=-1, gamma_regularizer=regularizers.l2(weight_decay),beta_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter, 3, kernel_initializer='he_uniform', padding='same', )(x)
    # x = Conv2D(int(nb_filter*0.75), 3, kernel_initializer='he_uniform', padding='same', use_bias=False,kernel_regularizer=regularizers.l2(weight_decay))(x)
    # x = BatchNormalization(axis=-1, gamma_regularizer=regularizers.l2(weight_decay),beta_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter, 1, kernel_initializer='he_uniform', padding='same', )(x)

    x = add([x, ip])
    # x = BatchNormalization(axis=-1, gamma_regularizer=regularizers.l2(weight_decay),beta_regularizer=regularizers.l2(weight_decay))(x)
    x = se_block(x)
    x = Activation('relu')(x)
    return x


def res_up(x, nb_filter, x_u, nb_filter_u, weight_decay=1e-4):
    x = Conv2DTranspose(nb_filter, 3, kernel_initializer='he_uniform', padding='same', strides=[2,2])(x)
    #x = BatchNormalization(axis=-1, gamma_regularizer=regularizers.l2(weight_decay),beta_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = concatenate([x,x_u], axis=-1)
    x = Conv2D(int((nb_filter+nb_filter_u)/2), 1, kernel_initializer='he_uniform', padding='same',  )(x)
    #x = BatchNormalization(axis=-1, gamma_regularizer=regularizers.l2(weight_decay),beta_regularizer=regularizers.l2(weight_decay))(x)
    x = se_block(x)
    x = Activation('relu')(x)

    return x,int((nb_filter+nb_filter_u)/2)


def get_model2():
    try:
        model = load_model('./model0.h5')
    except IOError:
        nb_filter = 24
        growth_rate = 24
        compression = 0.5
        weight_decay = 1e-4
        global shape_d
        img_shape = shape_d

        inputs = Input([img_shape,img_shape,31])
        x = inputs
        x = SeparableConv2D(nb_filter, 3, kernel_initializer='he_uniform', padding='same',strides=[2, 2])(x)

        for i in range(1):
            x = res_block(x,nb_filter)
        x_256, nb_filter_256 = x, nb_filter
        nb_filter *= 2
        x = Conv2D(nb_filter, 3, kernel_initializer='he_uniform', padding='same', strides=[2,2])(x)


        for i in range(2):
            x = res_block(x,nb_filter)
        x_128, nb_filter_128 = x, nb_filter
        nb_filter *= 2
        x = Conv2D(nb_filter, 3, kernel_initializer='he_uniform', padding='same', strides=[2,2])(x)


        for i in range(2):
            x = res_block(x,nb_filter)
        x_64, nb_filter_64 = x, nb_filter
        nb_filter *= 2
        x = Conv2D(nb_filter, 3, kernel_initializer='he_uniform', padding='same', strides=[2,2])(x)

        for i in range(4):
            x = res_block(x,nb_filter)

        x,nb_filter = res_up(x,nb_filter,x_64, nb_filter_64)
        for i in range(2):
            x = res_block(x,nb_filter)

        x,nb_filter = res_up(x,nb_filter,x_128, nb_filter_128)
        for i in range(2):
            x = res_block(x,nb_filter)

        x,nb_filter = res_up(x,nb_filter,x_256, nb_filter_256)
        for i in range(1):
            x = res_block(x,nb_filter)

        x = Conv2DTranspose(nb_filter, 3, kernel_initializer='he_uniform', padding='same', strides=[2,2])(x)
        # x = BatchNormalization(axis=-1, gamma_regularizer=regularizers.l2(weight_decay),beta_regularizer=regularizers.l2(weight_decay))(x)
        x = Activation('relu')(x)

        x = Conv2D(int(nb_filter*0.5), 3, kernel_initializer='he_uniform', padding='same',)(x)
        # x = Conv2D(int(nb_filter*0.5), (3,3), kernel_initializer='he_uniform', padding='same', use_bias=False,kernel_regularizer=regularizers.l2(weight_decay))(x)
        # x = BatchNormalization(axis=-1, gamma_regularizer=regularizers.l2(weight_decay),beta_regularizer=regularizers.l2(weight_decay))(x)
        x = Activation('relu')(x)

        x = Conv2D(int(nb_filter*0.5), 3, kernel_initializer='he_uniform', padding='same',)(x)
        # x = Conv2D(int(nb_filter*0.5), (3,3), kernel_initializer='he_uniform', padding='same', use_bias=False,kernel_regularizer=regularizers.l2(weight_decay))(x)
        # x = BatchNormalization(axis=-1, gamma_regularizer=regularizers.l2(weight_decay),beta_regularizer=regularizers.l2(weight_decay))(x)
        x = Activation('relu')(x)

        x1 = Conv2D(1, 3, kernel_initializer='he_uniform', padding='same', name='1',activation='relu')(x)
        x2 = Conv2D(1, 3, kernel_initializer='he_uniform', padding='same', name='2',activation='relu')(x)
        x3 = Conv2D(1, 3, kernel_initializer='he_uniform', padding='same', name='3',activation='relu')(x)
        x4 = Conv2D(1, 3, kernel_initializer='he_uniform', padding='same', name='4',activation='relu')(x)
        x5 = Conv2D(1, 3, kernel_initializer='he_uniform', padding='same', name='5',activation='relu')(x)
        x6 = Conv2D(1, 3, kernel_initializer='he_uniform', padding='same', name='6',activation='relu')(x)

        model = Model(inputs, [x1,x2,x3,x4,x5,x6])
    return model


model = get_model2()
model.summary()
# model = multi_gpu_model(model, gpus=2)
# plot_model(model, to_file='model.png',show_shapes=True)


def Metrics_datagen(path='./val.npy', size=1):
	data=np.load(path)
	global shape_d
	i = 0
	while True:
		X,Y = [],[]
		while len(X) < size:
			img = cv2.imread(data[i,0])
			if img.shape!=(shape_d,shape_d,3):
				img = cv2.resize(img,(shape_d,shape_d),interpolation=cv2.INTER_AREA)
			X.append(img)

			i += 1
			if i >= len(data):
				i -= len(data)
		yield np.array(X)/255.


class Metrics(Callback):
	def on_epoch_end(self, batch, logs={}):
		data=np.load('./val.npy')
		global val_acc
		pre = self.model.predict_generator(Metrics_datagen(),steps = len(data))
		real = np.array(data[:,-1],dtype=int)
		pre = np.argmax(pre,axis=-1)
		acc = len(pre[pre==real])/float(len(pre))
		print ('%.3f'%(acc*100))
		if acc >= val_acc:
			val_acc = acc
			weight = self.model.get_weights()
			np.save('./des_weight.npy', weight)
			json_string = np.array(model.to_json())
			np.save('./des_struct.npy',json_string)
		return


datagen = ImageDataGenerator(
			rotation_range=8,
			width_shift_range=0.1,
			height_shift_range=0.1,
			shear_range=0.0,
			zoom_range=0.0,         # resize : 0.05 or [0.05,0.5]
			fill_mode='nearest',
			channel_shift_range=10.  # color change
			)


# data generator
class threadsafe_iter:
	def __init__(self, it):
		self.it = it
		self.lock = threading.Lock()

	def __iter__(self):
		return self

	def __next__(self):
		with self.lock:
			return self.it.__next__()


def threadsafe_generator(f):
	def g(*a, **kw):
		return threadsafe_iter(f(*a, **kw))
	return g


@threadsafe_generator
def my_datagen(path, size=16):
	data = np.load(path)
	np.random.shuffle(data)
	global shape_d
	zd_size = 96
	N = 0
	while True:
		X, Y = [], []
		while len(X) < size:

			x = []
			for i in range(31):
				img = Image.open('./alldata/'+data[N]+'/'+data[N]+'_'+'%03d'%i+'.png')
				img = img.resize((512,512),Image.ANTIALIAS)
				img = np.array(img, dtype=float)
				img[img == 255] = 0.
				img /= 80.
				x.append(img)
			x = np.array(x)
			y = []
			for i in range(35, 60+1, 5):
				img1 = Image.open('./alldata/'+data[N]+'/'+data[N]+'_'+'%03d'%(i-2)+'.png')
				img2 = Image.open('./alldata/'+data[N]+'/'+data[N]+'_'+'%03d'%(i-1)+'.png')
				img3 = Image.open('./alldata/'+data[N]+'/'+data[N]+'_'+'%03d'%(i+0)+'.png')
				img1 = np.array(img1.resize((512, 512), Image.ANTIALIAS))
				img2 = np.array(img2.resize((512, 512), Image.ANTIALIAS))
				img3 = np.array(img3.resize((512, 512), Image.ANTIALIAS))
				img = (img1 + img2 + img3)/3.
				img[img > 80] = 0.
				img /= 80.
				y.append(img)
			N += 1
			if N >= len(data):
				N -= len(data)
				np.random.shuffle(data)

			X.append(x)
			Y.append(y)
		X = np.array(X)
		Y = np.array(Y)
		X = np.rollaxis(X, 1, 4)
		Y = np.rollaxis(Y, 1, 4)
		# print X.shape,Y.shape
		yield X, [Y[:, :, :, i].reshape([-1, 512, 512, 1]) for i in range(6)]


nb_data = 4500
size = 6
nb = 24
metrics = Metrics()


def my_mse_2(y_true, y_pred):
	return K.mean(K.square(y_pred**3 - y_true**3), axis=-1)


lr = 1e-6
print('------------------------------------------ ')
print(' train lr = ', str(lr))
adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=adam, loss=my_mse_2,)

model.fit_generator(
	my_datagen('./tra.npy', size),
	steps_per_epoch=nb_data/size,
	validation_data=my_datagen('./val.npy', size=size),
	validation_steps=500/size, epochs=10,
	callbacks=[ModelCheckpoint('./model0.h5', monitor='val_loss', save_best_only=True)],
	workers=4, verbose=1)





