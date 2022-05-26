#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 02:43:44 2020

@author: sxn265
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 22:10:10 2020

@author: sxn265
"""


import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import optimizers
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, Activation, Flatten, Dense
from tensorflow.keras.activations import sigmoid
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Add, SeparableConv2D, GlobalAveragePooling2D
from tensorflow.keras.models import load_model
import keras.backend as K



WORKING_DIR = os.getcwd()

os.chdir(WORKING_DIR)
DATA_DIR = os.getcwd()

folders = ['POST_IMAGES', 'POST_LABELS', 'PRE_IMAGES', 'PRE_LABELS']


IMAGE_ORDERING =  "channels_last"


img_input = Input(shape = (512, 512, 6), name = 'image_input')

def resnet_block(x, filters, block_num, trainable=True):

    input_ = Conv2D(filters*2, (1, 1), padding='same', 
               name='c0'+block_num, data_format=IMAGE_ORDERING, trainable=trainable)(x)
    
    x = Conv2D(filters, (1, 1), padding='same', 
               name='c1'+block_num, data_format=IMAGE_ORDERING, trainable=trainable)(x)
    x = Activation('relu', name='a1'+block_num ,trainable=trainable)(x)
    
    x = Conv2D(filters, (3, 3), padding='same', 
               name='c2'+block_num, data_format=IMAGE_ORDERING, trainable=trainable)(x)
    x = Activation('relu', name='a2'+block_num ,trainable=trainable)(x)
    
    x = Conv2D(filters*2, (1, 1), padding='same', 
               name='c3'+block_num, data_format=IMAGE_ORDERING, trainable=trainable)(x)
    
    x = Add()([input_, x])
    
    x = BatchNormalization(name='b3'+block_num,trainable=trainable)(x)
    x = Activation('relu', name='a3'+block_num ,trainable=trainable)(x)
    
    return x

x = resnet_block(img_input, 8, '1_res')
x = resnet_block(x, 16, '2_res')
x = resnet_block(x, 32, '3_res')
x = resnet_block(x, 64, '4_res')
x = resnet_block(x, 128, '5_res')

res_out = x

## Block 1
x = Conv2D(8, (3, 3), padding='same', name='c1',data_format=IMAGE_ORDERING, trainable=True )(res_out)
x = BatchNormalization(name='b1',trainable=True)(x)
x = Activation('relu', name='a1',trainable=True)(x)

f1 = x
x = MaxPooling2D((2, 2), strides=(2, 2), name='c3', data_format=IMAGE_ORDERING, trainable=True )(x)
x = Dropout(0.05, name='d1',trainable=True)(x)

## Block 2
x = Conv2D(16, (3, 3), padding='same', name='c4',data_format=IMAGE_ORDERING, trainable=True )(x)
x = BatchNormalization(name='b3',trainable=True)(x)
x = Activation('relu', name='a2',trainable=True)(x)

f2 = x
x = MaxPooling2D((2, 2), strides=(2, 2), name='c6', data_format=IMAGE_ORDERING, trainable=True )(x)
x = Dropout(0.1, name='d2',trainable=True)(x)

## Block 3
x = Conv2D(32, (3, 3), padding='same', name='c7',data_format=IMAGE_ORDERING, trainable=True )(x)
x = BatchNormalization(name='b5',trainable=True)(x)
x = Activation('relu', name='a3',trainable=True)(x)

f3 = x
x = MaxPooling2D((2, 2), strides=(2, 2), name='c9', data_format=IMAGE_ORDERING, trainable=True )(x)
x = Dropout(0.1, name='d3',trainable=True)(x)

## Block 4
x = Conv2D(64, (3, 3), padding='same', name='c10',data_format=IMAGE_ORDERING, trainable=True )(x)
x = BatchNormalization(name='b7',trainable=True)(x)
x = Activation('relu',name='a4',trainable=True)(x)

f4 = x
x = MaxPooling2D((2, 2), strides=(2, 2), name='c12', data_format=IMAGE_ORDERING, trainable=True )(x)
x = Dropout(0.1, name='d4',trainable=True)(x)

## Block 5
x = Conv2D(128, (3, 3), padding='same', name='c13',data_format=IMAGE_ORDERING, trainable=True )(x)
x = BatchNormalization(name='b9',trainable=True)(x)
x = Activation('relu',name='a5',trainable=True)(x)

f5 = x
x = MaxPooling2D((2, 2), strides=(2, 2), name='c14', data_format=IMAGE_ORDERING, trainable=True )(x)
x = Dropout(0.1, name='d5',trainable=True)(x)

## Block 5
x = Conv2D(256, (3, 3), padding='same', name='c13_',data_format=IMAGE_ORDERING, trainable=True )(x)
x = BatchNormalization(name='b10',trainable=True)(x)
x = Activation('relu',name='a5_',trainable=True)(x)

f6 = x
x = MaxPooling2D((2, 2), strides=(2, 2), name='c14_', data_format=IMAGE_ORDERING, trainable=True )(x)
x = Dropout(0.1, name='d5_',trainable=True)(x)

## Block 5
x = Conv2D(512, (3, 3), padding='same', name='c13_1',data_format=IMAGE_ORDERING, trainable=True )(x)
x = BatchNormalization(name='b10_',trainable=True)(x)
x = Activation('relu',name='a5_1',trainable=True)(x)

f7 = x



## Block 6
# expansive path

u4 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', name='cd1_',trainable=True) (f7)
u4 = Concatenate(name='co1_',trainable=True)([u4, f6])
u4 = Dropout(0.1,name='d6_',trainable=True)(u4)
x = Conv2D(256, (3, 3), padding='same', name='c15_',data_format=IMAGE_ORDERING, trainable=True )(u4)
x = BatchNormalization(name='b11_',trainable=True)(x)
x = Activation('relu',name='a6_',trainable=True)(x)

f8 = x

u5 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', name='cd1_1',trainable=True) (f8)
u5 = Concatenate(name='co1_1',trainable=True)([u5, f5])
u5 = Dropout(0.1,name='d6_1',trainable=True)(u5)
x = Conv2D(128, (3, 3), padding='same', name='c15_1',data_format=IMAGE_ORDERING, trainable=True )(u5)
x = BatchNormalization(name='b11_1',trainable=True)(x)
x = Activation('relu',name='a6_1',trainable=True)(x)

f9 = x


u6 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', name='cd1',trainable=True) (f9)
u6 = Concatenate(name='co1',trainable=True)([u6, f4])
u6 = Dropout(0.1,name='d6',trainable=True)(u6)
x = Conv2D(64, (3, 3), padding='same', name='c15',data_format=IMAGE_ORDERING, trainable=True )(u6)
x = BatchNormalization(name='b11',trainable=True)(x)
x = Activation('relu',name='a6',trainable=True)(x)

f10 = x

## Block 7
# expansive path
u7 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',name='cd2',trainable=True) (f10)
u7 = Concatenate(name='co2',trainable=True)([u7, f3])
u7 = Dropout(0.1,name='d7',trainable=True)(u7)
x = Conv2D(32, (3, 3), padding='same', name='c17',data_format=IMAGE_ORDERING, trainable=True )(u7)
x = BatchNormalization(name='b13',trainable=True)(x)
x = Activation('relu',name='a7',trainable=True)(x)

f11 = x

## Block 8
# expansive path
u8 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same',name='cd3',trainable=True) (f11)
u8 = Concatenate(name='co3',trainable=True)([u8, f2])
u8 = Dropout(0.1,name='d8',trainable=True)(u8)
x = Conv2D(16, (3, 3), padding='same', name='c19',data_format=IMAGE_ORDERING, trainable=True )(u8)
x = BatchNormalization(name='b15',trainable=True)(x)
x = Activation('relu',name='a8',trainable=True)(x)

f12 = x

## Block 9
# expansive path
u9 = Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='same',name='cd4',trainable=True) (f12)
u9 = Concatenate(name='co4',trainable=True)([u9, f1])
u9 = Dropout(0.1,name='d9',trainable=True)(u9)
x = Conv2D(8, (3, 3), padding='same', name='c21',data_format=IMAGE_ORDERING, trainable=True )(u9)
x = BatchNormalization(name='b17',trainable=True)(x)
x = Activation('relu',name='a9',trainable=True)(x)

f13 = x

output1 = Conv2D(1, (1, 1), name='c23',trainable=True) (f13)
output1 = Activation('sigmoid', name = 'decoder',trainable=True)(output1)


# dedicated encoder-decoder

## Block 1
x = Conv2D(8, (3, 3), padding='same', name='2_c1',data_format=IMAGE_ORDERING, trainable=True )(res_out)
x = BatchNormalization(name='2_b1',trainable=True)(x)
x = Activation('relu', name='2_a1',trainable=True)(x)

g1 = x
x = MaxPooling2D((2, 2), strides=(2, 2), name='2_c3', data_format=IMAGE_ORDERING, trainable=True )(x)
x = Dropout(0.05, name='2_d1',trainable=True)(x)

## Block 2
x = Conv2D(16, (3, 3), padding='same', name='2_c4',data_format=IMAGE_ORDERING, trainable=True )(x)
x = BatchNormalization(name='2_b3',trainable=True)(x)
x = Activation('relu', name='2_a2',trainable=True)(x)

g2 = x
x = MaxPooling2D((2, 2), strides=(2, 2), name='2_c6', data_format=IMAGE_ORDERING, trainable=True )(x)
x = Dropout(0.1, name='2_d2',trainable=True)(x)

## Block 3
x = Conv2D(32, (3, 3), padding='same', name='2_c7',data_format=IMAGE_ORDERING, trainable=True )(x)
x = BatchNormalization(name='2_b5',trainable=True)(x)
x = Activation('relu', name='2_a3',trainable=True)(x)

g3 = x
x = MaxPooling2D((2, 2), strides=(2, 2), name='2_c9', data_format=IMAGE_ORDERING, trainable=True )(x)
x = Dropout(0.1, name='2_d3',trainable=True)(x)

## Block 4
x = Conv2D(64, (3, 3), padding='same', name='2_c10',data_format=IMAGE_ORDERING, trainable=True )(x)
x = BatchNormalization(name='2_b7',trainable=True)(x)
x = Activation('relu',name='2_a4',trainable=True)(x)

g4 = x
x = MaxPooling2D((2, 2), strides=(2, 2), name='2_c12', data_format=IMAGE_ORDERING, trainable=True )(x)
x = Dropout(0.1, name='2_d4',trainable=True)(x)

## Block 5
x = Conv2D(128, (3, 3), padding='same', name='2_c13',data_format=IMAGE_ORDERING, trainable=True )(x)
x = BatchNormalization(name='2_b9',trainable=True)(x)
x = Activation('relu',name='2_a5',trainable=True)(x)

g5 = x
x = MaxPooling2D((2, 2), strides=(2, 2), name='2_c14', data_format=IMAGE_ORDERING, trainable=True )(x)
x = Dropout(0.1, name='2_d5',trainable=True)(x)

## Block 5
x = Conv2D(256, (3, 3), padding='same', name='2_c13_',data_format=IMAGE_ORDERING, trainable=True )(x)
x = BatchNormalization(name='2_b10',trainable=True)(x)
x = Activation('relu',name='2_a5_',trainable=True)(x)

g6 = x
x = MaxPooling2D((2, 2), strides=(2, 2), name='2_c14_', data_format=IMAGE_ORDERING, trainable=True )(x)
x = Dropout(0.1, name='2_d5_',trainable=True)(x)

## Block 5
x = Conv2D(512, (3, 3), padding='same', name='2_c13_1',data_format=IMAGE_ORDERING, trainable=True )(x)
x = BatchNormalization(name='2_b10_',trainable=True)(x)
x = Activation('relu',name='2_a5_1',trainable=True)(x)

g7 = x



## Block 6
# expansive path

u4 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', name='2_cd1_',trainable=True) (g7)
u4 = Concatenate(name='2_co1_',trainable=True)([u4, g6])
u4 = Dropout(0.1,name='2_d6_',trainable=True)(u4)
x = Conv2D(256, (3, 3), padding='same', name='2_c15_',data_format=IMAGE_ORDERING, trainable=True )(u4)
x = BatchNormalization(name='2_b11_',trainable=True)(x)
x = Activation('relu',name='2_a6_',trainable=True)(x)

g8 = x

u5 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', name='2_cd1_1',trainable=True) (g8)
u5 = Concatenate(name='2_co1_1',trainable=True)([u5, g5])
u5 = Dropout(0.1,name='2_d6_1',trainable=True)(u5)
x = Conv2D(128, (3, 3), padding='same', name='2_c15_1',data_format=IMAGE_ORDERING, trainable=True )(u5)
x = BatchNormalization(name='2_b11_1',trainable=True)(x)
x = Activation('relu',name='2_a6_1',trainable=True)(x)

g9 = x


u6 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', name='2_cd1',trainable=True) (g9)
u6 = Concatenate(name='2_co1',trainable=True)([u6, g4])
u6 = Dropout(0.1,name='2_d6',trainable=True)(u6)
x = Conv2D(64, (3, 3), padding='same', name='2_c15',data_format=IMAGE_ORDERING, trainable=True )(u6)
x = BatchNormalization(name='2_b11',trainable=True)(x)
x = Activation('relu',name='2_a6',trainable=True)(x)

g10 = x

## Block 7
# expansive path
u7 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',name='2_cd2',trainable=True) (g10)
u7 = Concatenate(name='2_co2',trainable=True)([u7, g3])
u7 = Dropout(0.1,name='2_d7',trainable=True)(u7)
x = Conv2D(32, (3, 3), padding='same', name='2_c17',data_format=IMAGE_ORDERING, trainable=True )(u7)
x = BatchNormalization(name='2_b13',trainable=True)(x)
x = Activation('relu',name='2_a7',trainable=True)(x)

g11 = x

## Block 8
# expansive path
u8 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same',name='2_cd3',trainable=True) (g11)
u8 = Concatenate(name='2_co3',trainable=True)([u8, g2])
u8 = Dropout(0.1,name='2_d8',trainable=True)(u8)
x = Conv2D(16, (3, 3), padding='same', name='2_c19',data_format=IMAGE_ORDERING, trainable=True )(u8)
x = BatchNormalization(name='2_b15',trainable=True)(x)
x = Activation('relu',name='2_a8',trainable=True)(x)

g12 = x

## Block 9
# expansive path
u9 = Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='same',name='2_cd4',trainable=True) (g12)
u9 = Concatenate(name='2_co4',trainable=True)([u9, g1])
u9 = Dropout(0.1,name='2_d9',trainable=True)(u9)
x = Conv2D(8, (3, 3), padding='same', name='2_c21',data_format=IMAGE_ORDERING, trainable=True )(u9)
x = BatchNormalization(name='2_b17',trainable=True)(x)
x = Activation('relu',name='2_a9',trainable=True)(x)

g13 = x

output2 = Conv2D(1, (1, 1), name='2_c23',trainable=True) (g13)
output2 = Activation('sigmoid', name = 'decoder2',trainable=True)(output2)

model = Model(inputs=img_input, outputs=[output1, output2])
model.load_weights('w-exp-obo-1-58-0.51.hdf5', by_name=True)

model.summary()


j=3

folder1 = os.path.join(DATA_DIR, 'TEST{}'.format(j), 'POST_IMAGES')
folder2 = os.path.join(DATA_DIR, 'TEST{}'.format(j), 'PRE_IMAGES')
folder3 = os.path.join(DATA_DIR, 'TEST{}'.format(j), 'POST_LABELS')


folder13 = os.path.join(DATA_DIR, 'VAL{}_TEST'.format(j), 'POST_IMAGES')
folder14 = os.path.join(DATA_DIR, 'VAL{}_TEST'.format(j), 'PRE_IMAGES')
folder15 = os.path.join(DATA_DIR, 'VAL{}_TEST'.format(j), 'POST_LABELS')


files1 = sorted(os.listdir(folder1))
files2 = sorted(os.listdir(folder2))
files3 = sorted(os.listdir(folder3))

# files5 = sorted(os.listdir(folder5))
# files6 = sorted(os.listdir(folder6))
# files7 = sorted(os.listdir(folder7))

# files9 = sorted(os.listdir(folder9))
# files10 = sorted(os.listdir(folder10))
# files11 = sorted(os.listdir(folder11))

# files17 = sorted(os.listdir(folder17))
# files18 = sorted(os.listdir(folder18))
# files19 = sorted(os.listdir(folder19))

files13 = sorted(os.listdir(folder13))
files14 = sorted(os.listdir(folder14))
files15 = sorted(os.listdir(folder15))

train_post_imgs = []
train_pre_imgs = []
train_post_labels = []

val_post_imgs = []
val_pre_imgs = []
val_post_labels = []


# import random
# random.seed(7)
# files = random.sample(range(len(files1)), 35)
# print(files)

# files1 = np.array(files1)
# files2 = np.array(files2)
# files3 = np.array(files3)

# files1 = files1[files]
# files2 = files2[files]
# files3 = files3[files]


print('Loading Data')
for idx, f in enumerate(files1):
    if files1[idx] == 'desktop.ini':
        os.remove(os.path.join(folder1, f))
        continue
    if files2[idx] == 'desktop.ini':
        os.remove(os.path.join(folder2, f))
        continue
    if files3[idx] == 'desktop.ini':
        os.remove(os.path.join(folder3, f))
        continue
        
#    if idx not in val_files:
    im = Image.open(os.path.join(folder1, files1[idx]))
    im = np.array(im)
    im = im/ 255.0
    im = cv2.resize(im, (512, 512), interpolation=cv2.INTER_AREA)
    train_post_imgs.append(im)

    im = Image.open(os.path.join(folder2, files2[idx]))
    im = np.array(im)
    im = im/ 255.0
    im = cv2.resize(im, (512, 512), interpolation=cv2.INTER_AREA)
    train_pre_imgs.append(im)    

    im = cv2.imread(os.path.join(folder3, files3[idx]),0)
    im = np.array(im)
    if np.unique(im)[1] == 38:
        im = im // 38
    im = cv2.resize(im, (512, 512), interpolation=cv2.INTER_AREA)
    train_post_labels.append(im)

for idx, f in enumerate(files13):
    if files13[idx] == 'desktop.ini':
        os.remove(os.path.join(folder13, f))
        continue
    if files14[idx] == 'desktop.ini':
        os.remove(os.path.join(folder14, f))
        continue
    if files15[idx] == 'desktop.ini':
        os.remove(os.path.join(folder15, f))
        continue
        
    # if idx not in val_files:
    im = Image.open(os.path.join(folder13, files13[idx]))
    im = np.array(im)
    im = im/ 255.0
    im = cv2.resize(im, (512, 512), interpolation=cv2.INTER_AREA)
    val_post_imgs.append(im)

    im = Image.open(os.path.join(folder14, files14[idx]))
    im = np.array(im)
    im = im/ 255.0
    im = cv2.resize(im, (512, 512), interpolation=cv2.INTER_AREA)
    val_pre_imgs.append(im)    

    im = cv2.imread(os.path.join(folder15, files15[idx]),0)
    im = np.array(im)
    if np.unique(im)[1] == 38:
        im = im // 38
    im = cv2.resize(im, (512, 512), interpolation=cv2.INTER_AREA)
    val_post_labels.append(im)
    
    



train_post_imgs = np.array(train_post_imgs)
train_pre_imgs = np.array(train_pre_imgs)
train_post_labels = np.array(train_post_labels)

train_post_imgs = np.float16(train_post_imgs)
train_pre_imgs = np.float16(train_pre_imgs)

val_post_imgs = np.array(val_post_imgs)
val_pre_imgs = np.array(val_pre_imgs)
val_post_labels = np.array(val_post_labels)

val_post_imgs = np.float16(val_post_imgs)
val_pre_imgs = np.float16(val_pre_imgs)



print(train_post_imgs.shape)
print(train_pre_imgs.shape)
print(train_post_labels.shape)

print(val_post_imgs.shape)
print(val_pre_imgs.shape)
print(val_post_labels.shape)


print(np.unique(train_post_labels))
print(np.unique(val_post_labels))



print('Data Loaded')


val_imgs = np.concatenate((val_pre_imgs, val_post_imgs), axis = 3)
train_imgs = np.concatenate((train_pre_imgs, train_post_imgs), axis = 3)

print(train_imgs.shape)
print(val_imgs.shape)

pred_train, _ = model.predict(train_imgs, batch_size=4)
pred_val, _ = model.predict(val_imgs, batch_size=4)

def dice_coef(y_true, y_pred, smooth = 1):
   y_true_f = K.flatten(y_true)
   y_pred_f = K.flatten(y_pred)
   
   intersection = K.sum(y_true_f * y_pred_f)
   dice = (2. * intersection + smooth) / (K.sum(y_true_f)+K.sum(y_pred_f) + smooth)
   
   return dice
def soft_dice_loss(y_true, y_pred):
   return 1-dice_coef(y_true, y_pred)

def iou_metric(y_pred, y_true):
   I = tf.reduce_sum(y_pred * y_true, axis=(1, 2))
   U = tf.reduce_sum(y_pred + y_true, axis=(1, 2)) - I
   return tf.reduce_mean(I / U)



train_post_labels = train_post_labels.reshape(train_post_labels.shape[0], 
                                              train_post_labels.shape[1],
                                             train_post_labels.shape[2],
                                             1)

val_post_labels = val_post_labels.reshape(val_post_labels.shape[0], 
                                              val_post_labels.shape[1],
                                             val_post_labels.shape[2],
                                             1)
print(train_post_labels.shape)
print(val_post_labels.shape)


dependencies = {
    'iou_metric': iou_metric
}

# model.load_weights('w-exp21-170-0.44.hdf5')
# model.summary()

opt = tensorflow.keras.optimizers.Adam(lr=1e-12)
model.compile(optimizer=opt, 
                 loss = {'decoder':'mean_squared_error',
                         # 'decoder2':'binary_crossentropy'
                         'decoder2': soft_dice_loss
                         }, 
                 metrics = [iou_metric])

from keras.callbacks import ModelCheckpoint

# checkpoint
filepath="w-exp-obo-12-jo-{epoch:02d}-{val_decoder2_iou_metric:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_decoder2_iou_metric', verbose=1, 
                             save_best_only=True, mode='max')

tbCallBack  = keras.callbacks.TensorBoard(log_dir='./Graph_obo12-jo', histogram_freq=0,  
          write_graph=True, write_images=True)

callbacks_list = [checkpoint, tbCallBack]

print('BEGIN TRAINING')

history = model.fit(train_imgs,
                     {"decoder":pred_train,
                         "decoder2":train_post_labels
                     
                     }, 
                    validation_data = (val_imgs,  
                                       {"decoder":pred_val,
                                           "decoder2":val_post_labels
                                        
                                    }), 
                    batch_size=2, epochs=200, verbose=1, callbacks=callbacks_list)


print('END TRAINING')





