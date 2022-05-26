# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 13:39:58 2020

@author: nagsa
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from PIL import Image
import cv2
import matplotlib.image as mpimg

WORKING_DIR = 'C:\\Users\\nagsa\\OneDrive\\Documents\\PhD_at_PennState\\Google_deepLDB2\\IEEE_JOURNAL'

os.chdir(WORKING_DIR)
DATA_DIR = os.path.join(WORKING_DIR, 'data2')

folders = ['POST_IMAGES', 'POST_LABELS', 'PRE_IMAGES']

ecoregions = ['EASTERN_TEMPERATE_FORESTS_AUG', 'MARINE_WEST_COAST_FOREST_AUG',
              'NORTH_AMERICAN_DESERTS_AUG', 'NORTHWESTERN_FORESTED_MOUNTAINS_AUG']

aug_times = [15, 13, 11]

idx = 2

eco = ecoregions[idx]
augX = aug_times[idx]


post_folder = os.path.join(DATA_DIR, eco, folders[0])
pre_folder = os.path.join(DATA_DIR, eco, folders[2])
label_folder = os.path.join(DATA_DIR, eco, folders[1])

post_files = sorted(os.listdir(post_folder))
pre_files = sorted(os.listdir(pre_folder))
label_files = sorted(os.listdir(label_folder))

# Augmentations
rotated_imgs_post = []
rotated_labels_post = []
rotated_imgs_pre = []

flipped_imgs_post = []
flipped_imgs_pre = []
flipped_labels_post = []

gauss_imgs_post = []
gauss_imgs_pre = []
gauss_labels_post = []

for i in range(len(post_files)):
    print(i)
    if post_files[i] == 'desktop.ini':
        os.remove(os.path.join(post_folder, post_files[i]))
        continue
    if pre_files[i] == 'desktop.ini':
        os.remove(os.path.join(pre_folder, pre_files[i]))
        continue
    if label_files[i] == 'desktop.ini':
        os.remove(os.path.join(label_folder, label_files[i]))
        continue
    
    image1 = Image.open(os.path.join(post_folder, post_files[i]))
    image2 = Image.open(os.path.join(pre_folder, pre_files[i]))
    image3 = cv2.imread(os.path.join(label_folder, label_files[i]), 0)
    print(np.unique(image3))
    image3 = image3 // 38
    print(np.unique(image3))
    image1 = np.array(image1)
    image2 = np.array(image2)
    image3 = np.array(image3)
    
    mpimg.imsave(os.path.join(post_folder, post_files[i]), image1)
    mpimg.imsave(os.path.join(pre_folder, pre_files[i]), image2)
    cv2.imwrite(os.path.join(label_folder, label_files[i][:-4]+'.png'), image3)
    os.remove(os.path.join(label_folder, label_files[i]))
    
    for j in range(5):
        
        rotate=iaa.Affine(rotate=(-20, 20))
        rotate = rotate.to_deterministic()
    
        rotated_image1=rotate.augment_image(image1)
        rotated_image2 = rotate.augment_image(image2)
        rotated_image3=rotate.augment_image(image3)
        
        rotated_imgs_post.append(rotated_image1)
        rotated_imgs_pre.append(rotated_image2)
        rotated_labels_post.append(rotated_image3)
        print(np.unique(rotated_image3))
    
        mpimg.imsave(os.path.join(post_folder, post_files[i][:-4]+'_00' + str(j) + '_rot'+ '.jpg'), rotated_image1)
        mpimg.imsave(os.path.join(pre_folder, pre_files[i][:-4]+'_00' + str(j) + '_rot' + '.jpg'), rotated_image2)
        cv2.imwrite(os.path.join(label_folder, label_files[i][:-4]+'_00' + str(j) + '_rot'+ '.png'), rotated_image3)
        
    
        
    flip=iaa.Fliplr(1.0)
    flip = flip.to_deterministic()

    flipped_image1=flip.augment_image(image1)
    flipped_image2 = flip.augment_image(image2)
    flipped_image3=flip.augment_image(image3)
    
    flipped_imgs_post.append(flipped_image1)
    flipped_imgs_pre.append(flipped_image2)
    flipped_labels_post.append(flipped_image3)
    print(np.unique(flipped_image3))

    mpimg.imsave(os.path.join(post_folder, post_files[i][:-4]+'_00' + str(j)+'_flip' + '.jpg'), flipped_image1)
    mpimg.imsave(os.path.join(pre_folder, pre_files[i][:-4]+'_00' + str(j) + '_flip' + '.jpg'), flipped_image2)
    cv2.imwrite(os.path.join(label_folder, label_files[i][:-4]+'_00' + str(j) + '_flip'+'.png'), flipped_image3)
        
    for j in range(5):
        
        gauss=iaa.GaussianBlur(sigma = (0.2, 2))
        gauss = gauss.to_deterministic()
    
        gauss_image1=gauss.augment_image(image1)
        gauss_image2 = gauss.augment_image(image2)
        gauss_image3=image3
        
        gauss_imgs_post.append(gauss_image1)
        gauss_imgs_pre.append(gauss_image2)
        gauss_labels_post.append(gauss_image3)
        print(np.unique(gauss_image3))
    
        mpimg.imsave(os.path.join(post_folder, post_files[i][:-4]+'_00' + str(j)+'_gauss' + '.jpg'), gauss_image1)
        mpimg.imsave(os.path.join(pre_folder, pre_files[i][:-4]+'_00' + str(j) + '_gauss' + '.jpg'), gauss_image2)
        cv2.imwrite(os.path.join(label_folder, label_files[i][:-4]+'_00' + str(j) + '_gauss'+'.png'), gauss_image3)


idx = 3
eco = ecoregions[idx]


post_folder = os.path.join(DATA_DIR, eco, folders[0])
pre_folder = os.path.join(DATA_DIR, eco, folders[2])
label_folder = os.path.join(DATA_DIR, eco, folders[1])

post_files = sorted(os.listdir(post_folder))
pre_files = sorted(os.listdir(pre_folder))
label_files = sorted(os.listdir(label_folder))

for i in range(len(post_files)):
    print(i)
    if post_files[i] == 'desktop.ini':
        os.remove(os.path.join(post_folder, post_files[i]))
        continue
    if pre_files[i] == 'desktop.ini':
        os.remove(os.path.join(pre_folder, pre_files[i]))
        continue
    if label_files[i] == 'desktop.ini':
        os.remove(os.path.join(label_folder, label_files[i]))
        continue
    
    image1 = Image.open(os.path.join(post_folder, post_files[i]))
    image2 = Image.open(os.path.join(pre_folder, pre_files[i]))
    image3 = cv2.imread(os.path.join(label_folder, label_files[i]), 0)
    print(np.unique(image3))
    image3 = image3 // 38
    print(np.unique(image3))
    image1 = np.array(image1)
    image2 = np.array(image2)
    image3 = np.array(image3)
    
    mpimg.imsave(os.path.join(post_folder, post_files[i]), image1)
    mpimg.imsave(os.path.join(pre_folder, pre_files[i]), image2)
    cv2.imwrite(os.path.join(label_folder, label_files[i][:-4]+'.png'), image3)
    os.remove(os.path.join(label_folder, label_files[i]))