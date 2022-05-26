# -*- coding: utf-8 -*-
"""
Created on Thu May 28 19:55:54 2020

@author: nagsa
"""
import os
import numpy as np
import matplotlib.image as mpimg
from PIL import Image
import cv2
import matplotlib.pyplot as plt

WORKING_DIR = 'C:\\Users\\nagsa\\OneDrive\\Documents\\PhD_at_PennState\\Google_deepLDB2\\IEEE_JOURNAL'
os.chdir(WORKING_DIR)

CODE_DIR = os.path.join(WORKING_DIR, 'code')
DATA_DIR = os.path.join(WORKING_DIR, 'data2')
MODEL_DIR = os.path.join(WORKING_DIR, 'models')
OUTPUT_DIR = os.path.join(WORKING_DIR, 'outputs')
PRED_DIR= os.path.join(WORKING_DIR, 'predictions')

ecoregions = ['EASTERN_TEMPERATE_FORESTS', 'MARINE_WEST_COAST_FOREST',
              'NORTH_AMERICAN_DESERTS', 'NORTHWESTERN_FORESTED_MOUNTAINS']

ETF_DIR = os.path.join(DATA_DIR, ecoregions[0])
MWCF_DIR = os.path.join(DATA_DIR, ecoregions[1])
NAD_DIR = os.path.join(DATA_DIR, ecoregions[2])
NWFM_DIR = os.path.join(DATA_DIR, ecoregions[3])

folders = ['POST_IMAGES', 'POST_LABELS', 'PRE_IMAGES', 'PRE_LABELS']

TRAIN1_DIR = os.path.join(WORKING_DIR, 'TRAIN1')
TRAIN2_DIR = os.path.join(WORKING_DIR, 'TRAIN2')
TRAIN3_DIR = os.path.join(WORKING_DIR, 'TRAIN3')
TRAIN4_DIR = os.path.join(WORKING_DIR, 'TRAIN4')

TEST1_DIR = os.path.join(WORKING_DIR, 'TEST1')
TEST2_DIR = os.path.join(WORKING_DIR, 'TEST2')
TEST3_DIR = os.path.join(WORKING_DIR, 'TEST3')
TEST4_DIR = os.path.join(WORKING_DIR, 'TEST4')

TEST_GLOBAL = os.path.join(WORKING_DIR, 'TEST_GLOBAL')


# for i in range(1, len(ecoregions)+1):
#     w = []
#     h = []
#     folder = os.path.join(DATA_DIR, 'TEST{}'.format(i), 'POST_IMAGES')
#     files = sorted(os.listdir(folder))
#     for f in files:
#         im = Image.open(os.path.join(folder, f))
#         im = np.array(im)
#         h_, w_, _ = im.shape
#         w.append(w_)
#         h.append(h_)
#     w_mean = np.mean(w)
#     h_mean = np.mean(h)
#     print('TEST{}_DIR: '.format(i), w_mean, h_mean)

w_resize = 1200
h_resize = 800

for i in range(1, len(ecoregions)+1):
    folder1 = os.path.join(DATA_DIR, 'TEST{}'.format(i), 'POST_IMAGES')
    folder2 = os.path.join(DATA_DIR, 'TEST{}'.format(i), 'PRE_IMAGES')
    folder3 = os.path.join(DATA_DIR, 'TEST{}'.format(i), 'POST_LABELS')
    
    folder5 = os.path.join(DATA_DIR, 'TRAIN{}'.format(i), 'POST_IMAGES')
    folder6 = os.path.join(DATA_DIR, 'TRAIN{}'.format(i), 'PRE_IMAGES')
    folder7 = os.path.join(DATA_DIR, 'TRAIN{}'.format(i), 'POST_LABELS')
    
    files1 = sorted(os.listdir(folder1))
    files2 = sorted(os.listdir(folder2))
    files3 = sorted(os.listdir(folder3))
    
    files5 = sorted(os.listdir(folder5))
    files6 = sorted(os.listdir(folder6))
    files7 = sorted(os.listdir(folder7))
    
    
    # for idx, f in enumerate(files1):
    #     if f == 'desktop.ini':
    #         os.remove(os.path.join(folder1, f))
    #         continue
    #     im = Image.open(os.path.join(folder1, f))
    #     im = np.array(im)
    #     im = cv2.resize(im, (w_resize, h_resize), interpolation = cv2.INTER_AREA)
    #     mpimg.imsave(os.path.join(folder1, f), im)
    #     print(idx)
        
    # for idx, f in enumerate(files2):
    #     if f == 'desktop.ini':
    #         os.remove(os.path.join(folder2, f))
    #         continue
    #     im = Image.open(os.path.join(folder2, f))
    #     im = np.array(im)
    #     im = cv2.resize(im, (w_resize, h_resize), interpolation = cv2.INTER_AREA)
    #     mpimg.imsave(os.path.join(folder2, f), im)
    #     print(idx)
        
    for idx, f in enumerate(files3):
        if f == 'desktop.ini':
            os.remove(os.path.join(folder3, f))
            continue
        im = cv2.imread(os.path.join(folder3, f), 0)
        im = np.array(im)
        im = im // 38
        im = cv2.resize(im, (w_resize, h_resize), interpolation = cv2.INTER_AREA)
        print(np.unique(im))
        cv2.imwrite(os.path.join(folder3, f[:-4]+'.png'), im)
        print(idx)
            
    
        
    # for idx, f in enumerate(files5):
    #     if f == 'desktop.ini':
    #         os.remove(os.path.join(folder5, f))
    #         continue
    #     im = Image.open(os.path.join(folder5, f))
    #     im = np.array(im)
    #     im = cv2.resize(im, (w_resize, h_resize), interpolation = cv2.INTER_AREA)
    #     mpimg.imsave(os.path.join(folder5, f), im)
    #     print(idx)
        
    # for idx, f in enumerate(files6):
    #     if f == 'desktop.ini':
    #         os.remove(os.path.join(folder6, f))
    #         continue
    #     im = Image.open(os.path.join(folder6, f))
    #     im = np.array(im)
    #     im = cv2.resize(im, (w_resize, h_resize), interpolation = cv2.INTER_AREA)
    #     mpimg.imsave(os.path.join(folder6, f), im)
    #     print(idx)
        
    for idx, f in enumerate(files7):
        if f == 'desktop.ini':
            os.remove(os.path.join(folder7, f))
            continue
        im = cv2.imread(os.path.join(folder7, f), 0)
        im = np.array(im)
        im = im // 38
        im = cv2.resize(im, (w_resize, h_resize), interpolation = cv2.INTER_AREA)
        print(np.unique(im))
        cv2.imwrite(os.path.join(folder7, f[:-4]+'.png'), im)
        print(idx)
        
    
folder9 = os.path.join(DATA_DIR, 'TEST_GLOBAL', 'POST_IMAGES')    
folder10 = os.path.join(DATA_DIR, 'TEST_GLOBAL', 'PRE_IMAGES')    
folder11 = os.path.join(DATA_DIR, 'TEST_GLOBAL', 'POST_LABELS')    

files9 = sorted(os.listdir(folder9))
files10 = sorted(os.listdir(folder10))
files11 = sorted(os.listdir(folder11))

# for idx, f in enumerate(files9):
#     if f == 'desktop.ini':
#         os.remove(os.path.join(folder9, f))
#         continue
#     im = Image.open(os.path.join(folder9, f))
#     im = np.array(im)
#     im = cv2.resize(im, (w_resize, h_resize), interpolation = cv2.INTER_AREA)
#     mpimg.imsave(os.path.join(folder9, f), im)
#     print(idx)
    
# for idx, f in enumerate(files10):
#     if f == 'desktop.ini':
#         os.remove(os.path.join(folder10, f))
#         continue
#     im = Image.open(os.path.join(folder10, f))
#     im = np.array(im)
#     im = cv2.resize(im, (w_resize, h_resize), interpolation = cv2.INTER_AREA)
#     mpimg.imsave(os.path.join(folder10, f), im)
#     print(idx)
    
for idx, f in enumerate(files11):
        if f == 'desktop.ini':
            os.remove(os.path.join(folder11, f))
            continue
        im = cv2.imread(os.path.join(folder11, f), 0)
        im = np.array(im)
        im = im // 38
        im = cv2.resize(im, (w_resize, h_resize), interpolation = cv2.INTER_AREA)
        print(np.unique(im))
        cv2.imwrite(os.path.join(folder11, f[:-4]+'.png'), im)
        print(idx)
    


