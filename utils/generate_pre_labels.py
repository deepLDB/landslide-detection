# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 14:17:28 2020

@author: nagsa
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

path_to_wd = 'C:\\Users\\nagsa\\OneDrive\\Documents\\PhD_at_PennState\\Google_deepLDB2\\deepLDB\\IEEE_JOURNAL'
WORKING_DIR = os.path.join(path_to_wd)

DATA_DIR = os.path.join(WORKING_DIR, 'data')
CODE_DIR = os.path.join(WORKING_DIR, 'code')

os.chdir(CODE_DIR)

ecoregions = ['EASTERN_TEMPERATE_FORESTS', 
              'MARINE_WEST_COAST_FOREST',
              'NORTH_AMERICAN_DESERTS',
              'NORTHWESTERN_FORESTED_MOUNTAINS']

folder_names = ['PRE_IMAGES',
                'PRE_LABELS']

for i in range(len(ecoregions)):

    path = os.path.join(DATA_DIR, ecoregions[i], folder_names[0])
    imgs = os.listdir(path)
    if 'desktop.ini' in imgs:
        imgs.remove('desktop.ini')
    for j in range(len(imgs)):    
        im = Image.open(os.path.join(path, imgs[j]))
        im = np.array(im)
        print(im.shape)
        h, w = im.shape[0], im.shape[1]
        label = np.zeros((h, w))
        cv2.imwrite(os.path.join(DATA_DIR, ecoregions[i], folder_names[1], imgs[j]), label)
        
        